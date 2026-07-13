"""
Perturb-Prediction Balanced KRR Pipeline
------------------------------------------
This script implements a balanced Kernel Ridge Regression (KRR) baseline for predicting
transcriptional responses to genetic perturbations in a target cellular context. Biological
knowledge priors and cross-context data priors are aggregated into separate kernel groups,
then combined with a tunable weight rather than pooling all sources into one MKL average.

Key Features:
1. Balanced Kernel Aggregation: Biological sources (embeddings, local PCA loadings) and
   cross-context data priors are averaged within each group, then fused as
   K_final = (1 - λ) · K_bio + λ · K_context.
2. Multi-Output KRR: Fits ridge regression in the dual space of the precomputed kernel and
   predicts full expression vectors for held-out perturbations.
3. Hyperparameter Search: Cross-validated tuning of alpha, kernel type (linear or RBF), and
   gamma; kernels are re-aggregated only when manifold parameters change.
4. Evaluation: Reports MSE and mean per-sample Pearson correlation on seen and unseen test splits.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from sklearn.decomposition import PCA
import pickle

# --- 1. Data Handling & Kernel Generation ---

class DataModule:
    def __init__(self, h5ad_path: str, target_screen: str, split_column: str = 'split_random'):
        self.h5ad_path = h5ad_path
        self.target_screen = target_screen
        self.split_column = split_column
        
        self.adata = None
        self.vocab_pert = {} # Target pert -> global index within unique perts
        
        # Sources are partitioned for balanced aggregation
        self.bio_sources = []     # Biological (KP + local PCA)
        self.context_sources = [] # Data Priors (DP)
        
        self.aggregated_kernel = None
        self.Y = None
        self.output_dim = 0
        
    def load_and_preprocess(self, 
                            embedding_paths: Optional[List[str]] = None,
                            use_data_prior: bool = True,
                            seed: int = 42):
        print(f"Loading data from {self.h5ad_path}...")
        full_adata = sc.read_h5ad(self.h5ad_path)
        
        # 1. Filter to Target Screen samples
        valid_splits = ['train', 'test_seen', 'test_unseen']
        ignored_names = {'control', 'non-targeting', 'Control', 'Non-targeting', 'excluded'}
        mask_target = (full_adata.obs['context'] == self.target_screen) & (full_adata.obs[self.split_column].isin(valid_splits))
        mask_target &= ~full_adata.obs['target_gene'].isin(ignored_names)
        
        self.adata = full_adata[mask_target].copy()
        print(f"Kept {len(self.adata)} samples from {self.target_screen} screen.")
        
        # Determine all unique perturbations in the target set
        all_unique_perts = sorted(self.adata.obs['target_gene'].unique())
        self.vocab_pert = {p: i for i, p in enumerate(all_unique_perts)}
        n_unique_perts = len(all_unique_perts)
        
        # 2. Source: Data Priors (Context Sources)
        if use_data_prior:
            print("Generating Kernels from Cross-Context Data Priors...")
            other_contexts = [c for c in full_adata.obs['context'].unique() if c != self.target_screen]
            
            context_masks = (full_adata.obs['context'].isin(other_contexts)) & (full_adata.obs[self.split_column].isin(valid_splits))
            adata_others = full_adata[context_masks].copy()

            for context in other_contexts:
                adata_c = adata_others[adata_others.obs['context'] == context]
                if adata_c.shape[0] == 0: continue
                
                pert_means = {}
                for pert in adata_c.obs['target_gene'].unique():
                    if pert in ignored_names: continue
                    m = adata_c[adata_c.obs['target_gene'] == pert].X
                    if hasattr(m, 'toarray'): m = m.toarray()
                    pert_means[pert] = np.mean(m, axis=0)
                
                if not pert_means: continue
                
                X_c = np.full((n_unique_perts, full_adata.shape[1]), np.nan)
                for pert, idx in self.vocab_pert.items():
                    if pert in pert_means:
                        X_c[idx] = pert_means[pert]
                
                valid_mask = ~np.isnan(X_c).all(axis=1)
                if valid_mask.any():
                    self.context_sources.append((X_c, f"DP_{context}"))

        # 3. Source: Knowledge Priors (Bio Sources)
        if embedding_paths:
            print(f"Generating Kernels from {len(embedding_paths)} Knowledge Sources...")
            for path in embedding_paths:
                if not os.path.exists(path): continue
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                df = data if isinstance(data, pd.DataFrame) else pd.DataFrame.from_dict(data, orient='index')
                
                X_k = np.full((n_unique_perts, df.shape[1]), np.nan)
                for pert, idx in self.vocab_pert.items():
                    if pert in df.index:
                        X_k[idx] = df.loc[pert].values
                
                valid_mask = ~np.isnan(X_k).all(axis=1)
                if valid_mask.any():
                    self.bio_sources.append((X_k, f"KP_{os.path.basename(path)}"))

        # 4. Source: Target Context PCA Factor Loadings (Bio Sources)
        loadings_pca_dim = 128
        print(f"Generating Kernel from Target Context PCA Loadings (dim={loadings_pca_dim})...")
        X_loadings = np.full((n_unique_perts, loadings_pca_dim), np.nan)
        
        train_mask_full = (full_adata.obs['context'] == self.target_screen) & (full_adata.obs[self.split_column] == 'train')
        if train_mask_full.sum() >= 2:
            sub_y = full_adata[train_mask_full].X
            if hasattr(sub_y, 'toarray'): sub_y = sub_y.toarray()
            
            pca_load = PCA(n_components=min(loadings_pca_dim, sub_y.shape[0], sub_y.shape[1]), random_state=seed)
            pca_load.fit(sub_y)
            loadings = pca_load.components_.T # (G, PCs)
            
            var_to_idx = {name: i for i, name in enumerate(full_adata.var_names)}
            for pert, idx in self.vocab_pert.items():
                if pert in var_to_idx:
                    v = loadings[var_to_idx[pert]]
                    if v.shape[0] < loadings_pca_dim:
                        v = np.pad(v, (0, loadings_pca_dim - v.shape[0]))
                    X_loadings[idx] = v
                    
        valid_mask = ~np.isnan(X_loadings).all(axis=1)
        if valid_mask.any():
            self.bio_sources.append((X_loadings, "KP_Local_PCA_Loadings"))

        if not self.bio_sources and not self.context_sources:
            raise ValueError("No info sources found to generate kernels.")
            
        # 5. Y: Target full vector
        if hasattr(self.adata.X, 'toarray'):
            self.Y = self.adata.X.toarray()
        else:
            self.Y = self.adata.X
            
        self.output_dim = self.Y.shape[1]
        print(f"Preprocessing complete. Total unique perts: {n_unique_perts}, Output Dim: {self.output_dim}")
        
    def get_split_indices(self) -> Dict[str, np.ndarray]:
        indices = np.arange(len(self.adata))
        splits = {}
        for label in ['train', 'test_seen', 'test_unseen']:
            mask = self.adata.obs[self.split_column] == label
            splits[label] = indices[mask]
        return splits

# --- 2. KRR Implementation ---

def run_kfold_cv(dm: DataModule, 
                 config: Dict, 
                 n_folds: int = 5) -> float:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = dm.get_split_indices()
    train_indices = splits['train']
    
    pert_mapping = dm.adata.obs['target_gene'].map(dm.vocab_pert).values
    fold_scores = []

    for train_idx_local, val_idx_local in kf.split(train_indices):
        ti, vi = train_indices[train_idx_local], train_indices[val_idx_local]
        
        # Sub-indices for unique perturbations
        p_ti = pert_mapping[ti]
        p_vi = pert_mapping[vi]
        
        # Kernel matrices for this fold
        K_t = dm.aggregated_kernel[np.ix_(p_ti, p_ti)]
        K_vt = dm.aggregated_kernel[np.ix_(p_vi, p_ti)]
        
        y_t = dm.Y[ti]
        y_v = dm.Y[vi]
        
        # Solve dual problem: coeffs = (K + alpha*I)^-1 y
        model = KernelRidge(alpha=config['alpha'], kernel='precomputed')
        model.fit(K_t, y_t)
        preds_v = model.predict(K_vt)
        
        err = mean_squared_error(y_v, preds_v)
        fold_scores.append(err)
            
    return np.mean(fold_scores)

def compute_group_kernel(dm: DataModule, sources: List[Tuple[np.ndarray, str]], kernel_type: str, gamma: Optional[float]):
    import gc
    n_perts = len(dm.vocab_pert)
    
    if not sources:
        return np.zeros((n_perts, n_perts))
        
    sum_matrix = np.zeros((n_perts, n_perts), dtype=np.float32)
    count_matrix = np.zeros((n_perts, n_perts), dtype=np.int32)
    
    for X, name in sources:
        valid_mask = ~np.isnan(X).all(axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            X_valid = X[valid_indices]
            # Normalization
            norms = np.linalg.norm(X_valid, axis=1, keepdims=True)
            X_valid = X_valid / (norms + 1e-9)
            
            if kernel_type == 'linear':
                k_vals = linear_kernel(X_valid)
            else:
                from sklearn.metrics.pairwise import rbf_kernel
                k_vals = rbf_kernel(X_valid, gamma=gamma)
                
            idx_grid = np.ix_(valid_indices, valid_indices)
            sum_matrix[idx_grid] += k_vals
            count_matrix[idx_grid] += 1
            
            del k_vals
            gc.collect()
            
    with np.errstate(divide='ignore', invalid='ignore'):
        agg = np.where(count_matrix > 0, sum_matrix / count_matrix, 0.0)
        agg = np.nan_to_num(agg, nan=0.0)
        
    del sum_matrix, count_matrix
    gc.collect()
    
    return agg

def aggregate_kernels_balanced(dm: DataModule, 
                               kernel_type: str = 'linear', 
                               gamma: Optional[float] = None,
                               data_prior_weight: float = 0.5):
    """
    Balanced Aggregation:
    K_final = (1 - lambda) * K_bio + lambda * K_context
    """
    K_bio = compute_group_kernel(dm, dm.bio_sources, kernel_type, gamma)
    K_context = compute_group_kernel(dm, dm.context_sources, kernel_type, gamma)
    
    # If one group is empty, return the other
    if not dm.bio_sources: return K_context
    if not dm.context_sources: return K_bio
    
    return (1.0 - data_prior_weight) * K_bio + data_prior_weight * K_context

def optimize_and_train(h5ad_path: str,
                       target_screen: str = 'k562',
                       split_column: str = 'split_random',
                       param_grid: Optional[List[Dict]] = None,
                       embedding_paths: Optional[List[str]] = None,
                       use_data_prior: bool = True,
                       data_prior_weight: float = 0.5,
                       n_folds: int = 5,
                       seed: int = 42):
    
    dm = DataModule(h5ad_path, target_screen, split_column)
    dm.load_and_preprocess(
        embedding_paths=embedding_paths,
        use_data_prior=use_data_prior,
        seed=seed
    )
    
    if param_grid is None:
        param_grid = list(ParameterGrid([
            {'alpha': [0.1, 1.0, 10.0], 'kernel_type': ['linear']},
            {'alpha': [0.1, 1.0, 10.0], 'kernel_type': ['rbf'], 'gamma': [0.01, 0.1]}
        ]))
        
    best_score = 1e9
    best_config = None
    
    print(f"\n--- Starting HPO (Balanced Mode, lambda={data_prior_weight}) ---")
    last_kernel_params = None
    for i, config in enumerate(param_grid):
        # Re-aggregate if manifold parameters change (weight is constant)
        current_kernel_params = (config.get('kernel_type', 'linear'), config.get('gamma', None))
        if current_kernel_params != last_kernel_params:
            dm.aggregated_kernel = aggregate_kernels_balanced(dm, 
                                                              kernel_type=current_kernel_params[0], 
                                                              gamma=current_kernel_params[1],
                                                              data_prior_weight=data_prior_weight)
            last_kernel_params = current_kernel_params
            
        score = run_kfold_cv(dm, config, n_folds=n_folds)
        print(f"[{i+1}/{len(param_grid)}] Config {config} -> Avg Val Error: {score:.6f}")
        if score < best_score:
            best_score = score
            best_config = config
            
    print(f"\nBest Config: {best_config} with CV Error: {best_score:.6f}")
    
    # --- Final Training with Best ---
    dm.aggregated_kernel = aggregate_kernels_balanced(dm, 
                                                      kernel_type=best_config.get('kernel_type', 'linear'), 
                                                      gamma=best_config.get('gamma', None),
                                                      data_prior_weight=data_prior_weight)
    
    print("\n--- Final Training and Prediction ---")
    splits = dm.get_split_indices()
    train_idx = splits['train']
    test_idx = np.concatenate([splits['test_seen'], splits['test_unseen']])
    
    pert_mapping = dm.adata.obs['target_gene'].map(dm.vocab_pert).values
    p_train, p_test = pert_mapping[train_idx], pert_mapping[test_idx]
    
    K_train = dm.aggregated_kernel[np.ix_(p_train, p_train)]
    K_test_train = dm.aggregated_kernel[np.ix_(p_test, p_train)]
    
    model = KernelRidge(alpha=best_config['alpha'], kernel='precomputed')
    model.fit(K_train, dm.Y[train_idx])
    all_preds_test = model.predict(K_test_train)
        
    # --- Metrics ---
    def calc_metrics(indices_local, tag):
        local_mask = np.isin(test_idx, indices_local)
        p, t = all_preds_test[local_mask], dm.Y[indices_local]
        mse = mean_squared_error(t, p)
        corrs = [pearsonr(p[i], t[i])[0] if np.std(p[i]) > 1e-9 and np.std(t[i]) > 1e-9 else 0 for i in range(p.shape[0])]
        print(f"{tag}: MSE={mse:.4f}, Mean Pearson={np.mean(corrs):.4f}")

    calc_metrics(splits['test_seen'], "Test Seen")
    calc_metrics(splits['test_unseen'], "Test Unseen")

    # --- Save ---
    test_obs = dm.adata.obs.iloc[test_idx][['target_gene', 'context', split_column]].copy()
    test_obs = test_obs.rename(columns={split_column: 'test_split'})
    
    pred_adata = sc.AnnData(X=all_preds_test, obs=test_obs, var=dm.adata.var.copy())
    pred_adata.uns['training_params'] = {
        'target_screen': target_screen,
        'split_column': split_column,
        'model_type': 'Balanced_KRR',
        'lambda': data_prior_weight,
        'best_hyperparams': best_config
    }
    
    tag = 'KDP' if use_data_prior else 'KP'
    output_filename = f"./baseline_results/BalancedKRR_lambda{data_prior_weight}_predictions__{split_column}__MSE__{tag}.h5ad"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    pred_adata.write_h5ad(output_filename)
    print(f"Saved predictions to {output_filename}")

    return output_filename


if __name__ == "__main__":
    param_grid = list(ParameterGrid([
        {'alpha': [0.01, 0.1, 1.0, 10.0], 'kernel_type': ['linear']},
        {'alpha': [0.01, 0.1, 1.0, 10.0], 'kernel_type': ['rbf'], 'gamma': [0.001, 0.01, 0.1]},
    ]))
    embedding_paths = [
        "presage_data/other_embeddings/GenePT_protein_embedding.pkl",
        "presage_data/other_embeddings/esm_emb_gene2esm.pkl",
        "presage_data/pathway_embeddings/c5.go.bp.v2023.2.Hs.symbols.pkl",
        "presage_data/pathway_embeddings/c2.cp.reactome.v2023.2.Hs.symbols.pkl",
        "presage_data/pathway_embeddings/c2.cp.pid.v2023.2.Hs.symbols.pkl",
        "presage_data/pathway_embeddings/c2.cp.wikipathways.v2023.2.Hs.symbols.pkl",
        "presage_data/other_embeddings/HeLa_HPLM.pkl",
        "presage_data/other_embeddings/HeLa_DMEM.pkl",
        "presage_data/other_embeddings/A549_CP186.pkl",
        "presage_data/other_embeddings/CRISPRGeneEffectDepMap.pkl",
        "presage_data/pathway_embeddings/c5.hpo.v2023.2.Hs.symbols.pkl",
        "presage_data/pathway_embeddings/h.all.v2023.2.Hs.symbols.pkl",
        "presage_data/pathway_embeddings/stringdb.human.highest.pkl",
    ]

    res_path = optimize_and_train(
        h5ad_path="./dataset.h5ad",
        target_screen="k562",
        split_column="k562_TF_0.1_UF_0.1_rs_1_random",
        param_grid=param_grid,
        embedding_paths=embedding_paths,
        use_data_prior=True,
        data_prior_weight=0.9,
        n_folds=5,
    )
