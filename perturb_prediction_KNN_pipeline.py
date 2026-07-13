"""
Perturb-Prediction KNN Pipeline
------------------------------------------
This script implements a K-Nearest Neighbors (KNN) baseline for predicting transcriptional
responses to genetic perturbations in a target cellular context. Perturbation similarity is
defined on an aggregated kernel manifold built from biological embeddings, cross-context data
priors, and target-screen PCA loadings.

Key Features:
1. Multiple Kernel Aggregation (MKL): Combines knowledge priors (ESM, GenePT, pathways),
   cross-context mean responses, and local PCA loadings into a consensus linear kernel.
   Missing entries are handled by averaging only over available sources per perturbation pair.
2. Kernel K-NN Prediction: Each test perturbation is assigned the mean expression profile
   of its k most similar training perturbations on the aggregated manifold.
3. Hyperparameter Search: Cross-validated tuning of n_neighbors over a user-defined grid.
4. Evaluation: Reports MSE and mean per-sample Pearson correlation on seen and unseen test splits.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import linear_kernel
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
        
        self.kernel_sources = [] # List of (feature_matrix, source_name)
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
        
        # 2. Source: Data Priors (Cross-Context Mean Responses)
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
                    self.kernel_sources.append((X_c, f"DP_{context}"))

        # 3. Source: Knowledge Priors (Embeddings)
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
                    self.kernel_sources.append((X_k, f"KP_{os.path.basename(path)}"))

        # 4. Source: Target Context PCA Factor Loadings (Local Prior)
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
            self.kernel_sources.append((X_loadings, "KP_Local_PCA_Loadings"))

        if not self.kernel_sources:
            raise ValueError("No info sources found to generate kernels.")
            
        # 5. Y: Target full vector
        if hasattr(self.adata.X, 'toarray'):
            self.Y = self.adata.X.toarray()
        else:
            self.Y = self.adata.X
            
        self.output_dim = self.Y.shape[1]
        print(f"Preprocessing complete. Total perturbations: {n_unique_perts}, Output Dim: {self.output_dim}")
        
    def get_split_indices(self) -> Dict[str, np.ndarray]:
        indices = np.arange(len(self.adata))
        splits = {}
        for label in ['train', 'test_seen', 'test_unseen']:
            mask = self.adata.obs[self.split_column] == label
            splits[label] = indices[mask]
        return splits

# --- 2. K-NN Implementation ---

class KernelKNN:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.y_train = None
        
    def fit(self, K_train: np.ndarray, y_train: np.ndarray):
        # In this context, K_train is the similarity matrix among training perturbations
        # But for K-NN, we actually only need the labels (y_train) mapped to the perturbations
        self.y_train = y_train
        
    def predict(self, K_test_train: np.ndarray) -> np.ndarray:
        # K_test_train shape: (n_test, n_train) where each row i gives 
        # similarities of test perturbation i to all training perturbations
        n_test = K_test_train.shape[0]
        n_out = self.y_train.shape[1]
        preds = np.zeros((n_test, n_out))
        
        for i in range(n_test):
            # Find indices of top n_neighbors similarities
            sims = K_test_train[i]
            # Use argpartition for efficiency
            top_k_indices = np.argpartition(sims, -self.n_neighbors)[-self.n_neighbors:]
            # Simple average of the neighbors
            preds[i] = np.mean(self.y_train[top_k_indices], axis=0)
            
        return preds

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
        
        p_ti = pert_mapping[ti]
        p_vi = pert_mapping[vi]
        
        K_vt = dm.aggregated_kernel[np.ix_(p_vi, p_ti)]
        
        y_t = dm.Y[ti]
        y_v = dm.Y[vi]
        
        model = KernelKNN(n_neighbors=config['n_neighbors'])
        model.fit(None, y_t) # K_train not used in this implementation
        preds_v = model.predict(K_vt)
        
        err = mean_squared_error(y_v, preds_v)
        fold_scores.append(err)
            
    return np.mean(fold_scores)

def aggregate_kernels(dm: DataModule):
    """Generates the aggregated linear kernel from multiple sources using memory-efficient online aggregation."""
    import gc
    n_perts = len(dm.vocab_pert)
    
    if not dm.kernel_sources:
        return np.zeros((n_perts, n_perts))
        
    sum_matrix = np.zeros((n_perts, n_perts), dtype=np.float32)
    count_matrix = np.zeros((n_perts, n_perts), dtype=np.int32)
    
    for X, name in dm.kernel_sources:
        valid_mask = ~np.isnan(X).all(axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            X_valid = X[valid_indices]
            # Normalize all kernel sources to unit norm
            norms = np.linalg.norm(X_valid, axis=1, keepdims=True)
            X_valid = X_valid / (norms + 1e-9)
            
            k_vals = linear_kernel(X_valid) #.astype(np.float32)
                
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

def optimize_and_train(h5ad_path: str,
                       target_screen: str = 'jurkat',
                       split_column: str = 'split_random',
                       param_grid: Optional[List[Dict]] = None,
                       embedding_paths: Optional[List[str]] = None,
                       use_data_prior: bool = True,
                       n_folds: int = 5,
                       seed: int = 42):
    
    dm = DataModule(h5ad_path, target_screen, split_column)
    dm.load_and_preprocess(
        embedding_paths=embedding_paths,
        use_data_prior=use_data_prior,
        seed=seed
    )
    
    if param_grid is None:
        param_grid = list(ParameterGrid({
            'n_neighbors': [1, 3, 5, 10, 20, 50]
        }))
        
    best_score = 1e9
    best_config = None
    
    print("\n--- Starting HPO ---")
    # Linear kernel is static across HPO
    dm.aggregated_kernel = aggregate_kernels(dm)
    
    for i, config in enumerate(param_grid):
        score = run_kfold_cv(dm, config, n_folds=n_folds)
        print(f"[{i+1}/{len(param_grid)}] Config {config} -> Avg Val Error: {score:.6f}")
        if score < best_score:
            best_score = score
            best_config = config
            
    print(f"\nBest Config: {best_config} with CV Error: {best_score:.6f}")
    
    # --- Final Training with Best ---
    print("\n--- Final Training and Prediction ---")
    splits = dm.get_split_indices()
    train_idx = splits['train']
    test_seen_idx = splits['test_seen']
    test_unseen_idx = splits['test_unseen']
    test_idx = np.concatenate([test_seen_idx, test_unseen_idx])
    
    pert_mapping = dm.adata.obs['target_gene'].map(dm.vocab_pert).values
    p_train = pert_mapping[train_idx]
    p_test = pert_mapping[test_idx]
    
    K_test_train = dm.aggregated_kernel[np.ix_(p_test, p_train)]
    
    model = KernelKNN(n_neighbors=best_config['n_neighbors'])
    model.fit(None, dm.Y[train_idx])
    all_preds_test = model.predict(K_test_train)
        
    # --- Metrics ---
    def calc_metrics(indices_local, tag):
        local_mask = np.isin(test_idx, indices_local)
        p = all_preds_test[local_mask]
        t = dm.Y[indices_local]
        
        mse = mean_squared_error(t, p)
        
        corrs = []
        for i in range(p.shape[0]):
            if np.std(p[i]) > 1e-9 and np.std(t[i]) > 1e-9:
                corrs.append(pearsonr(p[i], t[i])[0])
            else:
                corrs.append(0)
        print(f"{tag}: MSE={mse:.4f}, Mean Pearson={np.mean(corrs):.4f}")

    calc_metrics(test_seen_idx, "Test Seen")
    calc_metrics(test_unseen_idx, "Test Unseen")

    # --- Save ---
    test_obs = dm.adata.obs.iloc[test_idx][['target_gene', 'context', split_column]].copy()
    test_obs = test_obs.rename(columns={split_column: 'test_split'})
    
    pred_adata = sc.AnnData(X=all_preds_test, obs=test_obs, var=dm.adata.var.copy())
    pred_adata.uns['training_params'] = {
        'target_screen': target_screen,
        'split_column': split_column,
        'model_type': 'KNN',
        'kernel_type': 'linear',
        'n_neighbors': best_config['n_neighbors']
    }
    
    has_kp = (embedding_paths is not None and len(embedding_paths) > 0)
    has_dp = use_data_prior
    tag_p = "KDP" if (has_kp and has_dp) else ("KP" if has_kp else "DP")
    
    output_filename = f"./baseline_results/KNN_predictions__{split_column}__MSE__{tag_p}.h5ad"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    pred_adata.write_h5ad(output_filename)
    print(f"Saved predictions to {output_filename}")
    return output_filename


if __name__ == "__main__":
    param_grid = list(ParameterGrid({
        'n_neighbors': [1, 3, 5, 10, 20, 50],
    }))
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
        n_folds=5,
    )
