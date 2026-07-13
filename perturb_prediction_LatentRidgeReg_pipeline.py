"""
Perturb-Prediction Latent Ridge Regression Pipeline
------------------------------------------
This script implements a latent-embedding regression baseline for predicting transcriptional
responses to genetic perturbations in a target cellular context. Biological priors are aggregated
into a gene-level similarity kernel over both perturbation targets and measured features; latent
coordinates are extracted via truncated SVD and used as inputs to a ridge (or random forest) regressor.

Key Features:
1. Expanded Gene Space: The kernel manifold spans both perturbed genes and expression feature genes.
2. Multiple Kernel Aggregation (MKL): Combines knowledge priors, cross-context mean responses,
   and target-screen PCA loadings into a consensus linear kernel with missing-data averaging.
3. Latent Embedding Regression: Truncated SVD on the aggregated kernel yields low-dimensional
   gene embeddings; Ridge or Random Forest regression maps embeddings to expression profiles.
4. Hyperparameter Search: Cross-validated tuning of embedding dimension and regularization strength.
5. Evaluation: Reports MSE and mean per-sample Pearson correlation on seen and unseen test splits.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import pickle
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import gc

# --- 1. Data Handling & Kernel Generation ---

class DataModule:
    def __init__(self, h5ad_path: str, target_screen: str, split_column: str):
        self.h5ad_path = h5ad_path
        self.target_screen = target_screen
        self.split_column = split_column
        
        self.adata = None
        self.all_genes = []
        self.gene_to_idx = {}
        self.n_total_genes = 0
        
        self.kernel_sources = []
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
        
        # 2. Expand Gene Space: Union of Perturbations and Features
        pert_genes_unique = set(self.adata.obs['target_gene'].unique())
        feature_genes_unique = set(self.adata.var_names)
        self.all_genes = sorted(list(pert_genes_unique | feature_genes_unique))
        self.gene_to_idx = {g: i for i, g in enumerate(self.all_genes)}
        self.n_total_genes = len(self.all_genes)
        
        # 3. Source: Data Priors (Cross-Context Mean Responses)
        if use_data_prior:
            print("Generating Kernels from Cross-Context Data Priors...")
            other_contexts = [c for c in full_adata.obs['context'].unique() if c != self.target_screen]
            
            context_masks = (full_adata.obs['context'].isin(other_contexts)) & (full_adata.obs[self.split_column].isin(valid_splits))
            adata_others = full_adata[context_masks].copy()

            for context in other_contexts:
                adata_c = adata_others[adata_others.obs['context'] == context]
                if adata_c.shape[0] == 0: continue
                
                # Group by perturbation and take mean
                # Use mean-of-perturbation-means to avoid cell-count bias
                pert_means = {}
                for pert in adata_c.obs['target_gene'].unique():
                    if pert in ignored_names: continue
                    m = adata_c[adata_c.obs['target_gene'] == pert].X
                    if hasattr(m, 'toarray'): m = m.toarray()
                    pert_means[pert] = np.mean(m, axis=0)
                
                if not pert_means: continue
                
                # Map to expanded gene space
                X_c = np.full((self.n_total_genes, full_adata.shape[1]), np.nan)
                for pert, idx in self.gene_to_idx.items():
                    if pert in pert_means:
                        X_c[idx] = pert_means[pert]
                
                valid_mask = ~np.isnan(X_c).all(axis=1)
                if valid_mask.any():
                    self.kernel_sources.append((X_c, f"DP_{context}"))

        # 4. Source: Knowledge Priors (Embeddings)
        if embedding_paths:
            print(f"Generating Kernels from {len(embedding_paths)} Knowledge Sources...")
            for path in embedding_paths:
                if not os.path.exists(path):
                    print(f" Warning: Path not found: {path}")
                    continue
                
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                df = data if isinstance(data, pd.DataFrame) else pd.DataFrame.from_dict(data, orient='index')
                
                # Map source embeddings to our expanded gene space
                X_k = np.full((self.n_total_genes, df.shape[1]), np.nan)
                for i, gene in enumerate(self.all_genes):
                    if gene in df.index:
                        X_k[i] = df.loc[gene].values
                
                valid_mask = ~np.isnan(X_k).all(axis=1)
                if valid_mask.any():
                    self.kernel_sources.append((X_k, f"KP_{os.path.basename(path)}"))

        # 5. Source: Target Context PCA Factor Loadings (Local Prior)
        loadings_pca_dim = 128
        print(f"Generating Kernel from Target Context PCA Loadings (dim={loadings_pca_dim})...")
        X_loadings = np.full((self.n_total_genes, loadings_pca_dim), np.nan)
        
        train_mask_full = (full_adata.obs['context'] == self.target_screen) & (full_adata.obs[self.split_column] == 'train')
        if train_mask_full.sum() >= 2:
            sub_y = full_adata[train_mask_full].X
            if hasattr(sub_y, 'toarray'): sub_y = sub_y.toarray()
            
            pca_load = PCA(n_components=min(loadings_pca_dim, sub_y.shape[0], sub_y.shape[1]), random_state=seed)
            pca_load.fit(sub_y)
            loadings = pca_load.components_.T # (G_features, PCs)
            
            var_names = full_adata.var_names.tolist()
            for pert, idx in self.gene_to_idx.items():
                if pert in var_names:
                    v_idx = var_names.index(pert)
                    v = loadings[v_idx]
                    if v.shape[0] < loadings_pca_dim:
                        v = np.pad(v, (0, loadings_pca_dim - v.shape[0]))
                    X_loadings[idx] = v
                    
        valid_mask = ~np.isnan(X_loadings).all(axis=1)
        if valid_mask.any():
            self.kernel_sources.append((X_loadings, "KP_Local_PCA_Loadings"))

        if not self.kernel_sources:
            raise ValueError("No info sources found to generate kernels.")
        
        # Aggregate Kernels (Linear)
        print("Aggregating kernels...")
        
        # Initialize running sum and count matrices as float32/int32 to save memory
        sum_matrix = np.zeros((self.n_total_genes, self.n_total_genes), dtype=np.float32)
        count_matrix = np.zeros((self.n_total_genes, self.n_total_genes), dtype=np.int32)

        for X, name in tqdm(self.kernel_sources):
            valid_mask = ~np.isnan(X).all(axis=1)
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                X_valid = X[valid_indices]
                # Normalize all kernel sources to unit norm
                norms = np.linalg.norm(X_valid, axis=1, keepdims=True)
                X_valid = X_valid / (norms + 1e-9)
                
                # Compute kernel in float32
                k_vals = linear_kernel(X_valid)#.astype(np.float32)
                
                # Fast matrix imputation using advanced indexing
                idx_grid = np.ix_(valid_indices, valid_indices)
                sum_matrix[idx_grid] += k_vals
                count_matrix[idx_grid] += 1
                
                del k_vals
                gc.collect()
                
        # Compute mean safely
        with np.errstate(divide='ignore', invalid='ignore'):
            self.aggregated_kernel = np.where(count_matrix > 0, sum_matrix / count_matrix, 0.0)
            self.aggregated_kernel = np.nan_to_num(self.aggregated_kernel, nan=0.0)
            
        del sum_matrix, count_matrix
        gc.collect()
            
        # 6. Y: Target full vector
        if hasattr(self.adata.X, 'toarray'):
            self.Y = self.adata.X.toarray()
        else:
            self.Y = self.adata.X
            
        self.output_dim = self.Y.shape[1]
        print(f"Preprocessing complete. Total genes: {self.n_total_genes}, Output Dim: {self.output_dim}")
        
    def get_split_indices(self) -> Dict[str, np.ndarray]:
        indices = np.arange(len(self.adata))
        splits = {}
        for label in ['train', 'test_seen', 'test_unseen']:
            mask = self.adata.obs[self.split_column] == label
            splits[label] = indices[mask]
        return splits

# --- 2. Model & CV ---

class LatentRegModel:
    def __init__(self, emb_dim: int, model_type: str = 'ridge', model_params: Dict = None):
        self.emb_dim = emb_dim
        self.model_type = model_type
        self.model_params = model_params or {}
        self.svd = TruncatedSVD(n_components=emb_dim, random_state=42)
        
        if model_type == 'ridge':
            self.model = Ridge(**self.model_params)
        elif model_type == 'rf':
            self.model = RandomForestRegressor(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def fit(self, K_agg: np.ndarray, train_gene_indices: np.ndarray, y_train: np.ndarray):
        # Extract Latent Features from K
        self.gene_embeddings = self.svd.fit_transform(K_agg)
        
        # Prepare training data
        x_train = self.gene_embeddings[train_gene_indices]
        self.model.fit(x_train, y_train)
        
    def predict(self, test_gene_indices: np.ndarray):
        x_test = self.gene_embeddings[test_gene_indices]
        return self.model.predict(x_test)

def run_kfold_cv(dm: DataModule, config: Dict, n_folds: int = 5) -> float:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = dm.get_split_indices()
    train_indices = splits['train']
    
    pert_mapping = np.array([dm.gene_to_idx[p] for p in dm.adata.obs['target_gene']])
    
    fold_scores = []
    for train_idx_local, val_idx_local in kf.split(train_indices):
        ti, vi = train_indices[train_idx_local], train_indices[val_idx_local]
        
        model = LatentRegModel(
            emb_dim=config['emb_dim'], 
            model_type=config['model_type'], 
            model_params=config.get('model_params', {})
        )
        
        model.fit(dm.aggregated_kernel, pert_mapping[ti], dm.Y[ti])
        preds_v = model.predict(pert_mapping[vi])
        
        err = mean_squared_error(dm.Y[vi], preds_v)
        fold_scores.append(err)
            
    return np.mean(fold_scores)

# --- 3. Pipeline Entry point ---

def optimize_and_train(h5ad_path: str,
                       target_screen: str = 'k562',
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
            'emb_dim': [64, 128, 256],
            'model_type': ['ridge'],
            'model_params': [{'alpha': 1.0}, {'alpha': 10.0}]
        }))
        
    best_score = 1e9
    best_config = None
    
    print("\n--- Starting LatentReg HPO ---")
    for i, config in enumerate(param_grid):
        score = run_kfold_cv(dm, config, n_folds=n_folds)
        print(f"[{i+1}/{len(param_grid)}] Config {config} -> Avg Val Error: {score:.6f}")
        if score < best_score:
            best_score = score
            best_config = config
            
    print(f"\nBest Config: {best_config} with CV Error: {best_score:.6f}")
    
    # --- Final Training ---
    print("\n--- Final Training and Prediction ---")
    splits = dm.get_split_indices()
    train_idx = splits['train']
    test_seen_idx = splits['test_seen']
    test_unseen_idx = splits['test_unseen']
    test_idx = np.concatenate([test_seen_idx, test_unseen_idx])
    
    pert_mapping = np.array([dm.gene_to_idx[p] for p in dm.adata.obs['target_gene']])
    
    final_model = LatentRegModel(
        emb_dim=best_config['emb_dim'], 
        model_type=best_config['model_type'], 
        model_params=best_config.get('model_params', {})
    )
    final_model.fit(dm.aggregated_kernel, pert_mapping[train_idx], dm.Y[train_idx])
    all_preds_test = final_model.predict(pert_mapping[test_idx])
        
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
    
    has_kp = (embedding_paths is not None and len(embedding_paths) > 0)
    has_dp = use_data_prior
    tag_p = "KDP" if (has_kp and has_dp) else ("KP" if has_kp else "DP")
    
    pred_adata.uns['training_params'] = {
        'target_screen': target_screen,
        'split_column': split_column,
        'model_type': f'LatentReg_{best_config["model_type"]}',
        'best_config': best_config,
        'loss_type': 'MSE',
        'info_tag': tag_p
    }
    
    model_name = best_config["model_type"].upper()
    output_filename = f"./baseline_results/{model_name}_LatentReg_predictions__{split_column}__MSE__{tag_p}.h5ad"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    pred_adata.write_h5ad(output_filename)
    print(f"Saved predictions to {output_filename}") 
    return output_filename

if __name__ == "__main__":
    param_grid = list(ParameterGrid({
        'emb_dim': [128],
        'model_type': ['ridge'],
        'model_params': [{'alpha': 0.1}, {'alpha': 1.0}, {'alpha': 10.0}],
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
