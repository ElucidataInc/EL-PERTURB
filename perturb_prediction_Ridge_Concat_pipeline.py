"""
Perturb-Prediction Ridge Concat Pipeline
--------------------------------------------------
This script implements a Ridge regression pipeline for predicting 
transcriptional responses in a specific target screen.

Key Features:
1. Multi-Source Input Fusion:
   - KP: Concatenated Knowledge Priors including biological embeddings (e.g., ESM, GenePT) and context feature loadings.
   - DP: Mean response vector across all other screening contexts projected to 'emb_dim' via PCA.
   - All sources are concatenated into a single wide feature vector instead of being summed.
2. Ridge Regression Model: Uses L2 regularization for the linear model.
3. Hyperparameter Optimization (HPO): Systematic search over 'alpha' (regularization strength).
4. Full Transcriptome Output: Predicts the complete response vector in the target context using standard MSE.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import pickle
import json

# --- 1. Data Handling ---

class DataModule:
    def __init__(self, h5ad_path: str, target_screen: str, split_column: str = 'split_random'):
        self.h5ad_path = h5ad_path
        self.target_screen = target_screen
        self.split_column = split_column
        
        self.adata = None
        self.vocab_pert = {}
        
        self.X_kp = None # Knowledge Prior (Sum of projected embeddings)
        self.X_dp = None # Data Prior (Mean across other contexts)
        self.X_combined = None
        self.Y = None
        self.output_dim = 0
        
    def load_and_preprocess(self, 
                            emb_dim: int = 128, 
                            embedding_paths: Optional[List[str]] = None,
                            use_data_prior: bool = True,
                            seed: int = 42):
        print(f"Loading data from {self.h5ad_path}...")
        full_adata = sc.read_h5ad(self.h5ad_path)
        
        # 1. Initialize Vocabulary based on TARGET genes in the dataset
        valid_splits = ['train', 'test_seen', 'test_unseen']
        ignored_names = {'control', 'non-targeting', 'Control', 'Non-targeting'}
        mask_target = (full_adata.obs['context'] == self.target_screen) & (full_adata.obs[self.split_column].isin(valid_splits))
        mask_target &= ~full_adata.obs['target_gene'].isin(ignored_names)
        
        self.adata = full_adata[mask_target].copy()
        print(f"Kept {len(self.adata)} samples from {self.target_screen} screen.")
        
        all_perts = sorted(self.adata.obs['target_gene'].unique())
        self.vocab_pert = {p: i for i, p in enumerate(all_perts)}
        n_perts = len(all_perts)
        target_genes = all_perts
        
        # 2. X_dp: Individual context response vectors (PCA-reduced per context)
        if use_data_prior:
            print(f"Computing DP: Per-context response vectors (PCA to {emb_dim})...")
            other_contexts = [c for c in full_adata.obs['context'].unique() if c != self.target_screen]
            
            dp_matrices = []
            for ctx in other_contexts:
                ctx_mask = (full_adata.obs['context'] == ctx) & (full_adata.obs[self.split_column].isin(valid_splits))
                adata_ctx = full_adata[ctx_mask].copy()
                
                if len(adata_ctx) == 0:
                    continue
                    
                context_gene_matrix = np.zeros((n_perts, full_adata.shape[1]))
                pert_means = {}
                for pert in target_genes:
                    pert_mask = adata_ctx.obs['target_gene'] == pert
                    if pert_mask.sum() > 0:
                        X_sub = adata_ctx[pert_mask].X
                        if hasattr(X_sub, 'toarray'): X_sub = X_sub.toarray()
                        pert_means[pert] = np.mean(X_sub, axis=0)
                
                global_avg_ctx = np.mean([v for v in pert_means.values()], axis=0) if pert_means else np.zeros(full_adata.shape[1])
                
                for i, pert in enumerate(target_genes):
                    context_gene_matrix[i] = pert_means.get(pert, global_avg_ctx)
                    
                # Standardize columns and do PCA for this specific context
                context_gene_matrix_scaled = StandardScaler().fit_transform(context_gene_matrix)
                pca_dp = PCA(n_components=min(emb_dim, context_gene_matrix.shape[0], context_gene_matrix.shape[1]), random_state=seed)
                ctx_dp = pca_dp.fit_transform(context_gene_matrix_scaled)
                if ctx_dp.shape[1] < emb_dim:
                    ctx_dp = np.pad(ctx_dp, ((0,0), (0, emb_dim - ctx_dp.shape[1])))
                dp_matrices.append(ctx_dp)
                
            if dp_matrices:
                self.X_dp = np.concatenate(dp_matrices, axis=1)
            else:
                self.X_dp = np.zeros((n_perts, 0))
        else:
            self.X_dp = np.zeros((n_perts, 0))

        # 3. Knowledge Prior (KP) Aggregation
        print(f"Aggregating Knowledge Priors (dim={emb_dim})...")
        projected_embeddings = []

        # 3a. External Biological Embeddings
        if embedding_paths:
            print(f"  Incorporating {len(embedding_paths)} external knowledge sources...")
            for path in embedding_paths:
                if not os.path.exists(path):
                    continue
                if path.endswith('.json'):
                    with open(path, 'r') as f:
                        data = json.load(f)
                else:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                df = data if isinstance(data, pd.DataFrame) else pd.DataFrame.from_dict(data, orient='index')
                df_aligned = df.reindex(target_genes)
                df_aligned = df_aligned.fillna(df.mean(axis=0)) if not df.empty else df_aligned.fillna(0)
                vals = df_aligned.values
                # Standardize columns before PCA
                vals_scaled = StandardScaler().fit_transform(vals)
                
                pca = PCA(n_components=min(emb_dim, vals_scaled.shape[0], vals_scaled.shape[1]), random_state=seed)
                red = pca.fit_transform(vals_scaled)
                #print(f"Reduced matrix has shape {red.shape}")
                if red.shape[1] < emb_dim:
                    red = np.pad(red, ((0,0), (0, emb_dim - red.shape[1])))
                projected_embeddings.append(red)
        
        # 3b. Target Context PCA Factor Loadings (Context Prior)
        loadings_pca_dim = emb_dim
        print(f"Generating Target Context PCA Loadings (dim={loadings_pca_dim})...")
        X_loadings = np.full((n_perts, loadings_pca_dim), np.nan)
        
        train_mask_full = (full_adata.obs['context'] == self.target_screen) & (full_adata.obs[self.split_column] == 'train')
        if train_mask_full.sum() >= 2:
            sub_y = full_adata[train_mask_full].X
            if hasattr(sub_y, 'toarray'): sub_y = sub_y.toarray()
            
            # Standardize before PCA
            sub_y_scaled = StandardScaler().fit_transform(sub_y)
            pca_load = PCA(n_components=min(loadings_pca_dim, sub_y.shape[0], sub_y.shape[1]), random_state=seed)
            pca_load.fit(sub_y_scaled)
            loadings = pca_load.components_.T # (G, PCs)
            
            var_to_idx = {name: i for i, name in enumerate(full_adata.var_names)}
            for pert, idx in self.vocab_pert.items():
                if pert in var_to_idx:
                    v = loadings[var_to_idx[pert]]
                    if v.shape[0] < loadings_pca_dim:
                        v = np.pad(v, (0, loadings_pca_dim - v.shape[0]))
                    X_loadings[idx] = v
        projected_embeddings.append(X_loadings)
        print("# of embeddings loaded: {}".format(len(projected_embeddings)))

        if not projected_embeddings:
            self.X_kp = np.zeros((n_perts, min(emb_dim, n_perts)))
        else:
            self.X_kp = np.concatenate(projected_embeddings, axis=1)

        # 5. Final X per sample
        print(f"Constructing combined input feature matrix...")
        pert_to_idx = {p: i for i, p in enumerate(target_genes)}
        sample_pert_idxs = [pert_to_idx[p] for p in self.adata.obs['target_gene']]

        X_per_pert = np.concatenate([self.X_kp, self.X_dp], axis=1) 
        self.X_combined = X_per_pert[sample_pert_idxs] 
        
        if hasattr(self.adata.X, 'toarray'):
            self.Y = self.adata.X.toarray()
        else:
            self.Y = self.adata.X
            
        self.output_dim = self.Y.shape[1]
        print(f"Preprocessing complete. X shape: {self.X_combined.shape}, Y shape: {self.Y.shape}")
        
    def get_split_indices(self) -> Dict[str, np.ndarray]:
        indices = np.arange(len(self.adata))
        splits = {}
        for label in ['train', 'test_seen', 'test_unseen']:
            mask = self.adata.obs[self.split_column] == label
            splits[label] = indices[mask]
        return splits

# --- 3. Ridge Trainer ---

def run_kfold_cv(X: np.ndarray, 
                 Y: np.ndarray, 
                 train_indices: np.ndarray,
                 config: Dict, 
                 n_folds: int = 5) -> float:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []
    
    X_train_full = X[train_indices]
    Y_train_full = Y[train_indices]
    
    for fold_i, (t_idx, v_idx) in enumerate(kf.split(X_train_full)):
        X_t, X_v = X_train_full[t_idx], X_train_full[v_idx]
        Y_t, Y_v = Y_train_full[t_idx], Y_train_full[v_idx]
        
        # Ridge is fast, so we can train on all genes simultaneously
        model = Ridge(alpha=config['alpha'], random_state=42)
        model.fit(X_t, Y_t)
        preds_v = model.predict(X_v)
        
        err = mean_squared_error(Y_v, preds_v)
        fold_scores.append(err)
            
    return np.mean(fold_scores)

def optimize_and_train(h5ad_path: str,
                       target_screen: str = 'jurkat',
                       split_column: str = 'split_random',
                       param_grid: Optional[List[Dict]] = None,
                       emb_dim: int = 128,
                       embedding_paths: Optional[List[str]] = None,
                       use_data_prior: bool = True,
                       n_folds: int = 5,
                       seed: int = 42):
    
    dm = DataModule(h5ad_path, target_screen, split_column)
    dm.load_and_preprocess(
        emb_dim=emb_dim,
        embedding_paths=embedding_paths,
        use_data_prior=use_data_prior,
        seed=seed
    )
    
    splits = dm.get_split_indices()
    train_idx = splits['train']
    
    if param_grid is None:
        param_grid = list(ParameterGrid({'alpha': [0.1, 1.0, 10.0]}))
        
    best_score = 1e9
    best_config = None
    
    print("\n--- Starting HPO (Ridge) ---")
    for i, config in enumerate(param_grid):
        score = run_kfold_cv(dm.X_combined, dm.Y, train_idx, config, n_folds=n_folds)
        print(f"[{i+1}/{len(param_grid)}] Config {config} -> Avg Val Error: {score:.6f}")
        if score < best_score:
            best_score = score
            best_config = config
            
    print(f"\nBest Config: {best_config} with CV Error: {best_score:.6f}")
    
    # --- Final Training ---
    print("\n--- Final Training ---")
    test_indices = np.concatenate([splits['test_seen'], splits['test_unseen']])
    X_train = dm.X_combined[train_idx]
    X_test = dm.X_combined[test_indices]
    
    model = Ridge(alpha=best_config['alpha'], random_state=seed)
    model.fit(X_train, dm.Y[train_idx])
    final_preds_test = model.predict(X_test)
    
    # Reconstruct full preds matrix for indexing consistency
    final_preds = np.zeros_like(dm.Y)
    final_preds[test_indices] = final_preds_test
        
    # --- Evaluation ---
    def calc_metrics(indices, tag):
        p = final_preds[indices]
        t = dm.Y[indices]
        mse = mean_squared_error(t, p)
        
        corrs = []
        for i in range(len(indices)):
            row_p = p[i]
            row_t = t[i]
            if np.std(row_p) > 1e-9 and np.std(row_t) > 1e-9:
                corrs.append(pearsonr(row_p, row_t)[0])
            else:
                corrs.append(0)
        print(f"{tag}: MSE={mse:.4f}, Mean Pearson={np.mean(corrs):.4f}")

    calc_metrics(splits['test_seen'], "Test Seen")
    calc_metrics(splits['test_unseen'], "Test Unseen")

    # --- Save ---
    test_obs = dm.adata.obs.iloc[test_indices][['target_gene', 'context', split_column]].copy()
    test_obs = test_obs.rename(columns={split_column: 'test_split'})
    
    pred_adata = sc.AnnData(X=final_preds[test_indices], obs=test_obs, var=dm.adata.var.copy())
    pred_adata.uns['best_hyperparams'] = best_config
    pred_adata.uns['training_params'] = {
        'target_screen': target_screen,
        'split_column': split_column,
        'emb_dim': emb_dim,
        'loss_type': 'MSE',
        'use_knowledge_prior': (embedding_paths is not None and len(embedding_paths) > 0),
        'use_data_prior': use_data_prior,
        'n_folds': n_folds,
        'seed': seed,
        'model_type': 'Ridge'
    }
    
    has_kp = (embedding_paths is not None and len(embedding_paths) > 0)
    has_dp = use_data_prior
    if has_kp and has_dp: tag_prior = "KDP"
    elif has_kp: tag_prior = "KP"
    elif has_dp: tag_prior = "DP"
    else: tag_prior = "NoPrior"

    output_filename = f"./baseline_results/Ridge_Concat_predictions__{split_column}__MSE__{tag_prior}.h5ad"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    pred_adata.write_h5ad(output_filename)
    print(f"Saved predictions to {output_filename}")
    return output_filename


# ----- Main loop -----
if __name__ == "__main__":
    
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
    
    param_grid = list(ParameterGrid({'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}))

    res_path = optimize_and_train(
        h5ad_path="./dataset.h5ad",
        target_screen="k562",
        split_column="k562_TF_0.1_UF_0.1_rs_1_random",
        emb_dim=128,
        embedding_paths=embedding_paths,
        param_grid=param_grid,
        use_data_prior=True,
        seed=42
    )
