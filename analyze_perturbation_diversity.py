"""
Perturbation Response Diversity Analysis
-----------------------------------------------------

Purpose
- Quantify how diverse perturbation-response vectors are within and across screens/datasets.
- Provide apples-to-apples comparison by subsampling each screen to the same number of perturbations.

Main Workflows
1) `analyze_diversity(h5ad_path, group_col)`
   - Computes diversity metrics separately for each group in `adata.obs[group_col]`.
2) `compare_two_h5ads(path_1, path_2, name_1, name_2)`
   - Intersects common features, filters control/non-targeting perturbations, and compares per-screen
     diversity after matching perturbation count via repeated random subsampling.

Core Metrics (in `calc_diversity_metrics`)
- Effective dimensionality (Participation Ratio)
- Spectral entropy and normalized spectral entropy
- PCs needed for 90% variance
- Mean pairwise cosine distance
- Mean pairwise Pearson correlation

Inputs / assumptions
- `.h5ad` files readable by Scanpy
- Response matrix in `adata.X` (rows are perturbations/samples, columns are features/genes)
- `target_gene` in `adata.obs` for control filtering
- `context` in `adata.obs` when doing per-screen comparison (optional but recommended)

Outputs
- CSV files in `results/` with computed diversity metrics.

Run
- Configure paths in the `__main__` block, then:
  `python analyze_perturbation_diversity.py`
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances
import scipy

def calc_diversity_metrics(X, group_name="Group"):
    """
    Calculate matrix-based diversity metrics for a response matrix X.
    X shape: (N_perturbations, P_features)
    """
    if hasattr(X, "toarray"):
        X = X.toarray()
        
    N, P = X.shape
    if N == 0:
        return None
        
    # 1. Singular Value Decomposition
    # Remove mean for SVD/PCA context
    X_centered = X - np.mean(X, axis=0)
    
    # SVD
    # full_matrices=False ensures efficiency
    U, s, Vt = scipy.linalg.svd(X_centered, full_matrices=False)
    
    # Eigenvalues / variances
    eigvals = s**2
    eig_sum = np.sum(eigvals)
    
    # 2. Spectral Entropy
    p_i = eigvals / (eig_sum + 1e-12)
    # Filter out zeros for log
    p_i_valid = p_i[p_i > 0]
    spectral_entropy = -np.sum(p_i_valid * np.log(p_i_valid))
    
    # Normalized Spectral Entropy (0 to 1)
    max_entropy = np.log(min(N, P))
    norm_spectral_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0
    
    # 3. Participation Ratio (Effective Dimensionality)
    pr = (np.sum(eigvals)**2) / (np.sum(eigvals**2) + 1e-12)
    
    # 4. PCs to explain 90% variance
    cum_var = np.cumsum(p_i)
    pcs_90 = np.argmax(cum_var >= 0.90) + 1
    
    # 5. Average Pairwise Cosine Distance
    # Distances between perturbations
    if N > 1:
        dists = cosine_distances(X)
        # Get upper triangle excluding diagonal
        upper_tri = dists[np.triu_indices(N, k=1)]
        mean_cos_dist = np.mean(upper_tri)
        
        # 6. Mean Pairwise Pearson Correlation
        corrs = np.corrcoef(X)
        upper_tri_corr = corrs[np.triu_indices(N, k=1)]
        mean_pcc = np.nanmean(upper_tri_corr)
    else:
        mean_cos_dist = 0.0
        mean_pcc = 0.0
        
    return {
        "Perturbation_Set": group_name,
        "N_Perts": N,
        "N_Features": P,
        "Effective_Dim_(PR)": pr,
        "Spectral_Entropy": spectral_entropy,
        "Norm_Spectral_Entropy": norm_spectral_entropy,
        "PCs_for_90pct_Var": pcs_90,
        "Mean_Cosine_Dist": mean_cos_dist,
        "Mean_PCC": mean_pcc
    }

def analyze_diversity(h5ad_path: str, group_col: str):
    print(f"Loading data from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    
    if group_col not in adata.obs.columns:
        raise ValueError(f"Column '{group_col}' not found in adata.obs.")
        
    ignored_names = {'control', 'non-targeting', 'Control', 'Non-targeting', 'excluded'}
    mask_valid = ~adata.obs['target_gene'].isin(ignored_names)
    adata = adata[mask_valid].copy()
        
    mat = adata.X
        
    groups = adata.obs[group_col].unique()
    groups = [g for g in groups if pd.notna(g) and str(g).strip() != '']
    
    results = []
    
    print(f"Evaluating matrix diversity dynamically across {len(groups)} splits found in '{group_col}'...")
    for g in groups:
        mask = adata.obs[group_col] == g
        subset_X = mat[mask, :]
        
        # Calculate
        metrics = calc_diversity_metrics(subset_X, group_name=str(g))
        if metrics:
            results.append(metrics)
        
    if not results:
        print("No valid groups calculated.")
        return
        
    df_results = pd.DataFrame(results)
    print("\n========= Transcriptional Response Diversity =========")
    print(df_results.to_string(index=False))
    
    # Save to CSV
    os.makedirs("results", exist_ok=True)
    out_csv = f"results/perturbation_diversity_{group_col}.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved numerical diversity metrics to {out_csv}")
    
def process_screen_iterations(X, name, min_perts, n_iters=5):
    if X.shape[0] == min_perts:
        # No need to iterate, it's the exact same matrix every time naturally
        metrics = calc_diversity_metrics(X, group_name=name)
        return metrics if metrics else None
        
    all_metrics = []
    for i in range(n_iters):
        np.random.seed(42 + i) # Ensures distinct random seed for multiple iter draws
        idx = np.random.choice(X.shape[0], min_perts, replace=False)
        subset_X = X[idx, :]
        res = calc_diversity_metrics(subset_X, group_name=name)
        if res:
            all_metrics.append(res)
            
    if not all_metrics:
        return None
        
    # Average the distinct evaluations
    avg_res = {"Perturbation_Set": name}
    for k in all_metrics[0].keys():
        if k == "Perturbation_Set": continue
        avg_res[k] = np.mean([m[k] for m in all_metrics])
    
    return avg_res
    
def compare_two_h5ads(path_1: str, path_2: str, name_1: str = "Dataset_1", name_2: str = "Dataset_2"):
    print(f"\nLoading {name_1} from {path_1}...")
    ad1 = sc.read_h5ad(path_1)
    print(f"Loading {name_2} from {path_2}...")
    ad2 = sc.read_h5ad(path_2)
    
    # Filter controls
    ignored_names = {'control', 'non-targeting', 'Control', 'Non-targeting', 'excluded'}
    mask_1 = ~ad1.obs['target_gene'].isin(ignored_names) if 'target_gene' in ad1.obs else np.ones(len(ad1), dtype=bool)
    mask_2 = ~ad2.obs['target_gene'].isin(ignored_names) if 'target_gene' in ad2.obs else np.ones(len(ad2), dtype=bool)
    
    ad1 = ad1[mask_1].copy()
    ad2 = ad2[mask_2].copy()
    
    # Intersect Features
    common_features = list(set(ad1.var_names) & set(ad2.var_names))
    print(f"\nFound {len(common_features)} common feature genes between the two datasets.")
    
    if len(common_features) == 0:
        print("Error: No common features to compare.")
        return
        
    ad1 = ad1[:, common_features].copy()
    ad2 = ad2[:, common_features].copy()
    
    # Determine min perturbations across any single screen in either dataset
    sizes = []
    if 'context' in ad1.obs.columns:
        sizes.extend([np.sum(ad1.obs['context'] == c) for c in ad1.obs['context'].unique() if pd.notna(c)])
    else:
        sizes.append(ad1.shape[0])
        
    if 'context' in ad2.obs.columns:
        sizes.extend([np.sum(ad2.obs['context'] == c) for c in ad2.obs['context'].unique() if pd.notna(c)])
    else:
        sizes.append(ad2.shape[0])
        
    if not sizes:
        print("Error: No data available after filtering.")
        return
        
    min_perts = min(sizes)
    print(f"Minimum available perturbations across all individual screens: {min_perts}")
    print(f"Subsampling all independent screens identically to {min_perts} perturbations for apples-to-apples constraint.")
    
    results = []
    
    # Process Dataset 1
    if 'context' in ad1.obs.columns:
        for c in ad1.obs['context'].unique():
            if pd.isna(c) or str(c).strip() == '': continue
            X = ad1[ad1.obs['context'] == c].X
            res = process_screen_iterations(X, name=f"{name_1}_{c}", min_perts=min_perts, n_iters=5)
            if res: results.append(res)
    else:
        res = process_screen_iterations(ad1.X, name=name_1, min_perts=min_perts, n_iters=5)
        if res: results.append(res)
        
    # Process Dataset 2
    if 'context' in ad2.obs.columns:
        for c in ad2.obs['context'].unique():
            if pd.isna(c) or str(c).strip() == '': continue
            X = ad2[ad2.obs['context'] == c].X
            res = process_screen_iterations(X, name=f"{name_2}_{c}", min_perts=min_perts, n_iters=5)
            if res: results.append(res)
    else:
        res = process_screen_iterations(ad2.X, name=name_2, min_perts=min_perts, n_iters=5)
        if res: results.append(res)
    
    df_results = pd.DataFrame(results)
    print("\n========= Direct Apples-to-Apples Diversity Comparison (Per Screen) =========")
    print(df_results.to_string(index=False))
    
    os.makedirs("results", exist_ok=True)
    out_csv = "results/perturbation_diversity_essential_gw_comparison.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved numerical diversity metrics to {out_csv}")

if __name__ == "__main__":
    
    # Apples-to-Apples Comparison between Two different H5ADs
    H5AD_1 = "./ESSENTIAL_GENES_SCREEN_DATASET.h5ad"
    H5AD_2 = "GENOME_WIDE_SCREEN_DATASET.h5ad"
    
    compare_two_h5ads(H5AD_1, H5AD_2, name_1="Essential", name_2="GW")
