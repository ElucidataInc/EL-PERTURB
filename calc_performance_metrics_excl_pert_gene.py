
"""
Performance metrics with **responding genes excluding the perturbed target** 
wherever a scalar is taken over ``var`` (MSE, WMSE, Pearson, Pearson-pert, Top-K DEG Pearson, 
L1 and cosine **rank** scores, and matrix Frobenius distance on pert–pert similarity).

Results are written to ``uns['performance_metrics_summary_excl_pert_gene']``,
``uns['performance_metrics_per_gene_excl_pert_gene']``, and CSVs with
``*_metrics_excl_pert_gene_*``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def compute_similarity_matrix_frobenius_dist_excl_pert_genes(
    P: np.ndarray,
    T: np.ndarray,
    target_genes: List[str],
    gene_to_idx: Dict[str, int],
) -> float:
    """‖S_pred − S_true‖_F with pairwise cosines that exclude each row’s pert gene."""
    s_pred = _pairwise_cosine_similarity_matrix_excl_pert_genes(P, target_genes, gene_to_idx)
    s_true = _pairwise_cosine_similarity_matrix_excl_pert_genes(T, target_genes, gene_to_idx)
    return float(np.linalg.norm(s_pred - s_true, ord="fro"))


def _pairwise_cosine_similarity_matrix_excl_pert_genes(
    X: np.ndarray,
    target_genes: List[str],
    gene_to_idx: Dict[str, int],
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n, g = int(X.shape[0]), int(X.shape[1])
    pert_idx = [gene_to_idx.get(tg) for tg in target_genes]
    s = np.zeros((n, n), dtype=np.float64)
    eps = 1e-12
    for i in range(n):
        for j in range(n):
            excl = set()
            gi = pert_idx[i]
            gj = pert_idx[j]
            if gi is not None and 0 <= gi < g:
                excl.add(gi)
            if gj is not None and 0 <= gj < g:
                excl.add(gj)
            if not excl:
                xi = X[i].ravel()
                xj = X[j].ravel()
            else:
                m = np.ones(g, dtype=bool)
                for idx in excl:
                    m[idx] = False
                xi = X[i].ravel()[m]
                xj = X[j].ravel()[m]
            ni = float(np.linalg.norm(xi))
            nj = float(np.linalg.norm(xj))
            if xi.size == 0 or ni < eps or nj < eps:
                s[i, j] = 0.0
            else:
                s[i, j] = float(np.dot(xi, xj) / (ni * nj))
    return s


def _bool_mask_excl_pert(g_idx: Optional[int], n_genes: int) -> np.ndarray:
    m = np.ones(n_genes, dtype=bool)
    if g_idx is not None and 0 <= g_idx < n_genes:
        m[g_idx] = False
    return m


def _renorm_weights_excl(w: np.ndarray, g_idx: Optional[int]) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).ravel().copy()
    if g_idx is not None and 0 <= g_idx < w.size:
        w[g_idx] = 0.0
    s = float(w.sum())
    if s < 1e-12:
        return np.ones_like(w) / max(w.size, 1)
    return w / s


def compute_rank_metric(
    preds: np.ndarray,
    truths: np.ndarray,
    target_genes: List[str],
    var_names: List[str],
    metric: str = "l1",
) -> np.ndarray:
    """Same as ``calc_performance_metrics.compute_rank_metric`` (self-target excluded in distances)."""
    from scipy.spatial.distance import cdist

    n = preds.shape[0]
    if n <= 1:
        return np.ones(n)

    gene_to_idx = {gene: idx for idx, gene in enumerate(var_names)}

    if metric == "l1":
        dists = cdist(preds, truths, metric="cityblock")
    elif metric == "cosine":
        dots = np.dot(preds, truths.T)
        preds_norm_sq = np.sum(preds**2, axis=1)
        truths_norm_sq = np.sum(truths**2, axis=1)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    scores = np.zeros(n)
    for i in range(n):
        gene = target_genes[i]
        idx = gene_to_idx.get(gene)

        if metric == "l1":
            if idx is not None:
                row_dists = dists[i, :] - np.abs(preds[i, idx] - truths[:, idx])
            else:
                row_dists = dists[i, :]
        else:
            if idx is not None:
                adj_dots = dots[i, :] - (preds[i, idx] * truths[:, idx])
                adj_p_norm = np.sqrt(np.maximum(0, preds_norm_sq[i] - preds[i, idx] ** 2))
                adj_t_norms = np.sqrt(np.maximum(0, truths_norm_sq - truths[:, idx] ** 2))
                denom = adj_p_norm * adj_t_norms
                mask = denom > 1e-9
                row_dists = np.full(n, 2.0)
                row_dists[mask] = 1.0 - (adj_dots[mask] / denom[mask])
            else:
                denom = np.sqrt(preds_norm_sq[i]) * np.sqrt(truths_norm_sq)
                mask = denom > 1e-9
                row_dists = np.full(n, 2.0)
                row_dists[mask] = 1.0 - (dots[i, mask] / denom[mask])

        true_dist = row_dists[i]
        rank = np.sum(row_dists < true_dist - 1e-12) + 1
        score = (n - rank) / (n - 1)
        scores[i] = score

    return scores


def calc_performance_metrics_excl_pert_gene(
    pred_h5ad: str, truth_h5ad: str, output_prefix: str = "eval_results"
):
    print(f"[excl_pert_gene] Loading predictions: {pred_h5ad}")
    adata_pred = sc.read_h5ad(pred_h5ad)

    print(f"[excl_pert_gene] Loading ground truth: {truth_h5ad}")
    adata_truth = sc.read_h5ad(truth_h5ad)

    if "training_params" in adata_pred.uns:
        params = adata_pred.uns["training_params"]
        split_col = params.get("split_column")
        target_screen = params.get("target_screen")
    else:
        raise ValueError("Critical metadata not found in adata_pred.uns['training_params'].")

    print(f"Target Screen: {target_screen}, Split Column: {split_col}")

    t_score_layers = [l for l in adata_truth.layers if l.startswith("t_scores_")]
    if not t_score_layers:
        raise ValueError(
            f"No layers starting with 't_scores_' found in ground truth. Available layers: {list(adata_truth.layers.keys())}"
        )

    score_layer = None
    for layer in sorted(t_score_layers, key=len, reverse=True):
        suffix = layer.replace("t_scores_", "")
        if split_col.endswith(suffix):
            score_layer = layer
            break

    if score_layer is None:
        print(f"Warning: No exact suffix match for split column '{split_col}'. Attempting fuzzy matching...")
        best_match = None
        max_overlap = 0
        for layer in t_score_layers:
            suffix = layer.replace("t_scores_", "")
            for i in range(1, len(suffix) + 1):
                if split_col.endswith(suffix[-i:]):
                    if i > max_overlap:
                        max_overlap = i
                        best_match = layer
        score_layer = best_match

    if score_layer is None:
        raise ValueError(
            f"Could not find a matching t-score layer for split column '{split_col}'. Available layers: {t_score_layers}"
        )

    print(f"Matched split column '{split_col}' to score layer '{score_layer}'")

    print(f"Computing dynamic weights for WMSE from layer '{score_layer}'...")
    scores = adata_truth.layers[score_layer]
    if hasattr(scores, "toarray"):
        scores = scores.toarray()

    scores_abs = np.abs(scores)
    row_min = np.min(scores_abs, axis=1, keepdims=True)
    row_max = np.max(scores_abs, axis=1, keepdims=True)
    row_range = row_max - row_min
    row_range[row_range < 1e-12] = 1.0

    weights_matrix = (scores_abs - row_min) / row_range
    weights_matrix = weights_matrix**2
    weights_matrix = weights_matrix / (np.sum(weights_matrix, axis=1, keepdims=True) + 1e-12)

    train_target_mask = (adata_truth.obs[split_col] == "train") & (adata_truth.obs["context"] == target_screen)
    if train_target_mask.any():
        train_target_X = adata_truth[train_target_mask].X
        if hasattr(train_target_X, "toarray"):
            train_target_X = train_target_X.toarray()

        cell_counts = adata_truth.obs.loc[train_target_mask, "n_cells"].values
        cell_mean_vector = np.average(train_target_X, axis=0, weights=cell_counts)
        print(f"Computed weighted Cell-Mean Baseline from {train_target_mask.sum()} training samples in {target_screen}.")
    else:
        print(f"Warning: No training samples found for screen {target_screen}. Cell-Mean baseline will be zero.")
        cell_mean_vector = np.zeros(adata_truth.shape[1])

    other_screens_mask = adata_truth.obs["context"] != target_screen
    perturb_mean_dict = {}
    if other_screens_mask.any():
        print("Computing Perturb-Mean baseline from non-target screens...")
        adata_other = adata_truth[other_screens_mask].copy()
        unique_perts_other = adata_other.obs["target_gene"].unique()

        for gene in unique_perts_other:
            gene_mask = adata_other.obs["target_gene"] == gene
            gene_X = adata_other[gene_mask].X
            if hasattr(gene_X, "toarray"):
                gene_X = gene_X.toarray()
            perturb_mean_dict[gene] = np.mean(gene_X, axis=0)
    else:
        print("Warning: No non-target screens found. Perturb-Mean baseline will be zero.")

    test_mask = adata_truth.obs[split_col].isin(["test_seen", "test_unseen"])

    adata_truth_test = adata_truth[test_mask].copy()
    weights_test = weights_matrix[test_mask]

    common_targets = np.intersect1d(adata_pred.obs["target_gene"], adata_truth_test.obs["target_gene"])
    print(f"Aligning on {len(common_targets)} common target genes...")

    adata_pred = adata_pred[adata_pred.obs["target_gene"].isin(common_targets)].copy()
    adata_pred = adata_pred[adata_pred.obs["target_gene"].argsort()].copy()

    truth_sort_idx = adata_truth_test.obs["target_gene"].argsort()
    adata_truth_test = adata_truth_test[truth_sort_idx].copy()
    weights_test = weights_test[truth_sort_idx]

    mask_in_common = adata_truth_test.obs["target_gene"].isin(common_targets)
    adata_truth_test = adata_truth_test[mask_in_common].copy()
    weights_test = weights_test[mask_in_common.values]

    assert np.all(adata_pred.obs["target_gene"].values == adata_truth_test.obs["target_gene"].values), "Alignment Failed!"

    P = adata_pred.X
    T = adata_truth_test.X
    if hasattr(P, "toarray"):
        P = P.toarray()
    if hasattr(T, "toarray"):
        T = T.toarray()

    n_samples = P.shape[0]
    var_names = adata_truth_test.var_names.tolist()
    gene_to_idx = {g: i for i, g in enumerate(var_names)}

    print("Computing Rank Metrics (unchanged; self-coordinate already adjusted)...")
    target_genes = adata_pred.obs["target_gene"].tolist()
    ranks_l1 = compute_rank_metric(P, T, target_genes, var_names, metric="l1")
    ranks_cosine = compute_rank_metric(P, T, target_genes, var_names, metric="cosine")

    B = np.tile(cell_mean_vector, (n_samples, 1))
    ranks_cm_l1 = compute_rank_metric(B, T, target_genes, var_names, metric="l1")
    ranks_cm_cosine = compute_rank_metric(B, T, target_genes, var_names, metric="cosine")

    PM = np.zeros_like(T)
    for i, gene in enumerate(target_genes):
        PM[i] = perturb_mean_dict.get(gene, np.zeros(T.shape[1]))
    ranks_pm_l1 = compute_rank_metric(PM, T, target_genes, var_names, metric="l1")
    ranks_pm_cosine = compute_rank_metric(PM, T, target_genes, var_names, metric="cosine")

    print("Computing per-sample metrics (excluding perturbed gene from MSE/WMSE/Pearson/matrix-relevant features)...")
    metrics_list = []

    degs_dict = {}
    if "top50_degs" in adata_truth.uns and target_screen in adata_truth.uns["top50_degs"]:
        degs_dict = adata_truth.uns["top50_degs"][target_screen].get("by_padj_0.05", {})

    for i in range(n_samples):
        p_i = P[i]
        t_i = T[i]
        w_i = weights_test[i]
        gene = target_genes[i]
        g_idx = gene_to_idx.get(gene)
        n_g = int(t_i.size)
        m_excl = _bool_mask_excl_pert(g_idx, n_g)
        w_exc = _renorm_weights_excl(w_i, g_idx)
        gt_delta = t_i - cell_mean_vector
        pred_delta = p_i - cell_mean_vector

        k_keep = int(m_excl.sum())
        if k_keep < 2:
            mse = wmse = pears = np.nan
            mse_cm = wmse_cm = pears_cm = np.nan
            mse_zero = wmse_zero = np.nan
            pears_pert = pears_pert_zero = np.nan
        else:
            mse = float(mean_squared_error(t_i[m_excl], p_i[m_excl]))
            wmse = float(np.sum(w_exc * (p_i - t_i) ** 2))
            if np.std(p_i[m_excl]) > 1e-9 and np.std(t_i[m_excl]) > 1e-9:
                pears, _ = pearsonr(p_i[m_excl], t_i[m_excl])
                pears = float(pears)
            else:
                pears = 0.0

            mse_cm = float(mean_squared_error(t_i[m_excl], cell_mean_vector[m_excl]))
            wmse_cm = float(np.sum(w_exc * (cell_mean_vector - t_i) ** 2))
            if np.std(cell_mean_vector[m_excl]) > 1e-9 and np.std(t_i[m_excl]) > 1e-9:
                pears_cm, _ = pearsonr(cell_mean_vector[m_excl], t_i[m_excl])
                pears_cm = float(pears_cm)
            else:
                pears_cm = 0.0

            mse_zero = float(np.mean(t_i[m_excl] ** 2))
            wmse_zero = float(np.sum(w_exc * (t_i**2)))

            if np.std(gt_delta[m_excl]) > 1e-9 and np.std(pred_delta[m_excl]) > 1e-9:
                pears_pert, _ = pearsonr(gt_delta[m_excl], pred_delta[m_excl])
                pears_pert = float(pears_pert)
            else:
                pears_pert = 0.0

            cm_neg = -cell_mean_vector
            if np.std(gt_delta[m_excl]) > 1e-12 and np.std(cm_neg[m_excl]) > 1e-12:
                pears_pert_zero, _ = pearsonr(gt_delta[m_excl], cm_neg[m_excl])
                pears_pert_zero = float(pears_pert_zero)
            else:
                pears_pert_zero = 0.0

        pears_zero = np.nan

        is_unseen = adata_pred.obs["test_split"].iloc[i] == "test_unseen"
        if is_unseen:
            mse_pm = np.nan
            wmse_pm = np.nan
            pears_pm = np.nan
            pears_topk_pm = np.nan
            ranks_pm_l1[i] = np.nan
            ranks_pm_cosine[i] = np.nan
            pears_pert_pm = np.nan
        else:
            pm_i = perturb_mean_dict.get(gene, np.zeros_like(t_i))
            if k_keep < 2:
                mse_pm = wmse_pm = pears_pm = np.nan
            else:
                mse_pm = float(mean_squared_error(t_i[m_excl], pm_i[m_excl]))
                wmse_pm = float(np.sum(w_exc * (pm_i - t_i) ** 2))
                if np.std(pm_i[m_excl]) > 1e-9 and np.std(t_i[m_excl]) > 1e-9:
                    pears_pm, _ = pearsonr(pm_i[m_excl], t_i[m_excl])
                    pears_pm = float(pears_pm)
                else:
                    pears_pm = 0.0

            pm_delta = pm_i - cell_mean_vector
            if k_keep >= 2 and np.std(gt_delta[m_excl]) > 1e-12 and np.std(pm_delta[m_excl]) > 1e-12:
                pears_pert_pm, _ = pearsonr(gt_delta[m_excl], pm_delta[m_excl])
                pears_pert_pm = float(pears_pert_pm)
            else:
                pears_pert_pm = 0.0 if k_keep >= 2 else np.nan

        if gene in degs_dict:
            deg_genes = degs_dict[gene]
            deg_indices = [gene_to_idx[g] for g in deg_genes if g in gene_to_idx and g != gene]
            if len(deg_indices) > 2:
                p_deg = p_i[deg_indices]
                t_deg = t_i[deg_indices]
                cm_deg = cell_mean_vector[deg_indices]
                pears_topk, _ = pearsonr(p_deg, t_deg) if np.std(p_deg) > 1e-9 and np.std(t_deg) > 1e-9 else (0.0, 1.0)
                pears_topk_cm, _ = pearsonr(cm_deg, t_deg) if np.std(cm_deg) > 1e-9 and np.std(t_deg) > 1e-9 else (0.0, 1.0)

                if is_unseen:
                    pears_topk_pm = np.nan
                else:
                    pm_i = perturb_mean_dict.get(gene, np.zeros_like(t_i))
                    pm_deg = pm_i[deg_indices]
                    pears_topk_pm, _ = pearsonr(pm_deg, t_deg) if np.std(pm_deg) > 1e-9 and np.std(t_deg) > 1e-9 else (0.0, 1.0)
            else:
                pears_topk = np.nan
                pears_topk_cm = np.nan
                pears_topk_pm = np.nan
        else:
            pears_topk = np.nan
            pears_topk_cm = np.nan
            pears_topk_pm = np.nan

        metrics_list.append(
            {
                "target_gene": gene,
                "test_split": adata_pred.obs["test_split"].iloc[i],
                "mse": mse,
                "wmse": wmse,
                "pearson": pears,
                "pearson_pert": pears_pert,
                "pearson_topk_degs": pears_topk,
                "rank_score_l1": ranks_l1[i],
                "rank_score_cosine": ranks_cosine[i],
                "mse_cell_mean": mse_cm,
                "wmse_cell_mean": wmse_cm,
                "pearson_cell_mean": pears_cm,
                "pearson_topk_degs_cm": pears_topk_cm,
                "rank_score_cm_l1": ranks_cm_l1[i],
                "rank_score_cm_cosine": ranks_cm_cosine[i],
                "mse_zero": mse_zero,
                "wmse_zero": wmse_zero,
                "mse_pm": mse_pm,
                "wmse_pm": wmse_pm,
                "pearson_pm": pears_pm,
                "pearson_pert_pm": pears_pert_pm,
                "pearson_topk_degs_pm": pears_topk_pm,
                "rank_score_pm_l1": ranks_pm_l1[i],
                "rank_score_pm_cosine": ranks_pm_cosine[i],
                "pears_pert_zero": pears_pert_zero,
            }
        )

    df = pd.DataFrame(metrics_list)

    def scale_high(val, base):
        if pd.isna(val) or pd.isna(base):
            return np.nan
        denom = 1.0 - base
        return (val - base) / denom if abs(denom) > 1e-9 else 0.0

    def scale_low(val, base):
        if pd.isna(val) or pd.isna(base):
            return np.nan
        return (base - val) / base if abs(base) > 1e-9 else 0.0

    summary = []
    for group_name in ["test_seen", "test_unseen", "Combined"]:
        if group_name == "Combined":
            subset_df = df
        else:
            subset_df = df[df["test_split"] == group_name]

        if subset_df.empty:
            continue

        group_indices = subset_df.index.tolist()
        P_sub = P[group_indices]
        T_sub = T[group_indices]
        B_sub = B[group_indices]
        sub_target_genes = [target_genes[j] for j in group_indices]

        if len(P_sub) > 1:
            mat_dist = compute_similarity_matrix_frobenius_dist_excl_pert_genes(
                P_sub, T_sub, sub_target_genes, gene_to_idx
            )
            mat_dist_cm = compute_similarity_matrix_frobenius_dist_excl_pert_genes(
                B_sub, T_sub, sub_target_genes, gene_to_idx
            )

            pm_valid_mask = subset_df["test_split"] != "test_unseen"
            pm_indices = subset_df.index[pm_valid_mask].tolist()
            if len(pm_indices) > 1:
                PM_sub = PM[pm_indices]
                T_sub_pm = T[pm_indices]
                tg_pm = [target_genes[j] for j in pm_indices]
                mat_dist_pm = compute_similarity_matrix_frobenius_dist_excl_pert_genes(
                    PM_sub, T_sub_pm, tg_pm, gene_to_idx
                )
            else:
                mat_dist_pm = np.nan
        else:
            mat_dist = np.nan
            mat_dist_cm = np.nan
            mat_dist_pm = np.nan

        b_mse = subset_df["mse_cell_mean"].mean()
        b_wmse = subset_df["wmse_cell_mean"].mean()
        b_pears = subset_df["pearson_cell_mean"].mean()
        b_pears_pert = subset_df["pears_pert_zero"].mean()
        b_pears_topk = subset_df["pearson_topk_degs_cm"].dropna().mean()
        b_rank_l1 = subset_df["rank_score_cm_l1"].mean()
        b_rank_cos = subset_df["rank_score_cm_cosine"].mean()
        b_mat = mat_dist_cm

        s_mse = scale_low(subset_df["mse"].mean(), b_mse)
        s_wmse = scale_low(subset_df["wmse"].mean(), b_wmse)
        s_pears = scale_high(subset_df["pearson"].mean(), b_pears)
        s_pears_pert = scale_high(subset_df["pearson_pert"].mean(), b_pears_pert)
        s_pears_topk = scale_high(subset_df["pearson_topk_degs"].dropna().mean(), b_pears_topk)
        s_rank_l1 = scale_high(subset_df["rank_score_l1"].mean(), b_rank_l1)
        s_rank_cos = scale_high(subset_df["rank_score_cosine"].mean(), b_rank_cos)
        s_mat = scale_low(mat_dist, b_mat)

        all_scaled = [s_mse, s_wmse, s_pears, s_pears_pert, s_pears_topk, s_rank_l1, s_rank_cos, s_mat]
        global_score = np.nanmean(all_scaled)

        spec_list = [s_wmse, s_rank_l1, s_rank_cos, s_mat, s_pears_pert]
        spec_score = np.nanmean(spec_list)

        if group_name in ["test_unseen", "Combined"]:
            s_mse_pm = s_wmse_pm = s_pears_pm = s_pears_pert_pm = s_pears_topk_pm = s_rank_l1_pm = s_rank_cos_pm = s_mat_pm = np.nan
            global_pm = spec_pm = np.nan
        else:
            s_mse_pm = scale_low(subset_df["mse_pm"].mean(), b_mse)
            s_wmse_pm = scale_low(subset_df["wmse_pm"].mean(), b_wmse)
            s_pears_pm = scale_high(subset_df["pearson_pm"].mean(), b_pears)
            s_pears_pert_pm = scale_high(subset_df["pearson_pert_pm"].mean(), b_pears_pert)
            s_pears_topk_pm = scale_high(subset_df["pearson_topk_degs_pm"].dropna().mean(), b_pears_topk)
            s_rank_l1_pm = scale_high(subset_df["rank_score_pm_l1"].mean(), b_rank_l1)
            s_rank_cos_pm = scale_high(subset_df["rank_score_pm_cosine"].mean(), b_rank_cos)
            s_mat_pm = scale_low(mat_dist_pm, b_mat)

            all_scaled_pm = [
                s_mse_pm,
                s_wmse_pm,
                s_pears_pm,
                s_pears_pert_pm,
                s_pears_topk_pm,
                s_rank_l1_pm,
                s_rank_cos_pm,
                s_mat_pm,
            ]
            global_pm = np.nanmean(all_scaled_pm)
            spec_pm = np.nanmean([s_wmse_pm, s_rank_l1_pm, s_rank_cos_pm, s_mat_pm, s_pears_pert_pm])

        if group_name in ["test_unseen", "Combined"]:
            m_mse_pm = m_wmse_pm = m_pears_pm = m_pears_pert_pm = m_pears_topk_pm = m_rank_l1_pm = m_rank_cos_pm = m_mat_pm = np.nan
        else:
            m_mse_pm = subset_df["mse_pm"].mean()
            m_wmse_pm = subset_df["wmse_pm"].mean()
            m_pears_pm = subset_df["pearson_pm"].mean()
            m_pears_pert_pm = subset_df["pearson_pert_pm"].mean()
            m_pears_topk_pm = subset_df["pearson_topk_degs_pm"].dropna().mean()
            m_rank_l1_pm = subset_df["rank_score_pm_l1"].mean()
            m_rank_cos_pm = subset_df["rank_score_pm_cosine"].mean()
            m_mat_pm = mat_dist_pm

        summary.append(
            {
                "Group": group_name,
                "N": len(subset_df),
                "Mean_MSE": subset_df["mse"].mean(),
                "Mean_WMSE": subset_df["wmse"].mean(),
                "Mean_Pearson": subset_df["pearson"].mean(),
                "Mean_Pearson_Pert": subset_df["pearson_pert"].mean(),
                "Mean_Pearson_TopK_DEGs": subset_df["pearson_topk_degs"].dropna().mean(),
                "Mean_Rank_L1": subset_df["rank_score_l1"].mean(),
                "Mean_Rank_Cosine": subset_df["rank_score_cosine"].mean(),
                "Matrix_Dist": mat_dist,
                "Mean_MSE_CM": subset_df["mse_cell_mean"].mean(),
                "Mean_WMSE_CM": subset_df["wmse_cell_mean"].mean(),
                "Mean_Pearson_CM": subset_df["pearson_cell_mean"].mean(),
                "Mean_Pearson_TopK_DEGs_CM": subset_df["pearson_topk_degs_cm"].dropna().mean(),
                "Mean_Rank_L1_CM": subset_df["rank_score_cm_l1"].mean(),
                "Mean_Rank_Cosine_CM": subset_df["rank_score_cm_cosine"].mean(),
                "Matrix_Dist_CM": mat_dist_cm,
                "Mean_MSE_Zero": subset_df["mse_zero"].mean(),
                "Mean_WMSE_Zero": subset_df["wmse_zero"].mean(),
                "Mean_MSE_PM": m_mse_pm,
                "Mean_WMSE_PM": m_wmse_pm,
                "Mean_Pearson_PM": m_pears_pm,
                "Mean_Pearson_Pert_PM": m_pears_pert_pm,
                "Mean_Pearson_TopK_DEGs_PM": m_pears_topk_pm,
                "Mean_Rank_L1_PM": m_rank_l1_pm,
                "Mean_Rank_Cosine_PM": m_rank_cos_pm,
                "Matrix_Dist_PM": m_mat_pm,
                "Global_Score": global_score,
                "Specificity_Score": spec_score,
                "Global_Score_PM": global_pm,
                "Specificity_Score_PM": spec_pm,
            }
        )
    summary_df = pd.DataFrame(summary)
    print("\n--- Summary Results (excl perturbed gene from responding-gene metrics) ---")
    print(summary_df.T)

    print(f"Saving excl_pert_gene metrics back to {pred_h5ad}...")
    for old_key in [
        "performance_metrics_excl_pert_gene",
        "performance_metrics_summary_excl_pert_gene",
        "performance_metrics_per_gene_excl_pert_gene",
    ]:
        if old_key in adata_pred.uns:
            del adata_pred.uns[old_key]

    if "test_split" in df.columns:
        df["test_split"] = df["test_split"].astype(str)

    adata_pred.uns["performance_metrics_summary_excl_pert_gene"] = summary_df
    adata_pred.uns["performance_metrics_per_gene_excl_pert_gene"] = df

    os.makedirs("./results", exist_ok=True)
    file_prefix = os.path.splitext(os.path.basename(pred_h5ad))[0]

    df.to_csv(f"./results/{file_prefix}_metrics_excl_pert_gene_per_gene.csv", index=False)
    summary_df.to_csv(f"./results/{file_prefix}_metrics_excl_pert_gene_summary.csv", index=False)

    adata_pred.write_h5ad(pred_h5ad)
    print(f"Saved results with prefix: {file_prefix} (excl_pert_gene)")
    print("Done.")

    return df, summary_df


if __name__ == "__main__":

    # ------------------------------------------------------------
    pred_path = "./prediction.h5ad"
    truth_path = "./dataset.h5ad"
    calc_performance_metrics_excl_pert_gene(pred_path, truth_path)
