"""
PRESAGE Full Data Preparation Script (SC -> Pseudobulk)
======================================================
Supports two input modes:

  SINGLE FILE (--scdata_file):
    Reads one large merged h5ad. Uses backed mode for memory efficiency.

  PER-CELL-TYPE FILES (--celltype_dir + --manifest):
    Reads separate h5ad per cell type. Processes one at a time.
    Peak memory = largest single cell type (~30-40 GB).
    Much faster for kpdp variant — no 130 GB reads.

    Generate per-cell-type files with split_by_celltype.py first.

DEG options:
  --skip_degs: Write stub DEGs (instant, safe if not using PRESAGE evaluator)
  --precomputed_degs: Path to existing merged.degs.json
  --degs_only: Compute DEGs only and exit

Usage:
  # Split the merged file first (one-time)
  python split_by_celltype.py --input_h5ad merged.h5ad --output_dir data/per_ct/

  # Then run prep (fast, low memory)
  python prepare_presage_data_memopt.py \
      --celltype_dir data/per_ct/ \
      --split_col K562_TF_10_UF_10_rs_2_random \
      --target_name K562 --stage hpo --variant kpdp \
      --mode target_fold_5 --skip_degs \
      --prior_type all --prior_by_group cell_type \
      --n_nmf_embedding 128 \
      --output_dir out/hpo_kpdp_K562_TF10/
"""

import argparse
import gc
import json
import os
import pickle as pkl
import re
import shutil
import time

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import KFold
from tqdm import tqdm


# =========================================================================
# Utility
# =========================================================================

def _mem_gb_X(X):
    if sp.issparse(X):
        return X.data.nbytes / 1e9
    return X.nbytes / 1e9


def _load_backed(path):
    t0 = time.time()
    adata = sc.read_h5ad(path, backed="r")
    print(f"  Opened backed in {time.time()-t0:.0f}s: {adata.shape[0]:,} x {adata.shape[1]:,}")
    return adata


def _slice_to_memory(adata_backed, mask):
    t0 = time.time()
    indices = np.where(mask)[0]
    subset = adata_backed[indices].to_memory()
    print(f"  Sliced {len(indices):,} rows in {time.time()-t0:.0f}s (~{_mem_gb_X(subset.X):.1f} GB)")
    return subset


def _read_full(path):
    t0 = time.time()
    adata = sc.read_h5ad(path)
    print(f"  Read {path} in {time.time()-t0:.0f}s: {adata.shape[0]:,} x {adata.shape[1]:,} (~{_mem_gb_X(adata.X):.1f} GB)")
    return adata


# =========================================================================
# Cell-type file management
# =========================================================================

def load_manifest(celltype_dir):
    manifest_path = os.path.join(celltype_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    # Auto-discover h5ad files
    manifest = {}
    for fname in sorted(os.listdir(celltype_dir)):
        if fname.endswith(".h5ad"):
            ct = fname.replace(".h5ad", "")
            manifest[ct] = os.path.join(celltype_dir, fname)
    return manifest


def get_all_obs(manifest):
    """Read obs from all cell-type files (backed mode, no X loaded)."""
    all_obs = []
    for ct, path in manifest.items():
        adata_b = sc.read_h5ad(path, backed="r")
        obs = adata_b.obs.copy()
        adata_b.file.close()
        del adata_b
        all_obs.append(obs)
    combined = pd.concat(all_obs, axis=0)
    del all_obs; gc.collect()
    return combined


def get_filtered_obs(manifest, variant, target_name, target_key, stage, split_col):
    """Get combined obs after applying stage + variant filters."""
    if variant in ("only", "pplus"):
        # Only need the target cell type file
        relevant = {target_name: manifest[target_name]}
    else:
        relevant = manifest

    all_obs = []
    for ct, path in relevant.items():
        adata_b = sc.read_h5ad(path, backed="r")
        obs = adata_b.obs.copy()
        adata_b.file.close()
        del adata_b

        # Stage filter
        if stage == "hpo":
            obs = obs[obs[split_col].isin(["train", "control"])]
        else:
            obs = obs[~obs[split_col].isin(["excluded"])]

        all_obs.append(obs)

    combined = pd.concat(all_obs, axis=0)
    del all_obs; gc.collect()
    return combined, relevant


# =========================================================================
# DEG computation
# =========================================================================

def compute_degs_for_celltype(
    adata, deg_path, cell_type, cell_type_key, pert_key, pert_group_key, control_pert
):
    sdata = adata[adata.obs[cell_type_key] == cell_type, :]
    vc = sdata.obs[pert_key].value_counts()
    if vc.get(control_pert, 0) > 5000:
        ctrl_sample = (
            sdata.obs[sdata.obs[pert_key] == control_pert]
            .sample(5000, random_state=0).index.tolist()
        )
        pert_cells = sdata.obs[sdata.obs[pert_key] != control_pert].index.tolist()
        sdata = sdata[ctrl_sample + pert_cells, :].copy()

    degs = {}
    perts = set(sdata.obs.loc[sdata.obs[pert_key] != control_pert, pert_group_key])
    for pert in tqdm(perts, desc=f"DEGs [{cell_type}]"):
        target_file = os.path.join(deg_path, f"{pert}.json")
        if os.path.isfile(target_file):
            with open(target_file) as fp:
                degs[pert] = json.load(fp)
            continue

        temp = sdata[
            sdata.obs[pert_group_key].isin([cell_type + ":" + control_pert, pert]), :
        ].copy()
        sc.tl.rank_genes_groups(
            temp, groupby=pert_group_key,
            reference=cell_type + ":" + control_pert,
            method="t-test_overestim_var", rankby_abs=True,
        )
        result = (
            pd.DataFrame(temp.uns["rank_genes_groups"]["names"])
            .loc[:999, pert].to_list()
        )
        degs[pert] = result
        with open(target_file, "w") as fp:
            json.dump(result, fp)
        del temp
    return degs


# =========================================================================
# Pseudobulk computation
# =========================================================================

def compute_pseudobulk_from_adata(adata, pert_group_key, target_key, pert_key):
    """Pseudobulk a single in-memory adata. Returns (X_array, obs_rows, ncells_dict)."""
    X = adata.X
    obs = adata.obs
    groups = obs[pert_group_key]
    unique_groups = groups.unique()

    pbulk_X = np.zeros((len(unique_groups), X.shape[1]), dtype=np.float32)
    pbulk_obs_rows = []
    ncells = {}
    obs_idx = np.arange(X.shape[0])

    for i, gname in enumerate(unique_groups):
        mask = (groups == gname).values
        row_idx = obs_idx[mask]
        ncells[str(gname)] = int(len(row_idx))

        sub_X = X[row_idx]
        if sp.issparse(sub_X):
            pbulk_X[i] = np.asarray(sub_X.mean(axis=0)).ravel()
        else:
            pbulk_X[i] = np.mean(sub_X, axis=0).ravel()

        first = obs.iloc[row_idx[0]]
        row = {pert_group_key: gname, pert_key: first[pert_key], target_key: first[target_key]}
        for col in obs.columns:
            if col not in row:
                row[col] = first[col]
        pbulk_obs_rows.append(row)

    return pbulk_X, pbulk_obs_rows, ncells


def pseudobulk_from_celltype_files(manifest, relevant_cts, obs_filtered,
                                    pert_group_key, target_key, pert_key, control_key,
                                    stage, split_col):
    """Compute pseudobulk by loading one cell type file at a time."""
    print("  Pseudobulking per cell type...")

    all_X, all_obs_rows, all_ncells = [], [], {}
    var = None

    for ct, path in relevant_cts.items():
        print(f"\n  --- {ct} ---")
        adata = _read_full(path)

        # Apply stage filter
        if stage == "hpo":
            adata = adata[adata.obs[split_col].isin(["train", "control"]), :]
        else:
            adata = adata[~adata.obs[split_col].isin(["excluded"]), :]

        if var is None:
            var = adata.var.copy()

        pX, prows, ncells = compute_pseudobulk_from_adata(adata, pert_group_key, target_key, pert_key)
        all_X.append(pX)
        all_obs_rows.extend(prows)
        all_ncells.update(ncells)

        del adata, pX; gc.collect()

    # Combine
    pbulk_X = np.concatenate(all_X, axis=0)
    pbulk_obs = pd.DataFrame(all_obs_rows)
    pbulk_obs.index = [str(i) for i in range(len(pbulk_obs))]

    # Coerce types for h5ad
    for col in pbulk_obs.columns:
        dtype = pbulk_obs[col].dtype
        if dtype == object:
            pbulk_obs[col] = pbulk_obs[col].fillna("NA").astype(str)
        elif dtype == bool or (hasattr(dtype, "name") and "bool" in dtype.name.lower()):
            pbulk_obs[col] = pbulk_obs[col].astype(str)

    pbulk_adata = sc.AnnData(X=pbulk_X, obs=pbulk_obs, var=var)
    print(f"\n  Combined pseudobulk: {pbulk_adata.shape[0]} x {pbulk_adata.shape[1]}")
    return pbulk_adata, all_ncells


def compute_pseudobulk_from_backed(backed_path, keep_mask, pert_group_key,
                                    target_key, pert_key, control_key):
    """Compute pseudobulk from a single merged h5ad using backed mode.
    
    Loads cells in batches per perturbation_group to avoid loading the full matrix.
    """
    print("  Pseudobulking from backed h5ad...")
    adata_b = _load_backed(backed_path)
    obs = adata_b.obs.copy()
    n_genes = adata_b.shape[1]
    var = adata_b.var.copy()

    if keep_mask is not None:
        valid_indices = np.where(keep_mask)[0]
        obs_filtered = obs.iloc[valid_indices]
    else:
        valid_indices = np.arange(len(obs))
        obs_filtered = obs

    groups = obs_filtered[pert_group_key]
    unique_groups = groups.unique()
    print(f"  {len(unique_groups):,} perturbation_groups")

    ncells_per_pert = {}
    pbulk_X = np.zeros((len(unique_groups), n_genes), dtype=np.float32)
    pbulk_obs_rows = []

    BATCH_SIZE = 200
    n_batches = int(np.ceil(len(unique_groups) / BATCH_SIZE))

    for batch_idx in tqdm(range(n_batches), desc="  Pseudobulk batches"):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(unique_groups))
        batch_groups = unique_groups[batch_start:batch_end]

        batch_mask_filtered = groups.isin(batch_groups).values
        batch_global_indices = valid_indices[batch_mask_filtered]

        batch_adata = adata_b[batch_global_indices].to_memory()
        batch_X = batch_adata.X
        batch_obs = batch_adata.obs
        batch_groups_col = batch_obs[pert_group_key]

        for i_in_batch, group_name in enumerate(batch_groups):
            i_global = batch_start + i_in_batch
            group_mask = (batch_groups_col == group_name).values
            n_cells = group_mask.sum()
            ncells_per_pert[str(group_name)] = int(n_cells)

            sub_X = batch_X[group_mask]
            if sp.issparse(sub_X):
                pbulk_X[i_global] = np.asarray(sub_X.mean(axis=0)).ravel()
            else:
                pbulk_X[i_global] = np.mean(sub_X, axis=0).ravel()

            first_row = batch_obs[group_mask].iloc[0]
            obs_row = {pert_group_key: group_name, pert_key: first_row[pert_key],
                       target_key: first_row[target_key]}
            for col in batch_obs.columns:
                if col not in obs_row:
                    obs_row[col] = first_row[col]
            pbulk_obs_rows.append(obs_row)

        del batch_adata, batch_X; gc.collect()

    adata_b.file.close()
    del adata_b; gc.collect()

    pbulk_obs = pd.DataFrame(pbulk_obs_rows)
    pbulk_obs.index = [str(i) for i in range(len(pbulk_obs))]

    for col in pbulk_obs.columns:
        dtype = pbulk_obs[col].dtype
        if dtype == object:
            pbulk_obs[col] = pbulk_obs[col].fillna("NA").astype(str)
        elif dtype == bool or (hasattr(dtype, "name") and "bool" in dtype.name.lower()):
            pbulk_obs[col] = pbulk_obs[col].astype(str)

    pbulk_adata = sc.AnnData(X=pbulk_X, obs=pbulk_obs, var=var)
    print(f"  Result: {pbulk_adata.shape[0]} x {pbulk_adata.shape[1]}")
    return pbulk_adata, ncells_per_pert


# =========================================================================
# Embedding computation
# =========================================================================

INCREMENTAL_PCA_THRESHOLD = 900_000


def compute_coexpression_embedding(adata, n_components):
    n_cells, n_genes = adata.shape
    n_comp = min(n_cells, n_genes, n_components)
    print(f"    Coexpression PCA: {n_cells:,} cells x {n_genes:,} genes -> {n_comp} components")
    X = adata.X

    if n_cells > INCREMENTAL_PCA_THRESHOLD:
        print(f"    Using IncrementalPCA")
        batch_size = max(n_comp * 2, 1024)
        ipca = IncrementalPCA(n_components=n_comp, batch_size=batch_size)
        n_batches = int(np.ceil(n_cells / batch_size))
        for b in tqdm(range(n_batches), desc="    IncrementalPCA"):
            s, e = b * batch_size, min((b + 1) * batch_size, n_cells)
            chunk = X[s:e]
            if sp.issparse(chunk):
                chunk = np.asarray(chunk.todense())
            ipca.partial_fit(chunk)
        emb = ipca.components_.T
    else:
        if sp.issparse(X):
            X = np.asarray(X.todense())
        emb = PCA(n_components=n_comp).fit(X).components_.T

    emb = np.pad(emb, ((0, 0), (0, n_components - emb.shape[1])), constant_values=0)
    return pd.DataFrame(emb, index=adata.var.index)


def compute_transpose_matrix_embedding(adata, pert_field, n_components):
    X = adata.X
    groups = adata.obs[pert_field]
    unique_perts = groups.unique()
    avg_mat = np.zeros((len(unique_perts), X.shape[1]), dtype=np.float64)
    obs_idx = np.arange(X.shape[0])

    for i, pn in enumerate(unique_perts):
        sub = X[obs_idx[(groups == pn).values]]
        avg_mat[i] = np.asarray(sub.mean(axis=0)).ravel() if sp.issparse(sub) else np.mean(sub, axis=0).ravel()

    avg_df = pd.DataFrame(avg_mat, index=unique_perts, columns=adata.var.index)
    n_comp = min(*avg_df.shape, n_components)
    print(f"    Transpose PCA: {avg_df.shape[0]:,} perts x {avg_df.shape[1]:,} genes -> {n_comp} components")
    emb = PCA(n_components=n_comp).fit(avg_df).components_.T
    emb = np.pad(emb, ((0, 0), (0, n_components - emb.shape[1])), constant_values=0)
    return pd.DataFrame(emb, index=avg_df.columns)


def compute_all_fold_embeddings_from_celltype_files(
    relevant_cts, fold_infos, prior_type, prior_by_group,
    pert_key, pert_group_key, control_key, n_nmf_embedding,
    stage, split_col, emb_dir
):
    """Compute embeddings for ALL folds, loading each cell type file only ONCE.
    
    For --prior_by_group cell_type:
      For each cell type file:
        1. Load into memory (~30 GB)
        2. Apply stage filter
        3. For each fold: mask to that fold's training perts + controls, compute PCA
        4. Free the file
      Total disk reads: len(relevant_cts) instead of len(relevant_cts) × N_folds
    
    Returns list of {"fold": N, "coexpression_emb": path, "transpose_matrix_emb": path}
    """
    n_folds = len(fold_infos)

    if prior_type == "kp":
        print("  prior_type='kp': skipping all embeddings")
        return [{"fold": fi["fold"], "coexpression_emb": None, "transpose_matrix_emb": None}
                for fi in fold_infos]

    is_grouped = (prior_by_group is not None and str(prior_by_group) != "None")

    if is_grouped:
        # Accumulate per-fold, per-celltype embeddings
        fold_coex = {fi["fold"]: {} for fi in fold_infos}
        fold_trans = {fi["fold"]: {} for fi in fold_infos}

        for ct, path in relevant_cts.items():
            print(f"\n  === {ct} (computing all {n_folds} folds from single read) ===")
            adata = _read_full(path)

            # Stage filter once
            if stage == "hpo":
                adata = adata[adata.obs[split_col].isin(["train", "control"]), :]
            else:
                adata = adata[~adata.obs[split_col].isin(["excluded"]), :]
            print(f"  {ct} after stage filter: {adata.shape[0]:,} cells")

            for fi in fold_infos:
                fold_idx = fi["fold"]
                print(f"\n  --- {ct} | Fold {fold_idx} ---")

                train_mask = adata.obs[pert_group_key].isin(fi["train_perts"])
                ctrl_mask = adata.obs[pert_key] == control_key
                fold_adata = adata[train_mask | ctrl_mask]
                print(f"    {fold_adata.shape[0]:,} cells")

                fold_coex[fold_idx][ct] = compute_coexpression_embedding(fold_adata, n_nmf_embedding)
                fold_trans[fold_idx][ct] = compute_transpose_matrix_embedding(fold_adata, pert_key, n_nmf_embedding)
                del fold_adata; gc.collect()

            del adata; gc.collect()

        # Save
        emb_paths = []
        for fi in fold_infos:
            fold_idx = fi["fold"]
            label = f"fold_{fold_idx}_" if n_folds > 1 else ""

            coex_path = os.path.join(emb_dir, f"{label}precomputed_coexpression_emb.pkl")
            with open(coex_path, "wb") as f:
                pkl.dump(fold_coex[fold_idx], f, protocol=pkl.HIGHEST_PROTOCOL)

            trans_path = os.path.join(emb_dir, f"{label}precomputed_transpose_matrix_emb.pkl")
            with open(trans_path, "wb") as f:
                pkl.dump(fold_trans[fold_idx], f, protocol=pkl.HIGHEST_PROTOCOL)

            print(f"  Fold {fold_idx}: saved -> {coex_path}, {trans_path}")
            emb_paths.append({"fold": fold_idx, "coexpression_emb": coex_path, "transpose_matrix_emb": trans_path})

        del fold_coex, fold_trans; gc.collect()
        return emb_paths

    else:
        # Ungrouped: need all cell types combined per fold
        # Load all into memory, then compute per fold
        all_adatas = []
        for ct, path in relevant_cts.items():
            adata = _read_full(path)
            if stage == "hpo":
                adata = adata[adata.obs[split_col].isin(["train", "control"]), :]
            else:
                adata = adata[~adata.obs[split_col].isin(["excluded"]), :]
            all_adatas.append(adata)
        combined = sc.concat(all_adatas)
        del all_adatas; gc.collect()

        emb_paths = []
        for fi in fold_infos:
            fold_idx = fi["fold"]
            label = f"fold_{fold_idx}_" if n_folds > 1 else ""
            print(f"\n  --- Fold {fold_idx} (ungrouped) ---")

            train_mask = combined.obs[pert_group_key].isin(fi["train_perts"])
            ctrl_mask = combined.obs[pert_key] == control_key
            fold_adata = combined[train_mask | ctrl_mask]

            coex = compute_coexpression_embedding(fold_adata, n_nmf_embedding)
            trans = compute_transpose_matrix_embedding(fold_adata, pert_key, n_nmf_embedding)
            del fold_adata; gc.collect()

            coex_path = os.path.join(emb_dir, f"{label}precomputed_coexpression_emb.pkl")
            with open(coex_path, "wb") as f:
                pkl.dump(coex, f, protocol=pkl.HIGHEST_PROTOCOL)
            trans_path = os.path.join(emb_dir, f"{label}precomputed_transpose_matrix_emb.pkl")
            with open(trans_path, "wb") as f:
                pkl.dump(trans, f, protocol=pkl.HIGHEST_PROTOCOL)

            print(f"  Fold {fold_idx}: saved -> {coex_path}, {trans_path}")
            emb_paths.append({"fold": fold_idx, "coexpression_emb": coex_path, "transpose_matrix_emb": trans_path})
            del coex, trans; gc.collect()

        del combined; gc.collect()
        return emb_paths


# Keep the old single-fold function for backed mode fallback
def compute_embeddings_from_celltype_files(
    relevant_cts, obs_filtered, train_pert_groups, prior_type, prior_by_group,
    pert_key, pert_group_key, control_key, n_nmf_embedding,
    stage, split_col, emb_dir, label=""
):
    """Compute embeddings loading one cell type at a time.
    
    For --prior_by_group cell_type (the common case), each cell type's
    embeddings are computed independently — only that cell type's file is
    in memory. Perfect for low-memory parallel execution.
    """
    if prior_type == "kp":
        print(f"  {label}prior_type='kp': skipping")
        return None, None

    print(f"  {label}Computing embeddings")

    is_grouped = (prior_by_group is not None and str(prior_by_group) != "None")

    if is_grouped:
        # Compute per cell type — load one file at a time
        coex_dict, transpose_dict = {}, {}
        for ct, path in relevant_cts.items():
            print(f"\n  {label}--- {ct} ---")
            adata = _read_full(path)

            # Apply stage filter
            if stage == "hpo":
                adata = adata[adata.obs[split_col].isin(["train", "control"]), :]
            else:
                adata = adata[~adata.obs[split_col].isin(["excluded"]), :]

            # Apply train perts + control filter
            train_mask = adata.obs[pert_group_key].isin(train_pert_groups)
            ctrl_mask = adata.obs[pert_key] == control_key
            adata = adata[train_mask | ctrl_mask]
            print(f"  {label}{ct}: {adata.shape[0]:,} cells for embeddings")

            coex_dict[ct] = compute_coexpression_embedding(adata, n_nmf_embedding)
            transpose_dict[ct] = compute_transpose_matrix_embedding(adata, pert_key, n_nmf_embedding)
            del adata; gc.collect()

        coex_out, transpose_out = coex_dict, transpose_dict
    else:
        # Ungrouped: need all cell types in memory at once
        all_adatas = []
        for ct, path in relevant_cts.items():
            adata = _read_full(path)
            if stage == "hpo":
                adata = adata[adata.obs[split_col].isin(["train", "control"]), :]
            else:
                adata = adata[~adata.obs[split_col].isin(["excluded"]), :]
            train_mask = adata.obs[pert_group_key].isin(train_pert_groups)
            ctrl_mask = adata.obs[pert_key] == control_key
            all_adatas.append(adata[train_mask | ctrl_mask])

        combined = sc.concat(all_adatas)
        del all_adatas; gc.collect()
        coex_out = compute_coexpression_embedding(combined, n_nmf_embedding)
        transpose_out = compute_transpose_matrix_embedding(combined, pert_key, n_nmf_embedding)
        del combined; gc.collect()

    coex_path = os.path.join(emb_dir, f"{label}precomputed_coexpression_emb.pkl")
    with open(coex_path, "wb") as f:
        pkl.dump(coex_out, f, protocol=pkl.HIGHEST_PROTOCOL)

    transpose_path = os.path.join(emb_dir, f"{label}precomputed_transpose_matrix_emb.pkl")
    with open(transpose_path, "wb") as f:
        pkl.dump(transpose_out, f, protocol=pkl.HIGHEST_PROTOCOL)

    print(f"  {label}Saved -> {coex_path}, {transpose_path}")
    del coex_out, transpose_out; gc.collect()
    return coex_path, transpose_path


# Fallback: backed mode for single merged file
def compute_all_fold_embeddings_backed(
    backed_path, keep_mask, fold_infos, prior_type, prior_by_group, pert_key,
    pert_group_key, control_key, n_nmf_embedding, emb_dir
):
    """Compute embeddings for ALL folds from a single backed h5ad.
    
    Opens the file once, slices the stage-filtered data into memory,
    then computes all folds' PCA from that single in-memory copy.
    1 disk read instead of N.
    """
    n_folds = len(fold_infos)

    if prior_type == "kp":
        print("  prior_type='kp': skipping all embeddings")
        return [{"fold": fi["fold"], "coexpression_emb": None, "transpose_matrix_emb": None}
                for fi in fold_infos]

    # Load the stage-filtered data once
    print(f"  Loading stage-filtered data from backed h5ad...")
    adata_b = _load_backed(backed_path)
    obs = adata_b.obs

    if keep_mask is not None:
        adata = _slice_to_memory(adata_b, keep_mask)
    else:
        adata = adata_b[:].to_memory()
    adata_b.file.close()
    del adata_b; gc.collect()
    print(f"  In memory: {adata.shape[0]:,} cells (~{_mem_gb_X(adata.X):.1f} GB)")

    is_grouped = (prior_by_group is not None and str(prior_by_group) != "None")

    if is_grouped:
        groups = sorted(adata.obs[prior_by_group].unique().tolist())
        fold_coex = {fi["fold"]: {} for fi in fold_infos}
        fold_trans = {fi["fold"]: {} for fi in fold_infos}

        for g in groups:
            print(f"\n  === Group: {g} (all {n_folds} folds) ===")
            g_adata = adata[adata.obs[prior_by_group] == g, :]

            for fi in fold_infos:
                fold_idx = fi["fold"]
                print(f"\n  --- {g} | Fold {fold_idx} ---")
                train_mask = g_adata.obs[pert_group_key].isin(fi["train_perts"])
                ctrl_mask = g_adata.obs[pert_key] == control_key
                fold_adata = g_adata[train_mask | ctrl_mask]
                print(f"    {fold_adata.shape[0]:,} cells")

                fold_coex[fold_idx][g] = compute_coexpression_embedding(fold_adata, n_nmf_embedding)
                fold_trans[fold_idx][g] = compute_transpose_matrix_embedding(fold_adata, pert_key, n_nmf_embedding)
                del fold_adata; gc.collect()

            del g_adata; gc.collect()

        del adata; gc.collect()

        emb_paths = []
        for fi in fold_infos:
            fold_idx = fi["fold"]
            label = f"fold_{fold_idx}_" if n_folds > 1 else ""

            coex_path = os.path.join(emb_dir, f"{label}precomputed_coexpression_emb.pkl")
            with open(coex_path, "wb") as f:
                pkl.dump(fold_coex[fold_idx], f, protocol=pkl.HIGHEST_PROTOCOL)

            trans_path = os.path.join(emb_dir, f"{label}precomputed_transpose_matrix_emb.pkl")
            with open(trans_path, "wb") as f:
                pkl.dump(fold_trans[fold_idx], f, protocol=pkl.HIGHEST_PROTOCOL)

            print(f"  Fold {fold_idx}: saved -> {coex_path}, {trans_path}")
            emb_paths.append({"fold": fold_idx, "coexpression_emb": coex_path, "transpose_matrix_emb": trans_path})

        del fold_coex, fold_trans; gc.collect()
        return emb_paths

    else:
        emb_paths = []
        for fi in fold_infos:
            fold_idx = fi["fold"]
            label = f"fold_{fold_idx}_" if n_folds > 1 else ""
            print(f"\n  --- Fold {fold_idx} (ungrouped) ---")

            train_mask = adata.obs[pert_group_key].isin(fi["train_perts"])
            ctrl_mask = adata.obs[pert_key] == control_key
            fold_adata = adata[train_mask | ctrl_mask]

            coex = compute_coexpression_embedding(fold_adata, n_nmf_embedding)
            trans = compute_transpose_matrix_embedding(fold_adata, pert_key, n_nmf_embedding)
            del fold_adata; gc.collect()

            coex_path = os.path.join(emb_dir, f"{label}precomputed_coexpression_emb.pkl")
            with open(coex_path, "wb") as f:
                pkl.dump(coex, f, protocol=pkl.HIGHEST_PROTOCOL)
            trans_path = os.path.join(emb_dir, f"{label}precomputed_transpose_matrix_emb.pkl")
            with open(trans_path, "wb") as f:
                pkl.dump(trans, f, protocol=pkl.HIGHEST_PROTOCOL)

            print(f"  Fold {fold_idx}: saved -> {coex_path}, {trans_path}")
            emb_paths.append({"fold": fold_idx, "coexpression_emb": coex_path, "transpose_matrix_emb": trans_path})
            del coex, trans; gc.collect()

        del adata; gc.collect()
        return emb_paths


# Keep for backward compatibility (single-fold use cases)
def compute_embeddings_backed(
    backed_path, keep_mask, train_pert_groups, prior_type, prior_by_group, pert_key,
    pert_group_key, control_key, n_nmf_embedding, emb_dir, label=""
):
    if prior_type == "kp":
        return None, None

    adata_b = _load_backed(backed_path)
    obs = adata_b.obs
    train_mask = obs[pert_group_key].isin(train_pert_groups).values
    ctrl_mask = (obs[pert_key] == control_key).values
    emb_mask = (train_mask | ctrl_mask)
    if keep_mask is not None:
        emb_mask = emb_mask & keep_mask

    train_adata = _slice_to_memory(adata_b, emb_mask)
    adata_b.file.close()
    del adata_b; gc.collect()

    is_grouped = (prior_by_group is not None and str(prior_by_group) != "None")
    if is_grouped:
        coex_dict, transpose_dict = {}, {}
        for g in sorted(train_adata.obs[prior_by_group].unique().tolist()):
            sub = train_adata[train_adata.obs[prior_by_group] == g, :]
            coex_dict[g] = compute_coexpression_embedding(sub, n_nmf_embedding)
            transpose_dict[g] = compute_transpose_matrix_embedding(sub, pert_key, n_nmf_embedding)
            del sub; gc.collect()
        coex_out, transpose_out = coex_dict, transpose_dict
    else:
        coex_out = compute_coexpression_embedding(train_adata, n_nmf_embedding)
        transpose_out = compute_transpose_matrix_embedding(train_adata, pert_key, n_nmf_embedding)

    del train_adata; gc.collect()

    coex_path = os.path.join(emb_dir, f"{label}precomputed_coexpression_emb.pkl")
    with open(coex_path, "wb") as f:
        pkl.dump(coex_out, f, protocol=pkl.HIGHEST_PROTOCOL)
    transpose_path = os.path.join(emb_dir, f"{label}precomputed_transpose_matrix_emb.pkl")
    with open(transpose_path, "wb") as f:
        pkl.dump(transpose_out, f, protocol=pkl.HIGHEST_PROTOCOL)
    del coex_out, transpose_out; gc.collect()
    return coex_path, transpose_path


# =========================================================================
# Split generation
# =========================================================================

def parse_mode(mode_str):
    if mode_str is None or mode_str.lower() == "none":
        return ("direct", 1)
    match = re.match(r"target_fold_(\d+)", mode_str)
    if match:
        return ("target_fold", int(match.group(1)))
    raise ValueError(f"Unrecognized mode '{mode_str}'.")


def create_splits_direct(obs, split_col, pert_group_key, splits_dir):
    train_perts = obs.loc[obs[split_col] == "train", pert_group_key].unique().tolist()
    test_perts = obs.loc[obs[split_col].str.startswith("test_"), pert_group_key].unique().tolist()
    split = {"train": train_perts, "val": train_perts, "test": test_perts}
    sf = os.path.join(splits_dir, "seed_0.json")
    with open(sf, "w") as f:
        json.dump(split, f)
    print(f"  Direct: train={len(train_perts)}, test={len(test_perts)}")
    return [{"file": sf, "fold": 0, "train_perts": train_perts}]


def create_splits_target_fold(obs, split_col, target_name, target_key, pert_group_key, n_folds, splits_dir):
    target_train = obs.loc[
        (obs[split_col] == "train") & (obs[target_key] == target_name), pert_group_key
    ].unique().tolist()
    remaining = list(set(obs[pert_group_key].unique()) - set(target_train))
    print(f"  Target '{target_name}': {len(target_train)} perts, other: {len(remaining)}")

    folds = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (ti, vi) in enumerate(kf.split(target_train)):
        ft = [target_train[i] for i in ti] + remaining
        fv = [target_train[i] for i in vi]
        split = {"train": ft, "val": fv}
        sf = os.path.join(splits_dir, f"seed_{fold}.json")
        with open(sf, "w") as f:
            json.dump(split, f)
        print(f"  Fold {fold}: train={len(ft)}, val={len(fv)}")
        folds.append({"file": sf, "fold": fold, "train_perts": ft})
    return folds


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRESAGE data prep with per-cell-type file support"
    )
    # Input: either single file or cell-type directory
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--scdata_file", type=str, default=None,
                           help="Single merged h5ad (backed mode)")
    input_grp.add_argument("--celltype_dir", type=str, default=None,
                           help="Directory with per-cell-type h5ad files (from split_by_celltype.py)")

    parser.add_argument("--split_col", type=str, default=None)
    parser.add_argument("--target_name", type=str, default=None)
    parser.add_argument("--target_key", type=str, default="cell_type")
    parser.add_argument("--pert_key", type=str, default="perturbation")
    parser.add_argument("--pert_group_key", type=str, default="perturbation_group")
    parser.add_argument("--control_key", type=str, default="non-targeting")
    parser.add_argument("--prior_type", type=str, default="dp", choices=["dp", "kp", "all"])
    parser.add_argument("--prior_by_group", type=str, default=None)
    parser.add_argument("--n_nmf_embedding", type=int, default=512)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--stage", type=str, default=None, choices=["hpo", "final"])
    parser.add_argument("--variant", type=str, default=None, choices=["kpdp", "only", "pplus"])
    parser.add_argument("--output_dir", type=str, required=True)

    # DEG options
    parser.add_argument("--degs_only", action="store_true")
    parser.add_argument("--precomputed_degs", type=str, default=None)
    parser.add_argument("--skip_degs", action="store_true",
                        help="Write stub DEGs with empty lists (instant)")

    args = parser.parse_args()

    if not args.degs_only:
        for req in ["split_col", "target_name", "stage", "variant"]:
            if getattr(args, req) is None:
                parser.error(f"--{req} required unless --degs_only")

    use_celltype_files = args.celltype_dir is not None

    dataset_dir = os.path.join(args.output_dir, "dataset")
    deg_dir = os.path.join(dataset_dir, "degs")
    splits_dir = os.path.join(args.output_dir, "splits")
    emb_dir = os.path.join(args.output_dir, "embeddings")
    for d in [dataset_dir, deg_dir, splits_dir, emb_dir]:
        os.makedirs(d, exist_ok=True)

    # ==================================================================
    # Load manifest / determine relevant files
    # ==================================================================
    if use_celltype_files:
        manifest = load_manifest(args.celltype_dir)
        print(f"Cell-type files: {list(manifest.keys())}")
    else:
        manifest = None

    # ==================================================================
    # DEGs-only mode
    # ==================================================================
    if args.degs_only:
        print(f"\n{'='*60}\nDEGs-only mode\n{'='*60}")
        merged_deg_file = os.path.join(deg_dir, "merged.degs.json")
        if use_celltype_files:
            degs = {}
            for ct, path in manifest.items():
                print(f"\n  {ct}:")
                adata = _read_full(path)
                degs.update(compute_degs_for_celltype(
                    adata, deg_dir, ct, args.target_key, args.pert_key,
                    args.pert_group_key, args.control_key
                ))
                del adata; gc.collect()
            with open(merged_deg_file, "w") as f:
                json.dump(degs, f)
        else:
            adata_b = _load_backed(args.scdata_file)
            obs = adata_b.obs
            cell_types = obs[args.target_key].unique().tolist()
            degs = {}
            for ct in cell_types:
                mask = (obs[args.target_key] == ct).values
                ct_adata = _slice_to_memory(adata_b, mask)
                degs.update(compute_degs_for_celltype(
                    ct_adata, deg_dir, ct, args.target_key, args.pert_key,
                    args.pert_group_key, args.control_key
                ))
                del ct_adata; gc.collect()
            adata_b.file.close()
            with open(merged_deg_file, "w") as f:
                json.dump(degs, f)
        print(f"\nDone: {len(degs)} DEGs -> {merged_deg_file}")
        return

    mode_type, n_folds = parse_mode(args.mode)
    print(f"\nStage: {args.stage} | Variant: {args.variant} | Mode: {mode_type}")
    print(f"Input: {'cell-type files' if use_celltype_files else 'single h5ad'}")

    # ==================================================================
    # Step 1: Get filtered obs + determine relevant cell type files
    # ==================================================================
    print(f"\n{'='*60}\nStep 1: Filter obs\n{'='*60}")

    if use_celltype_files:
        obs_filtered, relevant_cts = get_filtered_obs(
            manifest, args.variant, args.target_name, args.target_key,
            args.stage, args.split_col
        )
    else:
        adata_b = _load_backed(args.scdata_file)
        obs_all = adata_b.obs.copy()
        adata_b.file.close(); del adata_b; gc.collect()

        keep_mask = np.ones(len(obs_all), dtype=bool)
        if args.variant in ("only", "pplus"):
            keep_mask &= (obs_all[args.target_key] == args.target_name).values
        if args.stage == "hpo":
            keep_mask &= obs_all[args.split_col].isin(["train", "control"]).values
        else:
            keep_mask &= (~obs_all[args.split_col].isin(["excluded"])).values

        obs_filtered = obs_all[keep_mask]
        relevant_cts = None
        del obs_all; gc.collect()

    # Clean obs columns
    cols_keep = [c for c in obs_filtered.columns if ("_stratified" not in c) and ("_random" not in c)]
    if args.split_col not in cols_keep:
        cols_keep.append(args.split_col)
    obs_filtered = obs_filtered[cols_keep]

    print(f"  Filtered: {len(obs_filtered):,} cells, {obs_filtered[args.pert_group_key].nunique():,} perts")

    # ==================================================================
    # Step 2: DEGs
    # ==================================================================
    print(f"\n{'='*60}\nStep 2: DEGs\n{'='*60}")
    merged_deg_file = os.path.join(deg_dir, "merged.degs.json")

    if args.skip_degs:
        if not os.path.exists(merged_deg_file):
            all_pgs = obs_filtered[args.pert_group_key].unique().tolist()
            with open(merged_deg_file, "w") as f:
                json.dump({pg: [] for pg in all_pgs}, f)
            print(f"  Stub DEGs ({len(all_pgs)} keys) -> {merged_deg_file}")
        else:
            print(f"  Found existing at {merged_deg_file}")
    elif args.precomputed_degs:
        if not os.path.exists(merged_deg_file):
            src_dir = os.path.dirname(args.precomputed_degs)
            for fn in os.listdir(src_dir):
                src, dst = os.path.join(src_dir, fn), os.path.join(deg_dir, fn)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            print(f"  Copied from {src_dir}")
        else:
            print(f"  Found existing at {merged_deg_file}")
    else:
        if use_celltype_files:
            degs = {}
            for ct, path in relevant_cts.items():
                adata = _read_full(path)
                if args.stage == "hpo":
                    adata = adata[adata.obs[args.split_col].isin(["train", "control"]), :]
                else:
                    adata = adata[~adata.obs[args.split_col].isin(["excluded"]), :]
                degs.update(compute_degs_for_celltype(
                    adata, deg_dir, ct, args.target_key, args.pert_key,
                    args.pert_group_key, args.control_key
                ))
                del adata; gc.collect()
            with open(merged_deg_file, "w") as f:
                json.dump(degs, f)
        else:
            # backed fallback
            adata_b = _load_backed(args.scdata_file)
            obs = adata_b.obs
            degs = {}
            for ct in obs_filtered[args.target_key].unique():
                ct_mask = (obs[args.target_key] == ct).values & keep_mask
                ct_adata = _slice_to_memory(adata_b, ct_mask)
                degs.update(compute_degs_for_celltype(
                    ct_adata, deg_dir, ct, args.target_key, args.pert_key,
                    args.pert_group_key, args.control_key
                ))
                del ct_adata; gc.collect()
            adata_b.file.close()
            with open(merged_deg_file, "w") as f:
                json.dump(degs, f)
        print(f"  Computed {len(degs)} DEGs")

    # ==================================================================
    # Step 3: Pseudobulk
    # ==================================================================
    print(f"\n{'='*60}\nStep 3: Pseudobulking\n{'='*60}")
    pbulk_h5ad = dataset_dir + ".h5ad"
    pbulk_prep = dataset_dir + "_preprocessed.h5ad"
    ncells_path = os.path.join(dataset_dir, "ncells_per_perturbation.json")

    if not os.path.exists(pbulk_h5ad):
        if use_celltype_files:
            pbulk_adata, ncells = pseudobulk_from_celltype_files(
                manifest, relevant_cts, obs_filtered, args.pert_group_key,
                args.target_key, args.pert_key, args.control_key,
                args.stage, args.split_col
            )
        else:
            # backed fallback
            pbulk_adata, ncells = compute_pseudobulk_from_backed(
                args.scdata_file, keep_mask, args.pert_group_key,
                args.target_key, args.pert_key, args.control_key
            )
        pbulk_adata.write_h5ad(pbulk_h5ad)
        pbulk_adata.write_h5ad(pbulk_prep)
        with open(ncells_path, "w") as f:
            json.dump(ncells, f)
        pbulk_shape = list(pbulk_adata.shape)
        del pbulk_adata; gc.collect()
    else:
        print(f"  Found existing at {pbulk_h5ad}")
        tmp = sc.read_h5ad(pbulk_h5ad)
        pbulk_shape = list(tmp.shape)
        del tmp

    # ==================================================================
    # Step 4: Splits
    # ==================================================================
    print(f"\n{'='*60}\nStep 4: Splits\n{'='*60}")
    if mode_type == "direct":
        fold_infos = create_splits_direct(obs_filtered, args.split_col, args.pert_group_key, splits_dir)
    else:
        fold_infos = create_splits_target_fold(
            obs_filtered, args.split_col, args.target_name, args.target_key,
            args.pert_group_key, n_folds, splits_dir
        )

    # ==================================================================
    # Step 5: Per-fold embeddings
    # ==================================================================
    print(f"\n{'='*60}\nStep 5: Per-fold embeddings\n{'='*60}")

    if use_celltype_files:
        # Load each cell type file ONCE, compute all folds from it
        emb_paths = compute_all_fold_embeddings_from_celltype_files(
            relevant_cts, fold_infos, args.prior_type, args.prior_by_group,
            args.pert_key, args.pert_group_key, args.control_key,
            args.n_nmf_embedding, args.stage, args.split_col, emb_dir
        )
    else:
        # Single file: load once, compute all folds
        emb_paths = compute_all_fold_embeddings_backed(
            args.scdata_file, keep_mask, fold_infos,
            args.prior_type, args.prior_by_group, args.pert_key,
            args.pert_group_key, args.control_key, args.n_nmf_embedding,
            emb_dir
        )

    # ==================================================================
    # Metadata
    # ==================================================================
    print(f"\n{'='*60}\nMetadata\n{'='*60}")
    metadata = {
        "scdata_file": args.scdata_file or args.celltype_dir,
        "split_col": args.split_col, "target_name": args.target_name,
        "target_key": args.target_key, "pert_key": args.pert_key,
        "pert_group_key": args.pert_group_key, "control_key": args.control_key,
        "prior_type": args.prior_type, "prior_by_group": args.prior_by_group,
        "n_nmf_embedding": args.n_nmf_embedding, "mode": args.mode,
        "mode_type": mode_type, "stage": args.stage, "variant": args.variant,
        "n_folds": n_folds, "num_splits": len(fold_infos),
        "pseudobulk_shape": pbulk_shape,
        "paths": {
            "dataset_dir": dataset_dir,
            "pseudobulk_h5ad": pbulk_h5ad, "pseudobulk_preprocessed_h5ad": pbulk_prep,
            "ncells_per_perturbation": ncells_path,
            "deg_dir": deg_dir, "merged_degs": merged_deg_file,
            "splits_dir": splits_dir, "embeddings_dir": emb_dir,
            "coexpression_emb": emb_paths[0]["coexpression_emb"] if len(emb_paths) == 1 else None,
            "transpose_matrix_emb": emb_paths[0]["transpose_matrix_emb"] if len(emb_paths) == 1 else None,
            "per_fold_embeddings": emb_paths if len(emb_paths) > 1 else None,
        },
    }
    with open(os.path.join(args.output_dir, "prepare_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}\nDONE\n{'='*60}")
    print(f"  Pseudobulk: {pbulk_shape[0]} x {pbulk_shape[1]}")
    print(f"  Splits: {len(fold_infos)}, Embeddings: {len(emb_paths)} set(s)")


if __name__ == "__main__":
    main()
