#Multi-context training

import os
import sys
import gc
import warnings
import traceback

import numpy as np
import pandas as pd
import anndata as ad
import torch
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=Warning)

#config
SCLAMBDA_REPO_PATH = "/home/scLAMBDA"
ADATA_PATH = "/home/merged_lognorm_single_cell_tens_preprocessed.h5ad"
GENE_EMB_PATH = "/home/GPT_3_5_gene_embeddings_3-large.pickle"

CONT_EMB_PATH = "/home/GPT_3_5_cell_line_embeddings_3-large.npy"

MODEL_ROOT = "/home/multi_context/k562_multicontext_models"
H5AD_OUTPUT_DIR = "/home/multi_context/k562_multicontext_predictions"
RESULTS_CSV_PATH = "/home/multi_context/k562_multicontext_split_results.csv"

TARGET_CONTEXT = "k562"
OTHER_CONTEXTS = ["jurkat", "hepg2", "rpe1"]

CTRL_LABEL = "non-targeting"
SEED = 0
VAL_FRACTION = 0.2
BATCH_SIZE = 2000
EXPERIMENT_SUFFIXES = None

#setup
sys.path.insert(0, SCLAMBDA_REPO_PATH)
import sclambda  
print("sclambda.model loaded from:", sclambda.model.__file__)

os.makedirs(MODEL_ROOT, exist_ok=True)
os.makedirs(H5AD_OUTPUT_DIR, exist_ok=True)

ALL_CONTEXTS = [TARGET_CONTEXT] + OTHER_CONTEXTS


def load_cont_emb(path, expected_contexts):
    raw = np.load(path, allow_pickle=True)

    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()

    if isinstance(raw, dict):
        cont_emb = {str(k).lower(): np.asarray(v, dtype=np.float32).reshape(-1) for k, v in raw.items()}
    elif isinstance(raw, np.ndarray) and raw.ndim == 2:
        print(
            " WARNING: cont_emb file loaded as a plain "
            f"{raw.shape} array with no keys. Assuming row order == "
            f"{expected_contexts}"
        )
        cont_emb = {
            ctx: raw[i].astype(np.float32)
            for i, ctx in enumerate(expected_contexts)
        }
    else:
        raise ValueError(
            f"Unrecognized structure for cont_emb file at {path}: "
            f"type={type(raw)}. Expected a dict or a 2D array."
        )

    print(f"Loaded cont_emb with keys: {sorted(cont_emb.keys())}")
    missing = [ctx for ctx in expected_contexts if ctx not in cont_emb]
    if missing:
        raise ValueError(
            f"cont_emb is missing embeddings for contexts {missing}. "
            f"Keys found : {sorted(cont_emb.keys())}."
        )
    dims = {ctx: cont_emb[ctx].shape[0] for ctx in expected_contexts}
    if len(set(dims.values())) > 1:
        raise ValueError(f"cont_emb vectors have inconsistent dims: {dims}")

    return cont_emb


def load_data():
    print("Loading full adata (all contexts)...")
    adata = ad.read_h5ad(ADATA_PATH)

    m_cols = [
        "target_gene", "cell_type", "source_file",
        "context", "perturbation_group", "perturbation",
    ]
    all_split_cols = [c for c in adata.obs.columns
                       if any(c.startswith(f"{ctx}_") for ctx in ALL_CONTEXTS)]
    adata_ = adata.copy()
    adata_.obs = adata_.obs[m_cols + all_split_cols].copy()

    suffixes_per_context = {}
    for ctx in ALL_CONTEXTS:
        prefix = f"{ctx}_"
        suffixes_per_context[ctx] = {
            c[len(prefix):] for c in all_split_cols if c.startswith(prefix)
        }

    common_suffixes = set.intersection(*suffixes_per_context.values())
    print(f"Experiment suffixes common to all {len(ALL_CONTEXTS)} contexts: {len(common_suffixes)}")
    for s in sorted(common_suffixes):
        print("  -", s)

    gene_embeddings = pd.read_pickle(GENE_EMB_PATH)
    cont_emb = load_cont_emb(CONT_EMB_PATH, ALL_CONTEXTS)

    pert_g = [g for g in adata_.obs["perturbation"].unique().tolist() if g != CTRL_LABEL]
    gene_embeds, missing = {}, []
    for g in pert_g:
        if g in gene_embeddings:
            gene_embeds[g] = gene_embeddings[g]
        else:
            missing.append(g)

    print(f"Gene embeddings matched: {len(gene_embeds)}/{len(pert_g)} | missing: {len(missing)}")
    adata_new = adata_[~adata_.obs["perturbation"].isin(missing)].copy()

    return adata_new, gene_embeds, cont_emb, sorted(common_suffixes)


def to_pseudobulk(cells_):
    return cells_.mean(axis=0)


def get_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


@torch.no_grad()
def predict_pseudobulk_batched(model, pert_test, ctrl_cells_absolute, target_context, batch_size=BATCH_SIZE):
    
    model.Net.eval()
    device = model.device

    ctrl_mean_ct = np.asarray(model.ctrl_mean[target_context]).reshape(1, -1)
    cont_emb_ct = np.asarray(model.cont_emb[target_context]).reshape(1, -1)

    ctrl_cells_delta = (ctrl_cells_absolute - ctrl_mean_ct).astype(np.float32)
    ctrl_delta_t = torch.from_numpy(ctrl_cells_delta).float()
    n_ctrl = ctrl_delta_t.shape[0]

    results_mean = {}

    for i in pert_test:
        if model.multi_gene:
            genes = i.split('+')
            pert_emb_p = model.gene_emb[genes[0]] + model.gene_emb[genes[1]]
        else:
            pert_emb_p = model.gene_emb[i]

        d_sum = None
        n_seen = 0

        for start in range(0, n_ctrl, batch_size):
            end = min(start + batch_size, n_ctrl)
            batch = ctrl_delta_t[start:end].to(device)
            bs = batch.shape[0]

            val_p = torch.from_numpy(np.tile(pert_emb_p, (bs, 1))).float().to(device)
            val_ct = torch.from_numpy(np.tile(cont_emb_ct, (bs, 1))).float().to(device)

            x_hat, p_hat, ct_hat, mean_z, log_var_z, s, c = model.Net(batch, val_p, val_ct)
            x_hat_abs = x_hat.detach().cpu().numpy() + ctrl_mean_ct  

            batch_sum = x_hat_abs.sum(axis=0)  
            d_sum = batch_sum if d_sum is None else d_sum + batch_sum
            n_seen += bs

            del batch, val_p, val_ct, x_hat, x_hat_abs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results_mean[i] = d_sum / n_seen  

    return results_mean


def run_experiment(adata_new, gene_embeds, cont_emb, suffix):
    print("\n" + "=" * 80)
    print(f"Running multi-context experiment: {suffix}  "
          f"(target: {TARGET_CONTEXT}, others: {OTHER_CONTEXTS})")
    print("=" * 80)

    #skip if this experiment's output h5ad already exists
    h5ad_path = os.path.join(H5AD_OUTPUT_DIR, f"{suffix}.h5ad")
    if os.path.exists(h5ad_path):
        print(f"  SKIPPING: output already exists at {h5ad_path}")
        return {"experiment": suffix, "status": "skipped_already_exists", "h5ad_path": h5ad_path}

    adata_all = adata_new[adata_new.obs["perturbation"].isin(
        list(gene_embeds.keys()) + [CTRL_LABEL]
    )].copy()
    adata_all.obs["condition"] = adata_all.obs["perturbation"].astype(str)
    adata_all.obs.loc[adata_all.obs["perturbation"] == CTRL_LABEL, "condition"] = "ctrl"

    adata_all.obs["cell_type"] = adata_all.obs["context"].astype(str)
    adata_all.obs["cell_type+condition"] = (
        adata_all.obs["cell_type"].astype(str) + "_---_" + adata_all.obs["condition"].astype(str)
    )

    role = pd.Series(index=adata_all.obs.index, dtype=object)
    for ctx in ALL_CONTEXTS:
        col = f"{ctx}_{suffix}"
        if col not in adata_all.obs.columns:
            print(f"  WARNING: column {col} missing for context {ctx} - skipping this experiment")
            return {"experiment": suffix, "status": "skipped_missing_column"}
        mask = adata_all.obs["context"].values == ctx
        role.loc[mask] = adata_all.obs.loc[mask, col].astype(str).values
    adata_all.obs["role"] = role

    is_target = adata_all.obs["context"].values == TARGET_CONTEXT

    keep_mask = (
        (adata_all.obs["role"] == "train").values
        | (adata_all.obs["role"] == "control").values
        | (is_target & adata_all.obs["role"].isin(["test_seen", "test_unseen"]).values)
    )
    adata_proc = adata_all[keep_mask].copy()
    del adata_all
    gc.collect()

    train_mask = adata_proc.obs["role"].values == "train"
    test_mask = (adata_proc.obs["context"].values == TARGET_CONTEXT) & \
                adata_proc.obs["role"].isin(["test_seen", "test_unseen"]).values
    target_mask = adata_proc.obs["context"].values == TARGET_CONTEXT

    non_target_test_leak = ((~target_mask) & adata_proc.obs["role"].isin(["test_seen", "test_unseen"]).values).sum()
    assert non_target_test_leak == 0, "non-target test cells leaked into adata_proc"

    all_train_perts = set(adata_proc.obs.loc[train_mask, "condition"].unique()) - {"ctrl"}
    target_train_perts = set(
        adata_proc.obs.loc[train_mask & target_mask, "condition"].unique()
    ) - {"ctrl"}
    test_perts = set(adata_proc.obs.loc[test_mask, "condition"].unique()) - {"ctrl"}

    if len(target_train_perts) < 2 or len(test_perts) < 1:
        print(f"  WARNING: not enough K562 perturbations ({len(target_train_perts)} train, "
              f"{len(test_perts)} test) — skipping")
        return {"experiment": suffix, "status": "skipped_too_few_perts"}

    rng = np.random.default_rng(SEED)
    target_train_arr = np.array(sorted(target_train_perts))
    rng.shuffle(target_train_arr)

    n_val = max(1, int(round(VAL_FRACTION * len(target_train_arr))))
    val_perts = set(target_train_arr[:n_val])

    adata_proc.obs["split"] = "train"
    adata_proc.obs.loc[
        adata_proc.obs["condition"].isin(val_perts) & train_mask & target_mask,
        "split",
    ] = "val"
    adata_proc.obs.loc[test_mask, "split"] = "test"

    assert set(adata_proc.obs.loc[adata_proc.obs["split"] == "val", "context"].unique()) <= {TARGET_CONTEXT}
    assert set(adata_proc.obs.loc[adata_proc.obs["split"] == "test", "context"].unique()) <= {TARGET_CONTEXT}

    print(f"  target ({TARGET_CONTEXT}) train perts: {len(target_train_perts)} | "
          f"Val perts (20% of target train, target-context only): {len(val_perts)} | "
          f"Total train perts across all contexts: {len(all_train_perts)} | "
          f"Test perts (target-context only): {len(test_perts)}")
    print(adata_proc.obs["split"].value_counts().to_string())
    print(adata_proc.obs.groupby(["context", "split"]).size().to_string())

    if sp.issparse(adata_proc.X):
        adata_proc.X = adata_proc.X.toarray()
    adata_proc.X = adata_proc.X.astype(np.float32)

    #train model
    model_path = os.path.join(MODEL_ROOT, suffix)
    os.makedirs(model_path, exist_ok=True)


    model = sclambda.model.Model_context(
        adata_proc, gene_embeds, cont_emb,
        model_path=model_path,
        multi_gene=False,
        split_name="split",
    )
    model.train()

    print(f"ctrl_mean computed for contexts: {sorted(model.ctrl_mean.keys())}")

    k562_ctrl_mask = (adata_proc.obs["condition"] == "ctrl") & (adata_proc.obs["context"] == TARGET_CONTEXT)
    k562_ctrl_cells_full = get_dense(adata_proc[k562_ctrl_mask].X).astype(np.float32)
    print(f"  hepg2 control cells available and used (no subsampling): {k562_ctrl_cells_full.shape[0]}")

    ctrl_mean_k562 = to_pseudobulk(k562_ctrl_cells_full)

    #evaluate on K562 test set
    pert_test = sorted(test_perts)
    pred_pseudobulk = predict_pseudobulk_batched(
        model, pert_test, k562_ctrl_cells_full, target_context=TARGET_CONTEXT
    )

    results = {}
    pred_delta_mat, true_delta_mat = [], []
    pred_pseudobulk_mat, true_pseudobulk_mat = [], []

    for i in pert_test:
        pred_abs = pred_pseudobulk[i]                 
        pred_delta = pred_abs - ctrl_mean_k562         
        true_mask = (adata_proc.obs["condition"].values == i) & (adata_proc.obs["context"].values == TARGET_CONTEXT)
        true_cells = get_dense(adata_proc[true_mask].X)
        true_abs = to_pseudobulk(true_cells)           
        true_delta = true_abs - ctrl_mean_k562         

        results[i] = np.corrcoef(pred_delta, true_delta)[0, 1]

        pred_delta_mat.append(pred_delta)
        true_delta_mat.append(true_delta)
        pred_pseudobulk_mat.append(pred_abs)
        true_pseudobulk_mat.append(true_abs)

    mean_pcc_test = float(np.nanmean(list(results.values())))

    target_split_col = f"{TARGET_CONTEXT}_{suffix}"
    seen_perts = set(
        adata_proc.obs.loc[adata_proc.obs[target_split_col].astype(str) == "test_seen", "condition"].unique()
    ) - {"ctrl"}
    unseen_perts = set(
        adata_proc.obs.loc[adata_proc.obs[target_split_col].astype(str) == "test_unseen", "condition"].unique()
    ) - {"ctrl"}

    pcc_seen = [results[g] for g in pert_test if g in seen_perts]
    pcc_unseen = [results[g] for g in pert_test if g in unseen_perts]
    mean_pcc_seen = float(np.nanmean(pcc_seen)) if pcc_seen else np.nan
    mean_pcc_unseen = float(np.nanmean(pcc_unseen)) if pcc_unseen else np.nan

    #save predictions
    pred_delta_mat = np.array(pred_delta_mat)
    true_delta_mat = np.array(true_delta_mat)
    pred_pseudobulk_mat = np.array(pred_pseudobulk_mat)
    true_pseudobulk_mat = np.array(true_pseudobulk_mat)

    adata_pred = ad.AnnData(
        X=pred_delta_mat,  
        obs=pd.DataFrame({
            "perturbation": pert_test,
            "pcc": [results[i] for i in pert_test],
            "test_category": [
                "test_seen" if g in seen_perts else ("test_unseen" if g in unseen_perts else "unknown")
                for g in pert_test
            ],
        }, index=pert_test),
        var=adata_proc.var.copy(),
    )
    adata_pred.layers["predicted_delta_"] = pred_delta_mat

    adata_pred.write_h5ad(h5ad_path)

    print(f"  Mean PCC (delta, k562 eval) - "
          f"Test_seen: {mean_pcc_seen:.4f} | test_unseen: {mean_pcc_unseen:.4f} | "
          f"Test_overall: {mean_pcc_test:.4f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved predictions to: {h5ad_path}")

    result = {
        "experiment": suffix,
        "target_context": TARGET_CONTEXT,
        "other_contexts": ",".join(OTHER_CONTEXTS),
        "status": "ok",
        "n_target_train_perts": len(target_train_perts),
        "n_val_perts": len(val_perts),
        "n_total_train_perts_all_contexts": len(all_train_perts),
        "n_test_perts": len(test_perts),
        "n_test_seen": len(pcc_seen),
        "n_test_unseen": len(pcc_unseen),
        "n_k562_ctrl_cells_used": int(k562_ctrl_cells_full.shape[0]),
        "mean_pcc_test_seen": mean_pcc_seen,
        "mean_pcc_test_unseen": mean_pcc_unseen,
        "mean_pcc_test_overall": mean_pcc_test,
        "model_path": model_path,
        "h5ad_path": h5ad_path,
    }

    del model, adata_proc, k562_ctrl_cells_full, pred_pseudobulk
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    adata_new, gene_embeds, cont_emb, common_suffixes = load_data()

    suffixes_to_run = EXPERIMENT_SUFFIXES if EXPERIMENT_SUFFIXES is not None else common_suffixes

    all_results = []
    for suffix in suffixes_to_run:
        try:
            r = run_experiment(adata_new, gene_embeds, cont_emb, suffix)
        except Exception as e:
            print(f"ERROR on experiment {suffix}: {e}")
            traceback.print_exc()
            r = {"experiment": suffix, "status": f"error: {e}"}
        all_results.append(r)
        pd.DataFrame(all_results).to_csv(RESULTS_CSV_PATH, index=False)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)

    print("\nAll experiments completed. Summary:")
    print(results_df)
    print(f"\nFinal results CSV saved to: {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
