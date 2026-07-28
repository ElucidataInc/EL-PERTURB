#Single-context K562 training - Knowledge Priors (KP) only

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

MODEL_ROOT = "/home/scLAMBDA/single_context/k562_single_context_models"
H5AD_OUTPUT_DIR = "/home/scLAMBDA/single_context/k562_single_context_predictions"
RESULTS_CSV_PATH = "/home/scLAMBDA/single_context/k562_single_context_results.csv"

TARGET_CONTEXT = "k562"
CTRL_LABEL = "non-targeting"
SEED= 0
VAL_FRACTION = 0.2
BATCH_SIZE= 2000

SPLIT_COLS = None

#setup
sys.path.insert(0, SCLAMBDA_REPO_PATH)
import sclambda 
print("sclambda.model loaded from:", sclambda.model.__file__)

os.makedirs(MODEL_ROOT, exist_ok=True)
os.makedirs(H5AD_OUTPUT_DIR, exist_ok=True)


def get_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def to_pseudobulk(cells_log1p):
    return cells_log1p.mean(axis=0)


@torch.no_grad()
def predict_pseudobulk_batched(model, pert_test, ctrl_cells_absolute, batch_size=BATCH_SIZE):
    model.Net.eval()
    device = model.device
    ctrl_mean = np.asarray(model.ctrl_mean).reshape(1, -1)

    ctrl_delta = (ctrl_cells_absolute - ctrl_mean).astype(np.float32)
    ctrl_delta_t = torch.from_numpy(ctrl_delta).float()
    n_ctrl = ctrl_delta_t.shape[0]

    results = {}
    for i in pert_test:
        pert_emb_p = model.gene_emb[i]  

        d_sum = None
        n_seen = 0
        for start in range(0, n_ctrl, batch_size):
            end = min(start + batch_size, n_ctrl)
            batch = ctrl_delta_t[start:end].to(device)
            val_p = torch.from_numpy(np.tile(pert_emb_p, (batch.shape[0], 1))).float().to(device)

            x_hat, p_hat, mean_z, log_var_z, s = model.Net(batch, val_p)
            x_hat_abs = x_hat.detach().cpu().numpy() + ctrl_mean

            batch_sum = x_hat_abs.sum(axis=0)
            d_sum = batch_sum if d_sum is None else d_sum + batch_sum
            n_seen += batch.shape[0]

            del batch, val_p, x_hat, x_hat_abs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results[i] = d_sum / n_seen

    return results


def load_data():
    print(f"Loading adata, filtering to context == '{TARGET_CONTEXT}'...")
    adata = ad.read_h5ad(ADATA_PATH)
    adata_k562 = adata[adata.obs["context"] == TARGET_CONTEXT]

    m_cols = ["target_gene", "cell_type", "source_file", "context", "perturbation_group", "perturbation"]
    k562_cols = [c for c in adata_k562.obs.columns if c.startswith(f"{TARGET_CONTEXT}_")]
    adata_k562_ = adata_k562.copy()
    adata_k562_.obs = adata_k562_.obs[m_cols + k562_cols].copy()
    del adata
    gc.collect()

    print(f"Found {len(k562_cols)} K562 split columns:")
    for c in k562_cols:
        print("  -", c)

    gene_embeddings = pd.read_pickle(GENE_EMB_PATH)

    pert_g = [g for g in adata_k562_.obs["perturbation"].unique().tolist() if g != CTRL_LABEL]
    gene_embeds, missing = {}, []
    for g in pert_g:
        if g in gene_embeddings:
            gene_embeds[g] = gene_embeddings[g].astype(np.float32)
        else:
            missing.append(g)

    print(f"Gene embeddings matched: {len(gene_embeds)}/{len(pert_g)} | missing: {len(missing)}")
    adata_k562_new = adata_k562_[~adata_k562_.obs["perturbation"].isin(missing)].copy()

    return adata_k562_new, gene_embeds, k562_cols


def run_split(adata_k562_new, gene_embeds, split_col):
    print("\n" + "=" * 80)
    print(f"Running K562 split: {split_col}")
    print("=" * 80)

    adata_proc = adata_k562_new[adata_k562_new.obs["perturbation"].isin(
        list(gene_embeds.keys()) + [CTRL_LABEL]
    )].copy()
    adata_proc.obs["condition"] = adata_proc.obs["perturbation"].astype(str)
    adata_proc.obs.loc[adata_proc.obs["perturbation"] == CTRL_LABEL, "condition"] = "ctrl"

    cat_col = adata_proc.obs[split_col].astype(str)
    train_mask = cat_col == "train"
    test_mask = cat_col.isin(["test_seen", "test_unseen"])
    ctrl_mask = cat_col == "control"

    if not (train_mask | test_mask | ctrl_mask).all():
        print("WARNING: unexpected category values - skipping")
        return {"split_col": split_col, "status": "skipped_unexpected_category"}

    train_perts = set(adata_proc.obs.loc[train_mask, "condition"].unique()) - {"ctrl"}
    test_perts = set(adata_proc.obs.loc[test_mask, "condition"].unique()) - {"ctrl"}

    if len(train_perts) < 2 or len(test_perts) < 1:
        print("WARNING: not enough perturbations - skipping")
        return {"split_col": split_col, "status": "skipped_too_few_perts"}

    #80/20 carve of this split's own K562 train perturbations - for validation set
    rng = np.random.default_rng(SEED)
    train_perts_arr = np.array(sorted(train_perts))
    rng.shuffle(train_perts_arr)
    n_val = max(1, int(round(VAL_FRACTION * len(train_perts_arr))))
    val_perts = set(train_perts_arr[:n_val])
    inner_train_perts = set(train_perts_arr[n_val:])

    adata_proc.obs["split"] = "train"
    adata_proc.obs.loc[adata_proc.obs["condition"].isin(val_perts) & train_mask, "split"] = "val"
    adata_proc.obs.loc[test_mask, "split"] = "test"

    print(f"  train perts (80%): {len(inner_train_perts)} | val perts (20%): {len(val_perts)} | "
          f"test perts: {len(test_perts)}")
    print(adata_proc.obs["split"].value_counts().to_string())

    if sp.issparse(adata_proc.X):
        adata_proc.X = adata_proc.X.toarray()
    adata_proc.X = adata_proc.X.astype(np.float32)

    #train model
    model_path = os.path.join(MODEL_ROOT, split_col)
    os.makedirs(model_path, exist_ok=True)

    model = sclambda.model.Model(
        adata_proc, gene_embeds,
        model_path=model_path,
        multi_gene=False,
    )
    model.train()

    ctrl_cells = get_dense(adata_proc[adata_proc.obs["condition"] == "ctrl"].X).astype(np.float32)
    ctrl_log1p_mean = to_pseudobulk(ctrl_cells)
    print(f"  K562 control cells used (no subsampling): {ctrl_cells.shape[0]}")

    #evaluate on this split's K562 test set
    pert_test = sorted(test_perts)
    pred_pseudobulk_log1p = predict_pseudobulk_batched(model, pert_test, ctrl_cells)

    results = {}
    pred_delta_mat, true_delta_mat = [], []
    pred_pseudobulk_mat, true_pseudobulk_mat = [], []

    for i in pert_test:
        pred_abs = pred_pseudobulk_log1p[i]
        pred_delta = pred_abs - ctrl_log1p_mean

        true_cells = get_dense(adata_proc[adata_proc.obs["condition"].values == i].X)
        true_abs = to_pseudobulk(true_cells)
        true_delta = true_abs - ctrl_log1p_mean

        results[i] = np.corrcoef(pred_delta, true_delta)[0, 1]
        pred_delta_mat.append(pred_delta)
        true_delta_mat.append(true_delta)
        pred_pseudobulk_mat.append(pred_abs)
        true_pseudobulk_mat.append(true_abs)

    mean_pcc_test = float(np.nanmean(list(results.values())))

    seen_perts = set(adata_proc.obs.loc[cat_col == "test_seen", "condition"].unique()) - {"ctrl"}
    unseen_perts = set(adata_proc.obs.loc[cat_col == "test_unseen", "condition"].unique()) - {"ctrl"}
    pcc_seen = [results[g] for g in pert_test if g in seen_perts]
    pcc_unseen = [results[g] for g in pert_test if g in unseen_perts]

    #save predictions AnnData
    pred_delta_mat = np.array(pred_delta_mat)
    true_delta_mat = np.array(true_delta_mat)
    pred_pseudobulk_mat = np.array(pred_pseudobulk_mat)
    true_pseudobulk_mat = np.array(true_pseudobulk_mat)

    adata_pred = ad.AnnData(
        X=pred_delta_mat,
        obs=pd.DataFrame({
            "perturbation": pert_test,
            "pcc": [results[i] for i in pert_test],
            "test_category": ["test_seen" if g in seen_perts else "test_unseen" for g in pert_test],
        }, index=pert_test),
        var=adata_proc.var.copy(),
    )
    adata_pred.layers["predicted_delta_"] = pred_delta_mat             

    h5ad_path = os.path.join(H5AD_OUTPUT_DIR, f"k562_single_context_predictions_{split_col}.h5ad")
    adata_pred.write_h5ad(h5ad_path)

    mean_seen = float(np.nanmean(pcc_seen)) if pcc_seen else np.nan
    mean_unseen = float(np.nanmean(pcc_unseen)) if pcc_unseen else np.nan
    print(f" PCC: test_seen: {mean_seen:.4f} | "
          f"test_unseen: {mean_unseen:.4f} | test_overall: {mean_pcc_test:.4f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved predictions to: {h5ad_path}")

    result = {
        "split_col": split_col, "status": "ok",
        "n_train_perts": len(inner_train_perts), "n_val_perts": len(val_perts),
        "n_test_perts": len(test_perts), "n_test_seen": len(pcc_seen), "n_test_unseen": len(pcc_unseen),
        "n_ctrl_cells_used": int(ctrl_cells.shape[0]),
        "mean_pcc_test_unseen": mean_unseen, "mean_pcc_test_overall": mean_pcc_test,
        "model_path": model_path, "h5ad_path": h5ad_path,
    }

    del model, adata_proc, ctrl_cells, pred_pseudobulk_log1p
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    adata_k562_new, gene_embeds, k562_cols = load_data()
    split_cols_to_run = SPLIT_COLS if SPLIT_COLS is not None else k562_cols

    all_results = []
    for split_col in split_cols_to_run:
        try:
            r = run_split(adata_k562_new, gene_embeds, split_col)
        except Exception as e:
            print(f"ERROR on {split_col}: {e}")
            traceback.print_exc()
            r = {"split_col": split_col, "status": f"error: {e}"}
        all_results.append(r)
        pd.DataFrame(all_results).to_csv(RESULTS_CSV_PATH, index=False)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll splits completed. Summary:")
    print(pd.DataFrame(all_results))
    print(f"\nFinal results CSV: {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
