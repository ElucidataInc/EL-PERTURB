#Single-context K562 training - Knowledge Priors (KP) only

#Usage:
#python scLAMBDA_single_context.py                              # all splits, one process
#CUDA_VISIBLE_DEVICES=0 python scLAMBDA_single_context.py --shard 0 --n-shards 2 &
#CUDA_VISIBLE_DEVICES=1 python scLAMBDA_single_context.py --shard 1 --n-shards 2 &

import os
import sys
import gc
import time
import argparse
import warnings
import traceback
from contextlib import contextmanager

import numpy as np
import pandas as pd
import anndata as ad
import torch

warnings.filterwarnings("ignore", category=Warning)

#config
SCLAMBDA_REPO_PATH = "/home/jl_fs/elperturb_sclamb/scLAMBDA"
ADATA_PATH = "/home/jl_fs/elperturb_sclamb/merged_GW_screens_lognorm_single_cell.h5ad"
GENE_EMB_PATH = "/home/jl_fs/elperturb_sclamb/PRESAGE_gene_embeddings.pickle"

MODEL_ROOT = "/home/jl_fs/elperturb_sclamb/scLAMBDA/single_context_new/k562GW_single_context_models"
H5AD_OUTPUT_DIR = "/home/jl_fs/elperturb_sclamb/scLAMBDA/single_context_new/k562GW_single_context_predictions"
RESULTS_CSV_PATH = "/home/jl_fs/elperturb_sclamb/scLAMBDA/single_context_new/k562GW_single_context_results.csv"

TARGET_CONTEXT = "K562"
CTRL_LABEL = "non-targeting"
SEED = 0
VAL_FRACTION = 0.2
SPLIT_COLS = None

#speed knobs
PRED_BATCH_SIZE = 8192        #control cells per forward pass at prediction time
TRAIN_BATCH_SIZE = 500        #scLAMBDA default. Bigger = fewer steps/epoch but changes optimisation
TRAIN_EPOCHS = 200            #scLAMBDA default for Model
TRAIN_DATA_PLACEMENT = "auto" #"auto" | "gpu" | "cpu"
ALLOW_TF32 = True             #~2-3x faster matmuls on Ampere+ GPUs, tiny numeric change
SKIP_EXISTING = True          #resume: skip splits whose prediction h5ad already exists

#setup
sys.path.insert(0, SCLAMBDA_REPO_PATH)
import sclambda
from sclambda.model import as_dense_f32, group_means
print("sclambda.model loaded from:", sclambda.model.__file__)

torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
torch.backends.cudnn.allow_tf32 = ALLOW_TF32

os.makedirs(MODEL_ROOT, exist_ok=True)
os.makedirs(H5AD_OUTPUT_DIR, exist_ok=True)


@contextmanager
def timed(label):
    t0 = time.time()
    yield
    print(f"  [time] {label}: {time.time() - t0:.1f}s")


def read_rows(path, keep_fn):
    """Load only the rows we need. Backed mode avoids reading all 2.3M cells into RAM."""
    try:
        backed = ad.read_h5ad(path, backed="r")
        keep = keep_fn(backed.obs)
        view = backed[keep]
        if hasattr(view, "to_memory"):                              # anndata >= 0.8
            sub = view.to_memory()
        else:                                                       # anndata 0.7.x
            sub = ad.AnnData(X=view.X[:] if hasattr(view.X, "shape") else view.X,
                             obs=view.obs.copy(), var=view.var.copy())
        backed.file.close()
    except Exception as e:
        print(f"Backed read failed ({e}); falling back to a full read")
        full = ad.read_h5ad(path)
        sub = full[keep_fn(full.obs)].copy()
        del full
    return sub


@torch.no_grad()
def predict_pseudobulk_batched(model, pert_test, ctrl_cells_absolute, batch_size=PRED_BATCH_SIZE):
    """Mean predicted expression per perturbation. Control cells go to the GPU once,
    the average is accumulated on the GPU, only n_genes floats come back per pert."""
    model.Net.eval()
    device = model.device
    ctrl_mean = torch.as_tensor(np.asarray(model.ctrl_mean).reshape(-1), dtype=torch.float32, device=device)
    ctrl_delta = torch.from_numpy(ctrl_cells_absolute).to(device) - ctrl_mean
    n_ctrl = ctrl_delta.shape[0]

    results = {}
    for i in pert_test:
        p1 = torch.as_tensor(np.asarray(model.gene_emb[i]), dtype=torch.float32, device=device).view(1, -1)
        d_sum = torch.zeros_like(ctrl_mean)
        for start in range(0, n_ctrl, batch_size):
            batch = ctrl_delta[start:start + batch_size]
            x_hat = model.Net(batch, p1.expand(batch.shape[0], -1))[0]
            d_sum += x_hat.sum(0)
        results[i] = (d_sum / n_ctrl + ctrl_mean).cpu().numpy()

    del ctrl_delta
    return results


def load_data():
    gene_embeddings = pd.read_pickle(GENE_EMB_PATH)
    emb_keys = set(gene_embeddings.keys())

    def keep_fn(obs):
        in_ctx = obs["context"].astype(str).values == TARGET_CONTEXT
        pert = obs["perturbation"].astype(str).values
        perts = set(pert[in_ctx]) - {CTRL_LABEL}
        missing = perts - emb_keys
        print(f"Gene embeddings matched: {len(perts) - len(missing)}/{len(perts)} | missing: {len(missing)}")
        return in_ctx & ~np.isin(pert, list(missing))

    print(f"Loading adata rows with context == '{TARGET_CONTEXT}'...")
    with timed("read h5ad"):
        adata = read_rows(ADATA_PATH, keep_fn)

    m_cols = ["cell_type", "source_file", "context", "perturbation_group", "perturbation"]
    k562_cols = [c for c in adata.obs.columns if c.startswith(f"{TARGET_CONTEXT}_")]
    adata.obs = adata.obs[m_cols + k562_cols].copy()
    print(f"Found {len(k562_cols)} K562 split columns:")
    for c in k562_cols:
        print("  -", c)

    pert = adata.obs["perturbation"].astype(str).values
    adata.obs["condition"] = np.where(pert == CTRL_LABEL, "ctrl", pert)
    gene_embeds = {g: np.asarray(gene_embeddings[g], dtype=np.float32)
                   for g in np.unique(pert) if g != CTRL_LABEL}
    del gene_embeddings

    #Densify once (re-done for every split), then compute everything that is
    #identical across splits: control cells + true pseudobulk of every perturbation.
    with timed("densify + shared stats"):
        adata.X = as_dense_f32(adata.X)
        cond = adata.obs["condition"].values
        ctrl_cells = adata.X[cond == "ctrl"]                      # copy, absolute values
        ctrl_log1p_mean = ctrl_cells.mean(axis=0)
        cat = pd.Categorical(cond)
        means, counts = group_means(adata.X, cat.codes.astype(np.int64), len(cat.categories))
        true_pb = {g: means[k] for k, g in enumerate(cat.categories) if counts[k] > 0 and g != "ctrl"}
    print(f"Cells: {adata.n_obs} | genes: {adata.n_vars} | control cells: {ctrl_cells.shape[0]}")

    shared = dict(adata=adata, ctrl_cells=ctrl_cells, ctrl_mean=ctrl_log1p_mean, true_pb=true_pb)
    return shared, gene_embeds, k562_cols


def run_split(shared, gene_embeds, split_col):
    print("\n" + "=" * 80)
    print(f"Running K562 split: {split_col}")
    print("=" * 80)

    h5ad_path = os.path.join(H5AD_OUTPUT_DIR, f"k562_single_context_predictions_{split_col}.h5ad")
    if SKIP_EXISTING and os.path.exists(h5ad_path):
        print(f"  SKIPPING: output already exists at {h5ad_path}")
        return {"split_col": split_col, "status": "skipped_already_exists", "h5ad_path": h5ad_path}

    adata = shared["adata"]
    obs = adata.obs
    cond = obs["condition"].values
    cat_col = obs[split_col].astype(str).values
    train_mask = cat_col == "train"
    test_mask = np.isin(cat_col, ["test_seen", "test_unseen"])
    ctrl_mask = cat_col == "control"

    if not (train_mask | test_mask | ctrl_mask).all():
        print("WARNING: unexpected category values - skipping")
        return {"split_col": split_col, "status": "skipped_unexpected_category"}

    train_perts = set(cond[train_mask]) - {"ctrl"}
    test_perts = set(cond[test_mask]) - {"ctrl"}
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

    split = np.full(adata.n_obs, "train", dtype=object)
    split[np.isin(cond, list(val_perts)) & train_mask] = "val"
    split[test_mask] = "test"

    print(f"  train perts (80%): {len(inner_train_perts)} | val perts (20%): {len(val_perts)} | "
          f"test perts: {len(test_perts)}")
    print(pd.Series(split).value_counts().to_string())

    #One copy of X per split; the model centers it in place (copy_input=False),
    #so no further full-size copies are made inside the model.
    adata_proc = ad.AnnData(
        X=adata.X.copy(),
        obs=pd.DataFrame({"condition": cond, "split": split}, index=obs.index),
        var=adata.var.copy(),
    )

    #train model
    model_path = os.path.join(MODEL_ROOT, split_col)
    os.makedirs(model_path, exist_ok=True)
    with timed("model init"):
        model = sclambda.model.Model(
            adata_proc, gene_embeds,
            model_path=model_path,
            multi_gene=False,
            copy_input=False,
            batch_size=TRAIN_BATCH_SIZE,
            training_epochs=TRAIN_EPOCHS,
            train_data_placement=TRAIN_DATA_PLACEMENT,
        )
    with timed("training"):
        model.train()

    ctrl_cells = shared["ctrl_cells"]
    ctrl_log1p_mean = shared["ctrl_mean"]
    print(f"  K562 control cells used (no subsampling): {ctrl_cells.shape[0]}")

    #evaluate on this split's K562 test set
    pert_test = sorted(test_perts)
    with timed("prediction"):
        pred_pseudobulk_log1p = predict_pseudobulk_batched(model, pert_test, ctrl_cells)

    results = {}
    pred_delta_mat = []
    for i in pert_test:
        pred_delta = pred_pseudobulk_log1p[i] - ctrl_log1p_mean
        true_delta = shared["true_pb"][i] - ctrl_log1p_mean
        results[i] = np.corrcoef(pred_delta, true_delta)[0, 1]
        pred_delta_mat.append(pred_delta)

    mean_pcc_test = float(np.nanmean(list(results.values())))
    seen_perts = set(cond[cat_col == "test_seen"]) - {"ctrl"}
    unseen_perts = set(cond[cat_col == "test_unseen"]) - {"ctrl"}
    pcc_seen = [results[g] for g in pert_test if g in seen_perts]
    pcc_unseen = [results[g] for g in pert_test if g in unseen_perts]

    #save predictions AnnData
    pred_delta_mat = np.array(pred_delta_mat)
    adata_pred = ad.AnnData(
        X=pred_delta_mat,
        obs=pd.DataFrame({
            "perturbation": pert_test,
            "pcc": [results[i] for i in pert_test],
            "test_category": ["test_seen" if g in seen_perts else "test_unseen" for g in pert_test],
        }, index=pert_test),
        var=adata.var.copy(),
    )
    adata_pred.layers["predicted_delta_"] = pred_delta_mat
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
        "mean_pcc_test_seen": mean_seen,
        "mean_pcc_test_unseen": mean_unseen, "mean_pcc_test_overall": mean_pcc_test,
        "model_path": model_path, "h5ad_path": h5ad_path,
    }

    del model, adata_proc, pred_pseudobulk_log1p
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    shared, gene_embeds, k562_cols = load_data()
    split_cols_to_run = SPLIT_COLS if SPLIT_COLS is not None else k562_cols
    split_cols_to_run = split_cols_to_run[args.shard::args.n_shards]

    csv_path = RESULTS_CSV_PATH
    if args.n_shards > 1:
        csv_path = RESULTS_CSV_PATH.replace(".csv", f"_shard{args.shard}of{args.n_shards}.csv")

    all_results = []
    for split_col in split_cols_to_run:
        t0 = time.time()
        try:
            r = run_split(shared, gene_embeds, split_col)
        except Exception as e:
            print(f"ERROR on {split_col}: {e}")
            traceback.print_exc()
            r = {"split_col": split_col, "status": f"error: {e}"}
        r["runtime_s"] = round(time.time() - t0, 1)
        all_results.append(r)
        pd.DataFrame(all_results).to_csv(csv_path, index=False)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll splits completed. Summary:")
    print(pd.DataFrame(all_results))
    print(f"\nFinal results CSV: {csv_path}")


if __name__ == "__main__":
    main()
