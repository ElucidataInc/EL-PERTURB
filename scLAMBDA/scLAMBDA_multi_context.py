#Multi-context training

#Usage:
#python scLAMBDA_multicontext.py
#CUDA_VISIBLE_DEVICES=0 python scLAMBDA_multicontext.py --shard 0 --n-shards 2 &
#CUDA_VISIBLE_DEVICES=1 python scLAMBDA_multicontext.py --shard 1 --n-shards 2 &

import os
import sys
import gc
import time
import pickle
import argparse
import warnings
import traceback
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import anndata as ad
import torch

warnings.filterwarnings("ignore", category=Warning)

#config
SCLAMBDA_REPO_PATH = "/home/scLAMBDA"
ADATA_PATH = "/home/GW_data/k10_merged_GW_screens_lognorm_single_cell.h5ad"
GENE_EMB_PATH = "/home/PRESAGE_gene_embeddings.pickle"
CONT_EMB_PATH = "/home/large_screen_cell_type_features_128d.pkl"

MODEL_ROOT = "/home/scLAMBDA/multi_context/k562_multicontext_models"
H5AD_OUTPUT_DIR = "/home/scLAMBDA/multi_context/k562_multicontext_predictions"
RESULTS_CSV_PATH = "/home/scLAMBDA/multi_context/k562_multicontext_split_results.csv"

TARGET_CONTEXT = "K562"
OTHER_CONTEXTS = ["D1", "HCT", "HEK"]

CTRL_LABEL = "non-targeting"
SEED = 0
VAL_FRACTION = 0.2
EXPERIMENT_SUFFIXES = None


#speed knobs
PRED_BATCH_SIZE = 8192
TRAIN_BATCH_SIZE = 500        # scLAMBDA default
TRAIN_EPOCHS = 100            # scLAMBDA default for Model_context
TRAIN_DATA_PLACEMENT = "auto" # "auto" | "gpu" | "cpu"
SHUFFLE_TRAIN = True          # original Model_context did NOT shuffle batches; False = reproduce old behaviour
ALLOW_TF32 = True
SKIP_EXISTING = True

#setup
sys.path.insert(0, SCLAMBDA_REPO_PATH)
import sclambda
from sclambda.model import gather_dense_rows, group_means
print("sclambda.model loaded from:", sclambda.model.__file__)

torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
torch.backends.cudnn.allow_tf32 = ALLOW_TF32

os.makedirs(MODEL_ROOT, exist_ok=True)
os.makedirs(H5AD_OUTPUT_DIR, exist_ok=True)

ALL_CONTEXTS = [TARGET_CONTEXT] + OTHER_CONTEXTS


@contextmanager
def timed(label):
    t0 = time.time()
    yield
    print(f"  [time] {label}: {time.time() - t0:.1f}s")


def read_rows(path, keep_fn):
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


def load_cont_emb(path, expected_contexts):
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            raw = pickle.load(f)
    else:
        raw = np.load(path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
            raw = raw.item()

    if isinstance(raw, dict):
        cont_emb = {str(k): np.asarray(v, dtype=np.float32).reshape(-1) for k, v in raw.items()}
    elif isinstance(raw, np.ndarray) and raw.ndim == 2:
        print(" WARNING: cont_emb file loaded as a plain "
              f"{raw.shape} array with no keys. Assuming row order == {expected_contexts}")
        cont_emb = {ctx: raw[i].astype(np.float32) for i, ctx in enumerate(expected_contexts)}
    else:
        raise ValueError(f"Unrecognized structure for cont_emb file at {path}: type={type(raw)}. "
                         "Expected a dict or a 2D array.")

    print(f"Loaded cont_emb with keys: {sorted(cont_emb.keys())}")
    missing = [ctx for ctx in expected_contexts if ctx not in cont_emb]
    if missing:
        raise ValueError(f"cont_emb is missing embeddings for contexts {missing}. "
                         f"Keys found : {sorted(cont_emb.keys())}.")
    dims = {ctx: cont_emb[ctx].shape[0] for ctx in expected_contexts}
    if len(set(dims.values())) > 1:
        raise ValueError(f"cont_emb vectors have inconsistent dims: {dims}")
    return cont_emb


def load_data():
    gene_embeddings = pd.read_pickle(GENE_EMB_PATH)
    emb_keys = set(gene_embeddings.keys())
    cont_emb = load_cont_emb(CONT_EMB_PATH, ALL_CONTEXTS)

    def keep_fn(obs):
        in_ctx = np.isin(obs["context"].astype(str).values, ALL_CONTEXTS)
        pert = obs["perturbation"].astype(str).values
        perts = set(pert[in_ctx]) - {CTRL_LABEL}
        missing = perts - emb_keys
        print(f"Gene embeddings matched: {len(perts) - len(missing)}/{len(perts)} | missing: {len(missing)}")
        return in_ctx & ~np.isin(pert, list(missing))

    print(f"Loading adata rows for contexts {ALL_CONTEXTS}...")
    with timed("read h5ad"):
        adata = read_rows(ADATA_PATH, keep_fn)

    m_cols = ["cell_type", "source_file", "context", "perturbation_group", "perturbation"]

    #only the target (K562) split columns define the experiments
    split_cols = [c for c in adata.obs.columns if c.startswith(f"{TARGET_CONTEXT}_")]
    adata.obs = adata.obs[m_cols + split_cols].copy()
    common_suffixes = {c[len(TARGET_CONTEXT) + 1:] for c in split_cols}
    print(f"Experiment suffixes (from {TARGET_CONTEXT} split columns only): {len(common_suffixes)}")
    for s in sorted(common_suffixes):
        print("  -", s)

    #per-cell columns that are identical for every experiment: build once
    pert = adata.obs["perturbation"].astype(str).values
    ctx = adata.obs["context"].astype(str).values
    adata.obs["condition"] = np.where(pert == CTRL_LABEL, "ctrl", pert)
    adata.obs["cell_type"] = ctx          # model uses cell_type as the context key
    adata.obs["cell_type+condition"] = pd.Series(ctx, index=adata.obs.index) + "_---_" + adata.obs["condition"]

    gene_embeds = {g: np.asarray(gene_embeddings[g], dtype=np.float32)
                   for g in np.unique(pert) if g != CTRL_LABEL}
    print(f"Cells: {adata.n_obs} | genes: {adata.n_vars}")
    return adata, gene_embeds, cont_emb, sorted(common_suffixes)


@torch.no_grad()
def predict_pseudobulk_batched(model, pert_test, ctrl_cells_absolute, target_context, batch_size=PRED_BATCH_SIZE):
    model.Net.eval()
    device = model.device
    ctrl_mean = torch.as_tensor(np.asarray(model.ctrl_mean[target_context]).reshape(-1),
                                dtype=torch.float32, device=device)
    ct1 = torch.as_tensor(np.asarray(model.cont_emb[target_context]), dtype=torch.float32, device=device).view(1, -1)
    ctrl_delta = torch.from_numpy(ctrl_cells_absolute).to(device) - ctrl_mean
    n_ctrl = ctrl_delta.shape[0]

    results_mean = {}
    for i in pert_test:
        if model.multi_gene:
            genes = i.split('+')
            pert_emb_p = model.gene_emb[genes[0]] + model.gene_emb[genes[1]]
        else:
            pert_emb_p = model.gene_emb[i]
        p1 = torch.as_tensor(np.asarray(pert_emb_p), dtype=torch.float32, device=device).view(1, -1)
        d_sum = torch.zeros_like(ctrl_mean)
        for start in range(0, n_ctrl, batch_size):
            batch = ctrl_delta[start:start + batch_size]
            bs = batch.shape[0]
            x_hat = model.Net(batch, p1.expand(bs, -1), ct1.expand(bs, -1))[0]
            d_sum += x_hat.sum(0)
        results_mean[i] = (d_sum / n_ctrl + ctrl_mean).cpu().numpy()

    del ctrl_delta
    return results_mean


def run_experiment(adata, gene_embeds, cont_emb, suffix):
    print("\n" + "=" * 80)
    print(f"Running multi-context experiment: {suffix}  "
          f"(target: {TARGET_CONTEXT}, others: {OTHER_CONTEXTS})")
    print("=" * 80)

    h5ad_path = os.path.join(H5AD_OUTPUT_DIR, f"{TARGET_CONTEXT}_{suffix}.h5ad")
    if SKIP_EXISTING and os.path.exists(h5ad_path):
        print(f"  SKIPPING: output already exists at {h5ad_path}")
        return {"experiment": suffix, "status": "skipped_already_exists", "h5ad_path": h5ad_path}

    obs_all = adata.obs
    ctx_all = obs_all["context"].astype(str).values

    target_split_col = f"{TARGET_CONTEXT}_{suffix}"
    if target_split_col not in obs_all.columns:
        print(f"  WARNING: column {target_split_col} missing - skipping this experiment")
        return {"experiment": suffix, "status": "skipped_missing_column"}

    is_target_all = ctx_all == TARGET_CONTEXT

    #role for every cell (all contexts) comes from the target split column
    role = obs_all[target_split_col].astype(str).values

    print(f"  Role by context (from {target_split_col}):")
    print(pd.crosstab(ctx_all, role).to_string())

    keep = (role == "train") | (role == "control") | (is_target_all & np.isin(role, ["test_seen", "test_unseen"]))
    rows = np.flatnonzero(keep)

    obs = obs_all.iloc[rows][["context", "cell_type", "condition", "cell_type+condition", target_split_col]].copy()
    obs["role"] = role[rows]

    ctx = obs["context"].values
    cond = obs["condition"].values
    role_k = obs["role"].values
    train_mask = role_k == "train"
    target_mask = ctx == TARGET_CONTEXT
    is_test_role = np.isin(role_k, ["test_seen", "test_unseen"])
    test_mask = target_mask & is_test_role
    assert ((~target_mask) & is_test_role).sum() == 0, "non-target test cells leaked into adata_proc"

    all_train_perts = set(cond[train_mask]) - {"ctrl"}
    target_train_perts = set(cond[train_mask & target_mask]) - {"ctrl"}
    test_perts = set(cond[test_mask]) - {"ctrl"}

    if len(target_train_perts) < 2 or len(test_perts) < 1:
        print(f"  WARNING: not enough {TARGET_CONTEXT} perturbations ({len(target_train_perts)} train, "
              f"{len(test_perts)} test) — skipping")
        return {"experiment": suffix, "status": "skipped_too_few_perts"}

    rng = np.random.default_rng(SEED)
    target_train_arr = np.array(sorted(target_train_perts))
    rng.shuffle(target_train_arr)
    n_val = max(1, int(round(VAL_FRACTION * len(target_train_arr))))
    val_perts = set(target_train_arr[:n_val])

    split = np.full(len(rows), "train", dtype=object)
    split[np.isin(cond, list(val_perts)) & train_mask & target_mask] = "val"
    split[test_mask] = "test"
    obs["split"] = split

    assert set(ctx[split == "val"]) <= {TARGET_CONTEXT}
    assert set(ctx[split == "test"]) <= {TARGET_CONTEXT}

    print(f"  target ({TARGET_CONTEXT}) train perts: {len(target_train_perts)} | "
          f"Val perts (20% of target train, target-context only): {len(val_perts)} | "
          f"Total train perts across all contexts: {len(all_train_perts)} | "
          f"Test perts (target-context only): {len(test_perts)}")
    print(obs["split"].value_counts().to_string())
    print(obs.groupby(["context", "split"]).size().to_string())

    #single dense materialization of the rows we need (no intermediate full copies)
    with timed("densify rows"):
        X = gather_dense_rows(adata.X, rows)

    #everything evaluation needs from absolute (uncentered) values - taken BEFORE
    #the model centers X in place
    with timed("eval stats"):
        k562_ctrl_cells_full = X[target_mask & (cond == "ctrl")]   # copy
        ctrl_mean_k562 = k562_ctrl_cells_full.mean(axis=0)
        cat = pd.Categorical(cond)
        codes = np.where(target_mask, cat.codes, -1).astype(np.int64)  # all K562 cells of each pert
        means, counts = group_means(X, codes, len(cat.categories))
        true_pb = {g: means[k] for k, g in enumerate(cat.categories) if counts[k] > 0}

    adata_proc = ad.AnnData(X=X, obs=obs, var=adata.var.copy())

    #train model
    model_path = os.path.join(MODEL_ROOT, suffix)
    os.makedirs(model_path, exist_ok=True)
    with timed("model init"):
        model = sclambda.model.Model_context(
            adata_proc, gene_embeds, cont_emb,
            model_path=model_path,
            multi_gene=False,
            split_name="split",
            copy_input=False,
            batch_size=TRAIN_BATCH_SIZE,
            training_epochs=TRAIN_EPOCHS,
            train_data_placement=TRAIN_DATA_PLACEMENT,
            shuffle_train=SHUFFLE_TRAIN,
        )
    with timed("training"):
        model.train()

    print(f"ctrl_mean computed for contexts: {sorted(model.ctrl_mean.keys())}")
    print(f"  {TARGET_CONTEXT} control cells available and used (no subsampling): {k562_ctrl_cells_full.shape[0]}")

    #evaluate on K562 test set
    pert_test = sorted(test_perts)
    with timed("prediction"):
        pred_pseudobulk = predict_pseudobulk_batched(
            model, pert_test, k562_ctrl_cells_full, target_context=TARGET_CONTEXT)

    results = {}
    pred_delta_mat = []
    for i in pert_test:
        pred_delta = pred_pseudobulk[i] - ctrl_mean_k562
        true_delta = true_pb[i] - ctrl_mean_k562
        results[i] = np.corrcoef(pred_delta, true_delta)[0, 1]
        pred_delta_mat.append(pred_delta)

    mean_pcc_test = float(np.nanmean(list(results.values())))
    tcol = obs[target_split_col].astype(str).values
    seen_perts = set(cond[target_mask & (tcol == "test_seen")]) - {"ctrl"}
    unseen_perts = set(cond[target_mask & (tcol == "test_unseen")]) - {"ctrl"}
    pcc_seen = [results[g] for g in pert_test if g in seen_perts]
    pcc_unseen = [results[g] for g in pert_test if g in unseen_perts]
    mean_pcc_seen = float(np.nanmean(pcc_seen)) if pcc_seen else np.nan
    mean_pcc_unseen = float(np.nanmean(pcc_unseen)) if pcc_unseen else np.nan

    #save predictions
    pred_delta_mat = np.array(pred_delta_mat)
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
        var=adata.var.copy(),
    )
    adata_pred.layers["predicted_delta_"] = pred_delta_mat
    adata_pred.write_h5ad(h5ad_path)

    print(f"  Mean PCC (delta, {TARGET_CONTEXT} eval) - "
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

    del model, adata_proc, X, k562_ctrl_cells_full, pred_pseudobulk
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    adata, gene_embeds, cont_emb, common_suffixes = load_data()
    suffixes_to_run = EXPERIMENT_SUFFIXES if EXPERIMENT_SUFFIXES is not None else common_suffixes
    suffixes_to_run = suffixes_to_run[args.shard::args.n_shards]

    if not suffixes_to_run:
        raise SystemExit("No experiment suffixes to run - check K562 split column naming / shard args")

    csv_path = RESULTS_CSV_PATH
    if args.n_shards > 1:
        csv_path = RESULTS_CSV_PATH.replace(".csv", f"_shard{args.shard}of{args.n_shards}.csv")

    all_results = []
    for suffix in suffixes_to_run:
        t0 = time.time()
        try:
            r = run_experiment(adata, gene_embeds, cont_emb, suffix)
        except Exception as e:
            print(f"ERROR on experiment {suffix}: {e}")
            traceback.print_exc()
            r = {"experiment": suffix, "status": f"error: {e}"}
        r["runtime_s"] = round(time.time() - t0, 1)
        all_results.append(r)
        pd.DataFrame(all_results).to_csv(csv_path, index=False)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(csv_path, index=False)
    print("\nAll experiments completed. Summary:")
    print(results_df)
    print(f"\nFinal results CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
    