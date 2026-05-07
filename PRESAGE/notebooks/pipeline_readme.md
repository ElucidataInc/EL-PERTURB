# PRESAGE Training Pipeline Guide

This document walks through the full pipeline from raw data to predictions, covering each step in order.

---

## Overview

The pipeline has 5 stages:

```
1. Data Prep for HPO     →  prepare_presage_data_memopt.py (stage=hpo)
2. HPO Runs              →  presage_pbulk_train.py (stage=hpo)
3. Assemble Final Config →  Manual: extract best parameters from HPO results
4. Data Prep for Final   →  prepare_presage_data_memopt.py (stage=final)
5. Final Training + Pred →  presage_pbulk_train.py (stage=final)
```

Each stage must complete before the next can begin (HPO results feed into final config, etc.).

---

## Prerequisites

### Input Data

- **Single-cell h5ad file** (merged, log-normalized, HVG-filtered) with obs columns:
  - `cell_type` — cell line identifier (e.g. K562, RPE1, HEPG2, JURKAT)
  - `perturbation` — perturbation name (e.g. TP53, non-targeting)
  - `perturbation_group` — cell_type:perturbation (e.g. K562:TP53)
  - Split columns — e.g. `K562_TF_10_UF_10_rs_2_stratified` with values: train, control, test_seen, test_unseen, excluded

### PRESAGE Config Files

Located in the PRESAGE dircetory under `configs/` :

- `defaults_config.json` / `defaults_config_noep.json` — default hyperparameters (HPO / final)
- `singles_config.json` / `singles_config_noep.json` — single-value overrides (HPO / final)
- `ds_config.json` — dataset-specific config (paths, gene counts, etc.)

### PRESAGE Prior Knowledge Files

- To run PRESAGE, you will need to download the cached gene embeddings (for 40 knowledge sources) from https://zenodo.org/records/15587986.
- Use the above to populate the "other_embeddings", "pathway_embeddings" folders within `PRESAGE/cache/`
- Pre-computed Cell type context embeddings file (for kpdp/multiCtx variant) are available in `PRESAGE/cache/celltype_embeddings` 

### Search Space JSON

For HPO, a JSON file defining the hyperparameter search space. Example:

```json
{
    "training.lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": true},
    "training.weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": true},
    "model.n_head": {"type": "categorical", "choices": [2, 4, 8]},
    "model.d_model": {"type": "categorical", "choices": [128, 256, 512]}
}
```

For final training, the same JSON format but with fixed values (first choice is used):

```json
{
    "training.lr": {"type": "categorical", "choices": [0.001]},
    "training.weight_decay": {"type": "categorical", "choices": [0.0001]},
    "model.n_head": {"type": "categorical", "choices": [4]},
    "model.d_model": {"type": "categorical", "choices": [256]},
    "num_epochs": {"type": "categorical", "choices": [25]}
}
```

---

## Stage 1: Data Prep for HPO

Prepares pseudobulked datasets, split files, and per-fold gene embeddings for HPO. The HPO stage **excludes test perturbations** from the data entirely — only train + control cells are kept.

### What it produces

```
output_dir/
├── prepare_metadata.json
├── dataset/
│   ├── dataset.h5ad                    # Pseudobulk (train+control only)
│   ├── dataset_preprocessed.h5ad
│   ├── ncells_per_perturbation.json
│   └── degs/
│       └── merged.degs.json            # Stub DEGs (with --skip_degs)
├── splits/
│   ├── seed_0.json                     # Fold 0: {train: [...], val: [...]}
│   ├── seed_1.json                     # Fold 1
│   └── ...                             # No "test" key in HPO splits
└── embeddings/
    ├── fold_0_precomputed_coexpression_emb.pkl
    ├── fold_0_precomputed_transpose_matrix_emb.pkl
    ├── fold_1_precomputed_coexpression_emb.pkl
    └── ...
```

### Single run

```bash
python prepare_presage_data_memopt_scdata.py \
    --scdata_file data/merged_sc.h5ad \
    --split_col K562_TF_10_UF_10_rs_2_stratified \
    --target_name K562 \
    --target_key cell_type \
    --pert_key perturbation \
    --pert_group_key perturbation_group \
    --control_key non-targeting \
    --prior_type all \
    --prior_by_group cell_type \
    --n_nmf_embedding 128 \
    --stage hpo \
    --variant kpdp \
    --mode target_fold_5 \
    --skip_degs \
    --output_dir data/hpo_kpdp_K562_TF10_stratified/
```

### Key flags

| Flag | HPO value | Description |
|------|-----------|-------------|
| `--stage` | `hpo` | Keeps only train+control cells (test excluded) |
| `--variant` | `kpdp` / `only` / `pplus` | kpdp=all CTs, only/pplus=target CT only |
| `--mode` | `target_fold_5` | 5-fold CV splits, per-fold embeddings |
| `--skip_degs` | set | Writes stub DEGs (instant, safe for train+predict) |
| `--prior_type` | `all` | Compute both data-derived and knowledge embeddings |
| `--prior_by_group` | `cell_type` | Compute embeddings separately per cell type |

---

## Stage 2: HPO Runs

Runs Optuna-based hyperparameter optimization using 5-fold cross-validation. Each fold uses its fold-specific precomputed embeddings from Stage 1.

### What it produces

```
PRESAGE/results/hpo_kpdp_K562_TF10_stratified/
├── *_final_summary.json     # Best trial, all trial results
├── *_study.pkl              # Pickled Optuna study (resumable)
└── trial_*/
    └── trial_result.json    # Per-trial val loss and params
```

### Single run

```bash
python presage_pbulk_train.py \
    --stage hpo \
    --variant kpdp \
    --prepared_dir data/hpo_kpdp_K562_TF10_stratified/ \
    --ds_config_file configs/ds_config.json \
    --hpo_num_folds 5 \
    --hpo_search_space configs/search_space.json \
    --n_optuna_trials 50 \
    --optuna_sampler tpe \
    --optuna_pruner median
```

### Key flags

| Flag | Description |
|------|-------------|
| `--prepared_dir` | Output directory from Stage 1 |
| `--ds_config_file` | PRESAGE dataset config |
| `--hpo_num_folds` | Must match `--mode target_fold_N` from Stage 1 |
| `--hpo_search_space` | JSON defining param search ranges |
| `--n_optuna_trials` | Number of Optuna trials |
| `--optuna_storage` | Optional: `sqlite:///optuna.db` for resumable studies |
| `--optuna_sampler` | `tpe` (default), `random`, or `cmaes` |
| `--optuna_pruner` | `median` (default), `hyperband`, or `none` |

### Important notes

- HPO runs on GPU. One GPU per job.
- Each trial trains 5 folds sequentially. With inter-fold pruning, bad trials are killed early.
- The model saves checkpoints based on best validation loss per fold.

---

## Stage 3: Assemble Final Config

After HPO completes, inspect the results to determine the number of training epochs for final training.

### Steps

1. **Find the best epoch count across HPO runs:**

```python
import json
import glob

# Collect best epochs from all HPO summaries for a given variant/target
summaries = glob.glob("results/hpo_kpdp_K562_*/final_summary.json")

best_epochs = []
for path in summaries:
    with open(path) as f:
        summary = json.load(f)
    # The best_params may contain epoch info, or check the trial logs
    print(f"{path}: best_trial={summary['best_trial_number']}, "
          f"loss={summary['best_mean_cv_loss']:.6f}")

# Alternatively, check the model checkpoint filenames for epoch numbers
# or parse the per-trial result JSONs
```

2. **Determine the max best epoch** across all folds and splits. Use this as `num_epochs` for final training.

3. **Create the final search space JSON** with fixed hyperparameters:

```json
{
    "training.lr": {"type": "categorical", "choices": [0.001]},
    "training.weight_decay": {"type": "categorical", "choices": [0.0001]},
    "model.n_head": {"type": "categorical", "choices": [4]},
    "model.d_model": {"type": "categorical", "choices": [256]},
    "num_epochs": {"type": "categorical", "choices": [25]}
}
```

The `num_epochs` value comes from the max best epoch observed during HPO. All other hyperparameters are set to the best values found, or kept at defaults if only epochs were optimized.

---

## Stage 4: Data Prep for Final Training

Similar to Stage 1 but with two critical differences:

1. **Test perturbations are included** in the data (stage=final keeps everything except "excluded")
2. **Single split** with train/val/test (mode=None, not K-fold)
3. **One set of embeddings** (not per-fold)

### What it produces

```
output_dir/
├── prepare_metadata.json
├── dataset/
│   ├── dataset.h5ad                    # Pseudobulk (train+test, no excluded)
│   ├── dataset_preprocessed.h5ad
│   └── degs/merged.degs.json
├── splits/
│   └── seed_0.json                     # {train: [...], val: [...], test: [...]}
└── embeddings/
    ├── precomputed_coexpression_emb.pkl
    └── precomputed_transpose_matrix_emb.pkl
```

### Single run

```bash
python prepare_presage_data_memopt_scdata.py \
    --scdata_file data/merged_sc.h5ad \
    --split_col K562_TF_10_UF_10_rs_2_stratified \
    --target_name K562 \
    --target_key cell_type \
    --pert_key perturbation \
    --pert_group_key perturbation_group \
    --control_key non-targeting \
    --prior_type all \
    --prior_by_group cell_type \
    --n_nmf_embedding 128 \
    --stage final \
    --variant kpdp \
    --mode None \
    --skip_degs \
    --output_dir data/final_kpdp_K562_TF10_stratified/
```

### Key differences from Stage 1

| | HPO (Stage 1) | Final (Stage 4) |
|-|---------------|-----------------|
| `--stage` | `hpo` | `final` |
| `--mode` | `target_fold_5` | `None` |
| Data kept | train + control only | everything except excluded |
| Split format | train/val per fold (no test) | train/val/test (single split) |
| Embeddings | per-fold (5 sets) | single set |

---

## Stage 5: Final Training + Prediction

Trains the model for a fixed number of epochs (from Stage 3) and generates predictions on the test set.

### What it produces

```
PRESAGE/results/final_kpdp_K562_TF10_stratified/
├── predictions/
│   └── fold_0_predictions.csv          # Predicted expression for test perts
└── saved_models/
    └── last.ckpt                        # Final model checkpoint
```

### Single run

```bash
python presage_pbulk_train.py \
    --stage final \
    --variant kpdp \
    --use_scheduler \
    --prepared_dir data/final_kpdp_K562_TF10_stratified/ \
    --ds_config_file configs/ds_config.json \
    --hpo_num_folds 1 \
    --hpo_search_space configs/final_hparams.json
```

### Key flags

| Flag | Description |
|------|-------------|
| `--stage final` | Fixed-epoch training, no validation early stopping |
| `--use_scheduler` | Enables cosine annealing LR schedule |
| `--hpo_num_folds 1` | Single fold (no CV in final training) |
| `--hpo_search_space` | JSON with fixed params + `num_epochs` from Stage 3 |

### Important notes

- Final training does NOT use validation loss for model selection — it trains for exactly `num_epochs` and saves the last checkpoint.
- After training, the script automatically runs prediction on the test set (test_seen + test_unseen).
- Predictions are saved as CSV files with perturbation names as rows and gene names as columns.
- Uses cosine annealing LR scheduler (`--use_scheduler`) for stable training.

---

## Variant Reference

| Variant | Training data | Use case |
|---------|--------------|----------|
| `kpdp` | All cell types | Multi-context model with knowledge + data priors |
| `only` | Target cell type only | Single-context baseline |
| `pplus` | Target cell type only | Single-context with different prior config |

For `kpdp`, other cell types' perturbation data is always included in the training set. The K-fold split (HPO) or train/test split (final) applies only to the target cell type's perturbations.

---
