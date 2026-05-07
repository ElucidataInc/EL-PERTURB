"""
PRESAGE Unified Training Script (Pseudobulk Mode)
==================================================

Unified script that replaces the 6 separate SC-based scripts:
  - presage_model_hpo_script_optuna.py          (HPO, kpdp variant)
  - presage_model_hpo_only_target_script_optuna.py  (HPO, only variant)
  - presage_model_hpo_pplus_script_optuna.py     (HPO, pplus variant)
  - presage_model_final_orig_Cosine.py           (Final, kpdp variant)
  - presage_model_final_only_target.py           (Final, only variant)
  - presage_model_final_pplus.py                 (Final, pplus variant)

This script consumes the OUTPUT of prepare_presage_data.py (pseudobulk h5ad,
pre-computed DEGs, splits, and SC-derived embeddings). It no longer reads
single-cell data directly.

Key flags:
  --stage hpo|final         HPO (Optuna K-fold CV) or Final (train + predict)
  --variant kpdp|only|pplus Which PRESAGE configuration variant to use
  --use_scheduler           Use cosine LR scheduler (ModelHarness_scheduler)

Usage (HPO):
    python presage_pbulk_train.py \\
        --stage hpo \\
        --variant kpdp \\
        --prepared_dir /path/to/prepare_presage_data_output/ \\
        --ds_config_file /path/to/ds_config.json \\
        --hpo_num_folds 5 \\
        --n_optuna_trials 50

Usage (Final training + prediction):
    python presage_pbulk_train.py \\
        --stage final \\
        --variant only \\
        --use_scheduler \\
        --prepared_dir /path/to/prepare_presage_data_output/ \\
        --ds_config_file /path/to/ds_config.json \\
        --hpo_search_space /path/to/best_hparams.json
"""

import sys
from pathlib import Path

PRESAGE_PATH = Path("/home/jl_fs/PRESAGE")
sys.path.append(str(PRESAGE_PATH / "src"))

from train import str2bool, set_seed, parse_config, get_predictions, get_attention, get_embedding

import json
from copy import deepcopy
import argparse
import datetime
import gc
import os
import pickle as pkl
import logging
import shutil
import re

import numpy as np
import pandas as pd
import scanpy as sc

import optuna
from optuna.exceptions import TrialPruned

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from datamodule_multicell_pbulk import ReplogleDataModule
from presage_datamodule import ReploglePRESAGEDataModule, PRESAGEDataModule
from presage_context_embs_pbulk import PRESAGE

logger = logging.getLogger(__name__)


# ===========================================================================
# Config helpers
# ===========================================================================

def get_updated_config_file(ds_config_file, defaults_config, singles_config, prior_file = None):
    """Merge default -> singles -> dataset configs and apply overrides."""
    with open(defaults_config, "r") as f:
        config = json.load(f)
    with open(singles_config, "r") as f:
        sc_config = json.load(f)
    with open(ds_config_file, "r") as f:
        ds_config = json.load(f)
        if prior_file:
            ds_config["model_pathway_files"] = prior_file

    sc_config.update(ds_config)

    new_config = {}
    for key, value in sc_config.items():
        if value is not None and key not in {"config", "data_config"}:
            new_config[key.replace("_", ".", 1)] = value
    sc_config = new_config
    config.update(sc_config)

    modify_config = {
        "training.eval_test": False,
        "data.data_dir": str(PRESAGE_PATH / "data"),
    }
    config.update(modify_config)
    return config


def load_search_space(path):
    with open(path, "r") as f:
        return json.load(f)


def inject_pseudobulk_base_config(config, metadata):
    """Inject pseudobulk dataset paths (stage-independent, fold-independent)."""
    paths = metadata["paths"]
    config["data.pre_pseudobulked"] = True
    config["data.dataset"] = os.path.basename(paths["dataset_dir"])
    config["data.data_dir"] = os.path.dirname(paths["dataset_dir"])
    return config


def inject_fold_embeddings(config, metadata, fold_idx=0):
    """Inject the correct precomputed embedding paths for a given fold.
    
    For direct mode (single split): uses top-level embedding paths.
    For K-fold mode: uses per_fold_embeddings[fold_idx] paths.
    """
    paths = metadata["paths"]
    per_fold = paths.get("per_fold_embeddings", None)

    if per_fold is not None:
        fold_emb = next((e for e in per_fold if e["fold"] == fold_idx), None)
        if fold_emb is None:
            raise ValueError(f"No embeddings found for fold {fold_idx} in metadata")
        coex_path = fold_emb.get("coexpression_emb", None)
        trans_path = fold_emb.get("transpose_matrix_emb", None)
    else:
        coex_path = paths.get("coexpression_emb", None)
        trans_path = paths.get("transpose_matrix_emb", None)

    config["model.precomputed_coexpression_emb"] = coex_path if coex_path else "None"
    config["model.precomputed_transpose_matrix_emb"] = trans_path if trans_path else "None"
    return config


# ===========================================================================
# HPO helpers
# ===========================================================================

DEFAULT_SEARCH_SPACE = {
    "model.lr": {"type": "categorical", "choices": [1e-3]}
} # edit to add other hyperparams that can be optimized


def suggest_hyperparameters_hpo(trial, search_space):
    """Optuna trial-based suggestion."""
    suggested = {}
    for key, spec in search_space.items():
        param_type = spec["type"]
        if param_type == "float":
            suggested[key] = trial.suggest_float(
                key, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif param_type == "int":
            suggested[key] = trial.suggest_int(
                key, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif param_type == "categorical":
            suggested[key] = trial.suggest_categorical(key, spec["choices"])
        else:
            raise ValueError(f"Unknown type '{param_type}' for key '{key}'")
    return suggested


def suggest_hyperparameters_final(search_space):
    """Non-Optuna: take first choice from each param. Extracts num_epochs."""
    suggested = {}
    num_epochs = 1
    for key, spec in search_space.items():
        if key == "num_epochs":
            num_epochs = int(spec["choices"][0])
            print(f"Set Num Epochs = {num_epochs}")
            continue
        suggested[key] = spec["choices"][0]
    return suggested, num_epochs


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ===========================================================================
# Single fold training
# ===========================================================================

def train_single_fold(
    training_config,
    dataset_path,
    splits_path,
    fold_idx,
    run_name,
    use_scheduler=False,
    trial=None,
    models_save_dir=str(PRESAGE_PATH / "saved_models"),
    num_epochs=None,
    stage="hpo",
):
    """Train one fold.
    
    For stage='hpo': returns (best_val_loss, best_model_filename, best_epoch)
    For stage='final': returns avg_predictions DataFrame
    """
    # Import the right ModelHarness
    if use_scheduler:
        from model_harness_scheduler import ModelHarness
    else:
        from model_harness import ModelHarness

    config = parse_config(training_config)
    set_seed(config["training"].pop("seed", None))
    config["training"].pop("offline", False)
    do_test_eval = config["training"].pop("eval_test", True)
    config["training"].pop("predictions_file", None)
    config["training"].pop("embedding_file", None)
    config["training"].pop("attention_file", None)

    config["data"]["dataset"] = dataset_path.name
    config["data"]["data_dir"] = str(dataset_path.parent)
    config["data"]["seed"] = str(splits_path / f"seed_{fold_idx}.json")

    # ---- DataModule ----
    seed = config["data"].pop("seed")
    datamodule = ReplogleDataModule.from_config(config["data"])
    datamodule.do_test_eval = do_test_eval

    if hasattr(datamodule, "set_seed"):
        datamodule.set_seed(seed)
    config["data"]["seed"] = seed

    print(datamodule.celltype_key, datamodule.perturb_group_key)
    datamodule.prepare_data()
    datamodule.setup("fit")
    print("datamodule setup complete.")

    # ---- Model ----
    print("\n\nInitializing Model.")
    model_config = config["model"]
    model_config["dataset"] = dataset_path.name
    model_config["pca_dim"] = None
    model_config["source"] = "temp"
    model_config["learnable_gene_embedding"] = False

    module = PRESAGE(
        model_config,
        datamodule,
        datamodule.pert_covariates.shape[1],
        datamodule.n_genes,
    )
    if hasattr(module, "custom_init"):
        module.custom_init()

    lightning_module = ModelHarness(module, datamodule, model_config)
    print("model initialization complete.... initializing trainer and training run")

    # ---- Callbacks & Trainer ----
    trial_tag = f"trial_{trial.number}" if trial is not None else "notrial"

    log_logger = pl.loggers.CSVLogger(
        save_dir=str(PRESAGE_PATH / "logs"),
        name=config["data"]["dataset"],
        version=f"{run_name}_{trial_tag}_fold_{fold_idx}",
    )

    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if stage == "hpo":
        # HPO: early stopping + best checkpoint on val_loss
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-6, patience=10, verbose=True, mode="min",
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss", dirpath=models_save_dir,
            filename=(
                f"my_model-{config['data']['dataset']}-{trial_tag}"
                f"-fold_{fold_idx}-{now_str}-{{epoch:02d}}-{{val_loss:.2f}}"
            ),
            save_top_k=1, mode="min",
        )
        callbacks = [early_stop_callback, checkpoint_callback]

        trainer = pl.Trainer(
            logger=log_logger, log_every_n_steps=3, num_sanity_val_steps=10,
            callbacks=callbacks, reload_dataloaders_every_n_epochs=1,
            **config["training"], gradient_clip_val=0.1,
        )
    else:
        # Final: fixed epochs, no validation, save last checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=models_save_dir,
            filename=(
                f"my_model-{config['data']['dataset']}-{trial_tag}"
                f"-fold_{fold_idx}-{now_str}-{{epoch:02d}}"
            ),
            save_last=True,
        )
        trainer = pl.Trainer(
            logger=log_logger, log_every_n_steps=3,
            max_epochs=num_epochs, num_sanity_val_steps=0,
            callbacks=[checkpoint_callback], reload_dataloaders_every_n_epochs=1,
            **config["training"], gradient_clip_val=0.1, limit_val_batches=0,
        )

    if fold_idx == 0 and (trial is None or trial.number == 0):
        torch.autograd.set_detect_anomaly(True)
        print("!! Autograd anomaly detection ENABLED for trial 0, fold 0")

    trainer.fit(lightning_module, datamodule=datamodule)
    torch.autograd.set_detect_anomaly(False)

    # ---- Collect results ----
    if stage == "hpo":
        best_model_path = checkpoint_callback.best_model_path
        match = re.search(r"epoch=(\d+)", str(best_model_path))
        best_epoch = int(match.group(1)) if match else "None"
        best_val_loss = checkpoint_callback.best_model_score.item()
        best_model_filename = str(best_model_path).split("/")[-1]

        print(f"Best validation loss for fold_{fold_idx}: {best_val_loss}")

        trainer.strategy.teardown()
        trainer._teardown()
        del trainer, lightning_module, module, datamodule
        del log_logger, checkpoint_callback
        cleanup_gpu()

        return best_val_loss, best_model_filename, best_epoch

    else:
        # Final: load last checkpoint and predict
        last_ckpt = checkpoint_callback.last_model_path
        checkpoint = torch.load(last_ckpt)
        lightning_module.load_state_dict(checkpoint["state_dict"])

        datamodule._data_setup = False
        datamodule.setup("test")
        dataloader = datamodule.test_dataloader()

        ds = dataloader.dataset
        print(f"Dataset type: {type(ds)}, length: {len(ds)}")

        avg_predictions = get_predictions(
            trainer, lightning_module, dataloader, datamodule.var_names
        )
        print(f"Predicted Data Shape: {avg_predictions.shape}")
        print(avg_predictions.head(3))

        return avg_predictions


# ===========================================================================
# HPO objective (Optuna)
# ===========================================================================

def make_objective(
    base_training_config, dataset_path, splits_path, hpo_num_folds,
    run_name, run_path, search_space, use_scheduler, metadata,
):
    def objective(trial):
        suggested = suggest_hyperparameters_hpo(trial, search_space)
        trial_config = deepcopy(base_training_config)
        trial_config.update(suggested)

        models_save_dir = PRESAGE_PATH / "saved_models" / f"{run_name}_T{trial.number}"
        models_save_dir.mkdir()

        print(f"\n{'='*70}")
        print(f"OPTUNA TRIAL {trial.number}")
        print(f"  Suggested: {json.dumps(suggested, indent=2, default=str)}")
        print(f"{'='*70}\n")

        fold_val_losses, fold_best_epochs = [], []

        for fold_idx in range(hpo_num_folds):
            print(f"\n--- Trial {trial.number} | Fold {fold_idx + 1}/{hpo_num_folds} ---")

            # Inject fold-specific embedding paths
            fold_config = deepcopy(trial_config)
            fold_config = inject_fold_embeddings(fold_config, metadata, fold_idx)

            try:
                best_val_loss, _, best_epoch = train_single_fold(
                    training_config=fold_config,
                    dataset_path=dataset_path,
                    splits_path=splits_path,
                    fold_idx=fold_idx,
                    run_name=run_name,
                    use_scheduler=use_scheduler,
                    trial=trial,
                    models_save_dir=str(models_save_dir),
                    stage="hpo",
                )
            except TrialPruned:
                shutil.rmtree(models_save_dir)
                raise
            except Exception as e:
                print(f"  !! Fold {fold_idx} failed: {e}")
                shutil.rmtree(models_save_dir)
                cleanup_gpu()
                raise TrialPruned(f"Fold {fold_idx} failed: {e}")

            fold_val_losses.append(best_val_loss)
            fold_best_epochs.append(best_epoch)

            running_mean = np.mean(fold_val_losses)
            trial.report(running_mean, step=fold_idx)
            if trial.should_prune():
                print(f"  >> Trial {trial.number} pruned after fold {fold_idx}")
                shutil.rmtree(models_save_dir)
                raise TrialPruned()

        mean_cv_loss = float(np.mean(fold_val_losses))
        std_cv_loss = float(np.std(fold_val_losses))
        print(f"\nTrial {trial.number}: mean_val_loss={mean_cv_loss:.6f} +/- {std_cv_loss:.6f}")

        trial_result = {
            "trial_number": trial.number,
            "suggested_params": {k: str(v) for k, v in suggested.items()},
            "fold_val_losses": fold_val_losses,
            "mean_cv_loss": mean_cv_loss,
            "std_cv_loss": std_cv_loss,
            "fold_best_epochs": fold_best_epochs,
        }
        with open(run_path / f"trial_{trial.number}_results.json", "w") as f:
            json.dump(trial_result, f, indent=2)

        shutil.rmtree(models_save_dir)
        return mean_cv_loss

    return objective


# ===========================================================================
# Stage runners
# ===========================================================================

def run_hpo(args, training_config, dataset_path, splits_path, run_name, run_path, search_space, metadata):
    """Run Optuna HPO."""

    if args.optuna_sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=42)
    elif args.optuna_sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=42)
    elif args.optuna_sampler == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(seed=42)
    else:
        sampler = optuna.samplers.TPESampler(seed=42)

    if args.optuna_pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    elif args.optuna_pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=args.hpo_num_folds)
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=run_name, storage=args.optuna_storage,
        direction="minimize", sampler=sampler, pruner=pruner, load_if_exists=True,
    )

    objective = make_objective(
        base_training_config=training_config,
        dataset_path=dataset_path,
        splits_path=splits_path,
        hpo_num_folds=args.hpo_num_folds,
        run_name=run_name,
        run_path=run_path,
        search_space=search_space,
        use_scheduler=args.use_scheduler,
        metadata=metadata,
    )

    print(f"\n{'#'*70}")
    print(f"Starting Optuna study '{run_name}' with {args.n_optuna_trials} trials")
    print(f"  Sampler : {args.optuna_sampler}")
    print(f"  Pruner  : {args.optuna_pruner}")
    print(f"  Folds   : {args.hpo_num_folds}")
    print(f"  Storage : {args.optuna_storage or 'in-memory'}")
    print(f"{'#'*70}\n")

    study.optimize(objective, n_trials=args.n_optuna_trials, gc_after_trial=True)

    # Report
    best_trial = study.best_trial
    print(f"\n{'='*70}\nOPTIMIZATION COMPLETE\n{'='*70}")
    print(f"  Best trial: {best_trial.number}, loss: {best_trial.value:.6f}")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")

    summary = {
        "best_trial_number": best_trial.number,
        "best_mean_cv_loss": best_trial.value,
        "best_params": best_trial.params,
        "all_trials": [
            {"number": t.number, "value": t.value, "state": str(t.state), "params": t.params}
            for t in study.trials
        ],
    }
    with open(run_path / f"{run_name}_final_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(run_path / f"{run_name}_study.pkl", "wb") as f:
        pkl.dump(study, f)

    print(f"\nResults saved to: {run_path}")


def run_final(args, training_config, dataset_path, splits_path, run_name, run_path, search_space, metadata):
    """Run final training (fixed epochs) + prediction."""

    suggested, num_epochs = suggest_hyperparameters_final(search_space)
    trial_config = deepcopy(training_config)
    trial_config.update(suggested)

    models_save_dir = PRESAGE_PATH / "saved_models" / run_name
    models_save_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"FINAL TRAINING RUN")
    print(f"  Hyperparameters: {json.dumps(suggested, indent=2, default=str)}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Folds: {args.hpo_num_folds}")
    print(f"{'='*70}\n")

    fold_preds = {}
    for fold_idx in range(args.hpo_num_folds):
        print(f"\n--- Fold {fold_idx + 1}/{args.hpo_num_folds} ---")

        # Inject fold-specific embedding paths
        fold_config = deepcopy(trial_config)
        fold_config = inject_fold_embeddings(fold_config, metadata, fold_idx)

        avg_predictions = train_single_fold(
            training_config=fold_config,
            dataset_path=dataset_path,
            splits_path=splits_path,
            fold_idx=fold_idx,
            run_name=run_name,
            use_scheduler=args.use_scheduler,
            trial=None,
            models_save_dir=str(models_save_dir),
            num_epochs=num_epochs,
            stage="final",
        )
        fold_preds[fold_idx] = avg_predictions

    for fold_idx, preds in fold_preds.items():
        out_path = run_path / f"final_results_{fold_idx}.csv"
        preds.to_csv(out_path)
        print(f"  Saved fold {fold_idx} predictions -> {out_path}")

    print(f"\n{'='*70}\nTRAINING COMPLETE\n{'='*70}")
    print(f"Results saved to: {run_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRESAGE Unified Training Script (Pseudobulk Mode)"
    )

    # Core flags
    parser.add_argument("--stage", type=str, required=True, choices=["hpo", "final"],
                        help="hpo = Optuna HPO with K-fold CV; final = train + predict")
    parser.add_argument("--variant", type=str, required=True, choices=["kpdp", "only", "pplus"],
                        help="kpdp = all cell types; only/pplus = target cell type only")
    parser.add_argument("--use_scheduler", action="store_true",
                        help="Use ModelHarness with cosine LR scheduler")

    # Data: points to prepare_presage_data.py output
    parser.add_argument("--prepared_dir", type=str, required=True,
                        help="Output directory from prepare_presage_data.py")
    parser.add_argument("--ds_config_file", type=str, required=True,
                        help="Path to dataset-specific PRESAGE config JSON")

    # Config file overrides (defaults based on stage)
    parser.add_argument("--defaults_config", type=str, default=None,
                        help="Override default config file. "
                             "Default: defaults_config.json (hpo) or defaults_config_noep.json (final)")
    parser.add_argument("--singles_config", type=str, default=None,
                        help="Override singles config file. "
                             "Default: singles_config.json (hpo) or singles_config_noep.json (final)")
    parser.add_argument("--prior_files", type=str, default=None,
                        help="Overrides model_pathway_files parameter in ds config file. "
                             "Default: None - implies use the file mentioned in ds config only")
    # Fold / run args
    parser.add_argument("--hpo_num_folds", type=int, default=5)
    parser.add_argument("--hpo_search_space", type=str, default=None,
                        help="Path to search space JSON")
    parser.add_argument("--split_name", type=str, default=None)

    # Optuna-specific (only used when stage=hpo)
    parser.add_argument("--n_optuna_trials", type=int, default=100)
    parser.add_argument("--optuna_storage", type=str, default=None)
    parser.add_argument("--optuna_pruner", type=str, default="median",
                        choices=["median", "hyperband", "none"])
    parser.add_argument("--optuna_sampler", type=str, default="tpe",
                        choices=["tpe", "random", "cmaes"])

    args = parser.parse_args()

    # ---- Resolve config files based on stage ----
    if args.defaults_config is None:
        if args.stage == "hpo":
            args.defaults_config = str(PRESAGE_PATH / "configs/defaults_config.json")
        else:
            args.defaults_config = str(PRESAGE_PATH / "configs/defaults_config_noep.json")

    if args.singles_config is None:
        if args.stage == "hpo":
            args.singles_config = str(PRESAGE_PATH / "configs/singles_config.json")
        else:
            args.singles_config = str(PRESAGE_PATH / "configs/singles_config_noep.json")

    # ---- Load prepare_presage_data metadata ----
    metadata_path = os.path.join(args.prepared_dir, "prepare_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print(f"Loaded metadata from: {metadata_path}")

    paths = metadata["paths"]
    dataset_path = Path(paths["dataset_dir"])
    splits_path = Path(paths["splits_dir"])

    # ---- Build run name ----
    now = datetime.datetime.now()
    prefix = f"{args.stage.upper()}_{args.variant}"
    run_name = now.strftime(f"{prefix}_%Y-%m-%d-%H-%M-%S")
    if args.split_name:
        run_name = f"{prefix}_{args.split_name}_{now.strftime('%Y-%m-%d-%H-%M-%S')}"

    output_subdir = {
        ("hpo", "kpdp"): "max_epoch_runs_new_kpdp/optuna_trials",
        ("hpo", "only"): "max_epoch_runs_new_only/optuna_trials",
        ("hpo", "pplus"): "max_epoch_runs_new_pplus/optuna_trials",
        ("final", "kpdp"): "max_epoch_runs_new_kpdp",
        ("final", "only"): "max_epoch_runs_new_only",
        ("final", "pplus"): "max_epoch_runs_new_pplus",
    }[(args.stage, args.variant)]

    run_path = PRESAGE_PATH / output_subdir / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    # ---- Build training config ----
    training_config = get_updated_config_file(
        args.ds_config_file, args.defaults_config, args.singles_config
    )

    # Inject pseudobulk mode settings (dataset paths only — embeddings injected per-fold)
    training_config = inject_pseudobulk_base_config(training_config, metadata)

    print(f"\nStage: {args.stage} | Variant: {args.variant} | Scheduler: {args.use_scheduler}")
    print(f"Dataset: {dataset_path}")
    print(f"Splits: {splits_path}")
    print(f"Output: {run_path}")

    # ---- Search space ----
    if args.hpo_search_space:
        search_space = load_search_space(args.hpo_search_space)
        print(f"Loaded search space from {args.hpo_search_space}")
    else:
        search_space = DEFAULT_SEARCH_SPACE
        print("Using DEFAULT search space")

    # ---- Save run metadata ----
    run_metadata = {
        "run_name": run_name,
        "stage": args.stage,
        "variant": args.variant,
        "use_scheduler": args.use_scheduler,
        "prepared_dir": args.prepared_dir,
        "prepare_metadata": metadata,
        "base_training_config": deepcopy(training_config),
        "search_space": search_space,
        "hpo_num_folds": args.hpo_num_folds,
    }
    with open(run_path / f"{run_name}_run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2, default=str)

    # ---- Dispatch ----
    if args.stage == "hpo":
        run_hpo(args, training_config, dataset_path, splits_path, run_name, run_path, search_space, metadata)
    else:
        run_final(args, training_config, dataset_path, splits_path, run_name, run_path, search_space, metadata)

    print("\nDone.")


if __name__ == "__main__":
    main()
