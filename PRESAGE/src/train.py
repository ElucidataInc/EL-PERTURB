import datetime
import os
import pickle as pkl
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

#from datamodule import  ReplogleDataModule
from presage_datamodule import ReploglePRESAGEDataModule
from model_harness import ModelHarness
from presage import PRESAGE

#from evaluator import Evaluator


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_config(config):
    if "." not in next(iter(config)):
        return config
    parsed = {"model": {}, "data": {}, "training": {}, "hyperparameter_tune": {}}
    for key, value in config.items():
        group, param = key.split(".")
        parsed[group][param] = value
    return parsed


def get_predictions(trainer, lightning_module, dataloader, var_names):
    predictions = trainer.predict(lightning_module, dataloader)

    keys, values = zip(*predictions)
    keys = np.concatenate(keys)
    values = np.concatenate(values)
    #embs = np.concatenate(embs)
    
    avg_mat = (
        pd.DataFrame(
            data=values, index=pd.Index(keys, name="perturbation"), columns=var_names
        )
        .groupby("perturbation")
        .mean()
    )
    return avg_mat


def get_attention(lightning_module, var_names):
    if hasattr(lightning_module, "attention_weights"):
        all_attention = []
        all_mapped_embeddings = []
        for batch in range(len(lightning_module.all_locs_gene)):
            attn_batch = lightning_module.attention_weights[batch].cpu().numpy()
            perts = var_names[lightning_module.all_locs_gene[batch]]

            attn_batch = pd.DataFrame(attn_batch, index=perts)
            all_attention.append(attn_batch)
            emb = lightning_module.transformed_embs[batch]
            all_mapped_embeddings.append(emb)

        all_attention = pd.concat(all_attention)
        pathway_weight_vec = lightning_module.pathway_weight_vector.cpu().numpy()
        all_mapped_embeddings = np.concatenate(all_mapped_embeddings, axis=0)

        return all_attention, pathway_weight_vec, all_mapped_embeddings

    else:

        return None


def get_embedding(lightning_module, var_names):
    all_emb = []
    all_coef = []
    all_perturbation = []
    for batch in range(len(lightning_module.all_embh)):
        emb = lightning_module.all_embh[batch]
        coef = lightning_module.all_coef[batch]
        locs_gene = lightning_module.all_locs_gene[batch]

        ind = lightning_module.all_locs_ind[batch]

        uniq_ind = np.unique(ind)
        emb_out = np.zeros((uniq_ind.shape[0], emb.shape[1]))
        coefs_out = np.zeros((uniq_ind.shape[0], 2))
        perturbation_out = []

        last_ind = -1
        perturbation = []
        for i, j in enumerate(ind):
            if j == last_ind:  # second of combo
                current_ind = 1
            else:  # single or new combo
                current_ind = 0
                if len(perturbation) != 0:
                    perturbation_out.append("_".join(perturbation))
                perturbation = []

            emb_out[j, :] = emb[j, :]
            coefs_out[j, current_ind] = coef[i]
            perturbation.append(var_names[locs_gene[i]])

            last_ind = j
        perturbation_out.append("_".join(perturbation))

        all_emb.append(emb_out)
        all_coef.append(coefs_out)
        all_perturbation.append(perturbation_out)

    all_emb = np.concatenate(all_emb, axis=0)
    all_coef = np.concatenate(all_coef, axis=0)
    all_perturbation = np.concatenate(all_perturbation, axis=0)

    all_emb = pd.DataFrame(all_emb, index=all_perturbation)
    all_coef = pd.DataFrame(all_coef, index=all_perturbation)

    return all_emb, all_coef


def train(config):
    config = parse_config(config)

    set_seed(config["training"].pop("seed", None))

    offline = config["training"].pop("offline", False)
    do_test_eval = config["training"].pop("eval_test", True)

    # initialize data module
    DataClass = None
    dataset = config["data"]["dataset"]
    # pop some settings that aren't used by datamodule constructor
    source = config["data"].pop("source")
    latent_dim = config["data"].pop("latent_dim")
    if latent_dim in ("None", None):
        latent_dim = None
    else:
        latent_dim = int(latent_dim)

    seed = config["data"].pop("seed")
    datamodule = ReploglePRESAGEDataModule.from_config(config["data"])
    datamodule.do_test_eval = do_test_eval

    # specify which seed to use for train/test splits
    if hasattr(datamodule, "set_seed"):
        datamodule.set_seed(seed)

    # return popped settings to config for logging
    config["data"]["source"] = source
    

    # we run prepare_data and setup ahead of time to get
    # the attributes needed for model initialization

    datamodule.prepare_data()

    datamodule.setup("fit")

    print("datamodule setup complete.")

    if latent_dim is not None:
        latent_dim = min(latent_dim, datamodule.train_dataset.X.shape[0] - 1)
    config["data"]["latent_dim"] = latent_dim
    config["model"]["pca_dim"] = latent_dim


    # initialize model
    model_config = config["model"]
    model_config["source"] = source
    model_config["dataset"] = dataset

    module = PRESAGE(
        model_config,
        datamodule,
        datamodule.pert_covariates.shape[1],
        datamodule.n_genes,
        # latent_dim or datamodule.n_genes,
    )

    if hasattr(module, "custom_init"):
        module.custom_init()

    lightning_module = ModelHarness(
        module,
        datamodule,
        model_config,
    )

    print("model initialization complete.")

    # run trainer
    logger = pl.loggers.WandbLogger(
        project="perturbation_prediction_rlv1",
        entity="in-silico-perturbations",
        job_type="dev",
        config=config,
        log_model=False,
        offline=offline,
        group=config["model"].get("group"),
        name=config["model"].get("name"),
    )
    predictions_file = config["training"].pop("predictions_file", None)
    embedding_pref = config["training"].pop("embedding_file", None)
    attention_file = config["training"].pop("attention_file", None)

    if predictions_file == "None":
        predictions_file = None

    # early_stop_callback = EarlyStopping(monitor="avg_cossim_top200_de", min_delta=0.01, patience=3, verbose=True, mode="max")
    early_stop_callback = EarlyStopping(
        # monitor="norm_normalized_mse_top200_de",
        # monitor="pn_mse_top20_unionde",
        monitor="val_loss",
        min_delta=1e-6,
        patience=10,
        verbose=True,
        mode="min",
    )

    # Get current date and time
    now = datetime.datetime.now()

    # Format the date and time
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./saved_models",
        filename=f"my_model-{dataset}-{seed.split('/')[-1].split('.json')[0]}-{now_str}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        mode="min",
    )
    torch.autograd.set_detect_anomaly(True)
    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=3,
        num_sanity_val_steps=10,
        callbacks=[early_stop_callback, checkpoint_callback],
        reload_dataloaders_every_n_epochs=1,
        **config["training"],
        gradient_clip_val=0.1,
    )
    trainer.fit(lightning_module, datamodule=datamodule)
    # lightning_module is the pytorch lighting, datamodule from datamodule.py
    # Get the best model path
    best_model_path = checkpoint_callback.best_model_path

    datamodule.setup("test")
    datamodule._data_setup = False

    checkpoint = torch.load(best_model_path)
    lightning_module.load_state_dict(checkpoint["state_dict"])
    os.remove(best_model_path)

    # log final eval metrics
    trainer.test(lightning_module, datamodule=datamodule)

    # if predictions_file is provided, save test set predictions
    combo_datasets = ["wessels_2023", "replogle_2020", "norman"]
    if np.isin(dataset, combo_datasets):
        if predictions_file:
            predictions = []
            dataloaders = datamodule.test_dataloader()
            if not isinstance(dataloaders, list):
                dataloaders = [dataloaders]

            for i, test_set in enumerate(datamodule.test_set_keys):
                avg_predictions = get_predictions(
                    trainer, lightning_module, dataloaders[i], datamodule.var_names
                )
                # print(avg_predictions)
                avg_predictions = avg_predictions.loc[
                    :, datamodule.train_dataset.adata.var.measured_gene
                ]

                predictions.append(
                    avg_predictions.set_index(
                        pd.Index([test_set] * len(avg_predictions), name="test_set"),
                        append=True,
                    )
                )
            predictions = pd.concat(predictions)
            predictions.loc[("control", "control")] = datamodule.pseudobulk.loc[
                "control"
            ]
            predictions.to_csv(predictions_file)
    else:
        dataloader = datamodule.test_dataloader()
        avg_predictions = get_predictions(
            trainer, lightning_module, dataloader, datamodule.var_names
        )
        avg_predictions = avg_predictions.loc[
            :, datamodule.train_dataset.adata.var.measured_gene
        ]
        avg_predictions.to_csv(predictions_file)

    attention = get_attention(lightning_module, datamodule.var_names)
    if attention is not None:
        attention, weight_vec, mapped_embeddings = attention
        if attention_file is not None:
            with open(f"{attention_file}.attention.pkl", "wb") as f_out:
                pkl.dump(attention, f_out)
            with open(f"{attention_file}.weight_vec.pkl", "wb") as f_out:
                pkl.dump(weight_vec, f_out)
            with open(f"{attention_file}.mapped_embeddings.pkl", "wb") as f_out:
                pkl.dump(mapped_embeddings, f_out)

    if hasattr(module, "pool"):
        if hasattr(module.pool.pool, "kg_weights"):
            kg_names = pd.read_csv(
                model_config["pathway_files"], header=None
            ).values.ravel()
            kg_names = [path.split("/")[-1].rsplit(".", 1)[0] for path in kg_names]
            kg_weight = pd.DataFrame(
                module.pool.pool.kg_weights.cpu().numpy().ravel(), index=kg_names
            )

            with open(f"{embedding_pref}.kg_weights.pkl", "wb") as f_out:
                pkl.dump(kg_weight, f_out)

        if hasattr(module.pool.pool, "kg_grouplasso_weights"):
            kg_names = pd.read_csv(
                model_config["pathway_files"], header=None
            ).values.ravel()
            kg_names = [path.split("/")[-1].rsplit(".", 1)[0] for path in kg_names]
            kg_weight = pd.DataFrame(
                module.pool.pool.kg_weights.cpu().numpy().ravel(), index=kg_names
            )

            with open(f"{embedding_pref}.kg_grouplasso_weights", "wb") as f_out:
                pkl.dump(kg_weight, f_out)

    """
    if embedding_pref:
        if hasattr(lightning_module, "all_embh"):
            merged_embedding, merged_coefficients = get_embedding(
                lightning_module, datamodule.var_names
            )

            with open(f"{embedding_pref}.embeddings.pkl", "wb") as f_out:
                pkl.dump(merged_embedding, f_out)
            with open(f"{embedding_pref}.coefficients.pkl", "wb") as f_out:
                pkl.dump(merged_coefficients, f_out)
    """
    if hasattr(lightning_module, "pool"):
        if hasattr(lightning_module.pool, "KG_weights"):
            with open(f"{embedding_pref}.KG_weights.pkl", "wb") as f_out:
                pkl.dump(lightning_module.pool.KG_weights, f_out)


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        # default="./config/mlp_baseline.yml",
        # default="./config/deepset_baseline.yml",
        default="./config/gears_adata.yml",
        help="Path to YAML config file",
    )
    args, unknown = parser.parse_known_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)
