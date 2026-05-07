import argparse

import wandb

from train import str2bool, train

import numpy as np
import json

parser = argparse.ArgumentParser()


def get_config_from_file(file):
    if file is None:
        return {}
    with open(file, "r") as f:
        config = json.load(f)
    return config


parser = argparse.ArgumentParser()

# Add a new argument for config file 
parser.add_argument("--config", type=str, help="Path to configuration file.")
parser.add_argument(
    "--data_config",
    type=str,
    help="Path to dataset-specific configuration file.",
    default=None,
)

parser.add_argument(
    "--model.class", type=str, default="PRESAGE"
)
# parser.add_argument("--model.dropout", type=float, default=0)


parser.add_argument("--model.lr", type=float, default=2.4e-3)
parser.add_argument("--model.weight_decay", type=float, default=2.17e-13)
parser.add_argument("--model.momentum", type=float, default=0.9)
parser.add_argument("--model.optimizer", type=str, default="Adam")

parser.add_argument(
    "--data.noisy_pseudobulk", type=str2bool, nargs="?", const=True, default=False
)

parser.add_argument("--model.n_nmf_embedding", type=int, default=128)
parser.add_argument("--model.node2vec_walk_length", type=int, default=100)
parser.add_argument("--model.node2vec_context_size", type=int, default=5)
parser.add_argument("--model.node2vec_walks_per_node", type=int, default=3)
parser.add_argument("--model.node2vec_num_negative_samples", type=int, default=3)
parser.add_argument("--model.node2vec_p", type=float, default=1.0)
parser.add_argument("--model.node2vec_q", type=float, default=2.0)
parser.add_argument("--model.node2vec_batchsize", type=int, default=32)

parser.add_argument(
    "--model.pathway_files",
    type=str,
    default="./sample_files/prior_files/sample.knowledge_experimental.txt",
)

parser.add_argument("--model.embedding_files", type=str, default="None")

parser.add_argument("--model.dim_red_alg", type=str, default="Node2Vec")

parser.add_argument("--model.cosine_loss_scale", type=float, default=1.0)
parser.add_argument("--model.contrastive_loss_scale", type=float, default=0)
parser.add_argument("--model.contrastive_nneigh", type=int, default=5)
parser.add_argument("--model.contrastive_neigh_metric", type=str, default="minkowski")

parser.add_argument("--model.univariate_cosine_loss_scale", type=float, default=0.0)
parser.add_argument("--model.mse_loss_scale", type=float, default=1.0)
parser.add_argument("--model.vector_norm_loss_scale", type=float, default=1.0)


parser.add_argument("--model.added_singles_loss_scale", type=float, default=1.0)


parser.add_argument(
    "--model.nonlinear_offset_sparsity_loss_scale", type=float, default=0.001
)

parser.add_argument("--model.item_hidden_size", type=int, default=512)
parser.add_argument("--model.item_nlayers", type=int, default=0)
parser.add_argument("--model.pathway_item_hidden_size", type=int, default=128)
parser.add_argument("--model.pathway_item_nlayers", type=int, default=2)
parser.add_argument("--model.pathway_pool_type", type=str, default="sum")
parser.add_argument("--model.pathway_weight_type", type=str, default="gat")
parser.add_argument("--model.pool_nlayers", type=int, default=2)
parser.add_argument("--model.softmax_temperature", type=float, default=0.15)
parser.add_argument("--model.gat_weight", type=float, default=0.85)

parser.add_argument("--model.linear_coefficient_hidden_size", type=int, default=32)
parser.add_argument("--model.linear_coefficient_nlayers", type=int, default=3)
parser.add_argument(
    "--model.linear_coefficient_features",
    type=str,
    choices=["gex", "emb", "both"],
    default="both",
)


parser.add_argument(
    "--model.batch_norm",
    type=str2bool,
    nargs="?",
    const=True,
    default=True,
)

parser.add_argument(
    "--model.pathway_pma_layer_norm",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
)


parser.add_argument("--model.set_hidden_size", type=int, default=32)
parser.add_argument("--model.set_nlayers", type=int, default=2)
parser.add_argument(
    "--model.pca_trainable_decoder", type=str2bool, nargs="?", const=True, default=False
)
parser.add_argument("--model.pca_loss_scale", type=float, default=1.0)
parser.add_argument(
    "--training.offline", type=str2bool, nargs="?", const=True, default=False
)
parser.add_argument(
    "--training.eval_test", type=str2bool, nargs="?", const=True, default=True
)


parser.add_argument(
    "--model.learnable_gene_embedding",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
)

parser.add_argument(
    "--hyperparameter_tune.sweep", type=str2bool, nargs="?", const=True, default=False
)

# parser.add_argument("--data.dataset", type=str, default="norman")
parser.add_argument("--data.dataset", type=str, default="None")

parser.add_argument("--data.seed", type=str, default=None)
parser.add_argument("--data.latent_dim", type=str, default="None")

parser.add_argument("--model.n_neigh_prune", type=str, default="5")

# quantile for cosine to keep
parser.add_argument("--model.min_cos", type=float, default=-1.0)

parser.add_argument("--model.min_genes_per_kg", type=int, default=10)

parser.add_argument(
    "--model.input_preparation", type=str, default="prep_gene_embeddings"
)

parser.add_argument(
    "--data.source",
    type=str,
    choices=["scperturb", "gears"],
    default="gears",
)

parser.add_argument(
    "--model.gex_coexpression_style", type=str, default="coexpression"
)  # other option SCING
parser.add_argument("--training.predictions_file", type=str, default="predictions.csv")
parser.add_argument("--training.embedding_file", type=str, default=None)
parser.add_argument("--training.attention_file", type=str, default=None)
parser.add_argument("--model.harness", type=str, default="ModelHarness")
parser.add_argument(
    "--data.use_pseudobulk", type=str2bool, nargs="?", const=True, default=True
)
parser.add_argument(
    "--data.preprocessing_zscore", type=str2bool, nargs="?", const=True, default=False
)
parser.add_argument("--data.batch_size", type=int, default=16)

parser.add_argument("--training.max_epochs", type=int, default=10000)

parser.add_argument(
    "--training.precision",
    type=str,
    choices=["16-mixed", "16", "32"],
    default="32",
)

parser.add_argument("--hyperparameter_tune.tag", type=str, default="")

parser.add_argument("--training.devices", type=int, default=1)
parser.add_argument("--training.seed", type=int, default=42)

# parser.add_argument("--model.attention_hidden_size", type=int, default=64)
# parser.add_argument("--model.attention_nlayers", type=int, default=2)

args = vars(parser.parse_args())

# differentiate between arguments passed through command line from defaults
cmd_line_args = {k: v for k, v in args.items() if v != parser.get_default(k)}


def get_best_hyperparameters(sweep_id, metric_name):
    api = wandb.Api()
    sweep = api.sweep(f"in-silico-perturbations/{sweep_id}")
    runs = sweep.runs
    best_run = max(runs, key=lambda run: run.summary.get(metric_name, float("-inf")))
    return best_run.config


# order of priority for arguments is: command line > data_config > config > defaults
# Load configuration from the file if provided
if args["config"] is not None or args["data_config"] is not None:

    config = get_config_from_file(args["config"]) if "config" in args else {}
    data_config = (
        get_config_from_file(args["data_config"]) if "data_config" in args else {}
    )

    # Merge dictionaries with dataset configuration overwriting the general one in case of conflicts
    config.update(data_config)
    # print(config)

    new_config = {}
    for key, value in config.items():
        if value is not None and key not in {"config", "data_config"}:
            new_config[key.replace("_", ".", 1)] = value
    config = new_config

    for key, value in args.items():
        if (
            key not in config
            and value is not None
            and key not in {"config", "data_config"}
        ):
            config[key] = value
    print(config)
    # overwrite with command line arguments
    for key, value in cmd_line_args.items():
        if np.isin(key, ["config", "data_config"], invert=True):
            print(key, value)
            config[key] = value

else:
    config = {}
    for key, value in args.items():
        if (
            key not in config
            and value is not None
            and key not in {"config", "data_config"}
        ):
            config[key] = value
args = config


if __name__ == "__main__":
    if args["hyperparameter_tune.sweep"]:
        single_datasets = {
            "replogle_k562_essential": {
                "seed": "replogle_k562_essential_random_splits/seed_0.json",
                "model.lr": [1.0e-3],
                "model.weight_decay": [4.0e-13],
            },
            "replogle_k562_essential_unfiltered": {
                "seed": "replogle_k562_essential_unfiltered_random_splits/seed_0.json",
                "model.lr": [0.0024353],
                "model.weight_decay": [1.0e-18],
            },
            "replogle_rpe1_essential": {
                "seed": "replogle_rpe1_essential_random_splits/seed_0.json",
                "model.lr": [0.0016987],
                "model.weight_decay": [8.0e-20],
            },
            "nadig_hepg2": {
                "seed": "nadig_hepg2_random_splits/seed_0.json",
                "model.lr": [0.0013972],
                "model.weight_decay": [1.0e-17],
            },
            "nadig_jurkat": {
                "seed": "nadig_jurkat_random_splits/seed_0.json",
                "model.lr": [0.0029912],
                "model.weight_decay": [1.0e-11],
            },
            "replogle_rpe1_essential_unfiltered": {
                "seed": "replogle_rpe1_essential_unfiltered_random_splits/seed_0.json",
                "model.lr": [0.0016987],
                "model.weight_decay": [1.0e-19],
            },
            "tf_atlas": {
                "seed": "tf_atlas_random_splits/seed_0.json",
            },
            "tf_atlas_filtered": {
                "seed": "tf_atlas_filtered_random_splits/seed_0.json",
            },
            "replogle_k562_gw": {
                "seed": "replogle_k562_gw_random_splits/seed_0.json",
                "model.lr": [2.1e-3],
                "model.weight_decay": [5.0e-19],
            },
            "adamson": {
                "seed": "adamson_random_splits/seed_0.json",
            },
            "dixit": {
                "seed": "dixit_random_splits/seed_0.json",
            },
        }

        combo_datasets = {
            "replogle_2020": {
                "seed": "replogle_2020_random_splits/seed_0.json",
                "model.item_hidden_size": [64, 256],
                "model.item_nlayers": [4],
                "model.linear_coefficient_hidden_size": [32, 64, 128, 256],
                "model.linear_coefficient_nlayers": [1, 3, 5, 7, 9, 12],
                "model.pathway_item_hidden_size": [128],
                "model.pathway_item_nlayers": [1, 2, 3],
                "model.set_hidden_size": [32],
                "model.set_nlayers": [1, 3, 5, 7, 9, 12],
                "model.transformer_layer": [1, 2, 4, 6, 8],
            },
            "wessels_2023": {
                "seed": "wessels_2023_random_splits/seed_0.json",
                "model.item_hidden_size": [
                    64,
                    256,
                ],  # 256 best hyperparamters commented here. I want to check if we can make them more uniform
                "model.item_nlayers": [0, 4],  # 0
                "model.linear_coefficient_hidden_size": [32, 128],  # 32
                "model.linear_coefficient_nlayers": [9, 12],  # 9
                "model.pathway_item_hidden_size": [128, 32],  # 128
                "model.pathway_item_nlayers": [2, 1],  # 2
                "model.set_hidden_size": [32],  # 32
                "model.set_nlayers": [4, 1],  # 4
                "model.transformer_layer": [1, 2, 4, 6, 8],
            },
            "norman": {
                "seed": "norman_gears_pub/seed_000.json",
                "model.item_hidden_size": [64, 256],  # 64
                "model.item_nlayers": [0, 4],  # 4
                "model.linear_coefficient_hidden_size": [32, 128],  # 128
                "model.linear_coefficient_nlayers": [9, 12],  # 12
                "model.pathway_item_hidden_size": [32, 128],  # 32
                "model.pathway_item_nlayers": [1, 2],  # 1
                "model.set_hidden_size": [32],  # 32
                "model.set_nlayers": [1, 4],  # 1
                "model.transformer_layer": [1, 2, 4, 6, 8],
            },
        }

        if args["data.dataset"] == "None":
            seed = args["data.seed"]
            print(seed)
            args["data.dataset"] = "_".join(seed.split("/")[0].split("_")[:-2])
        else:
            if args["data.dataset"] in single_datasets:
                seed = single_datasets[args["data.dataset"]]["seed"]
            else:
                seed = combo_datasets[args["data.dataset"]]["seed"]

        if args["data.dataset"] in single_datasets:
            sweep_config = {
                "metric": {
                    # "name": "avg_cossim_top200_de",
                    # "name":"test_avg_normalized_mse_top200_de",
                    # "name": "norm_normalized_mse_top200_de",
                    "name":"test_loss",
                    # "name": "test_pn_mse_top20_de",
                    #"name": "test_perturbations_with_effect_pn_mse_top20_unionde",
                    # "name":"test_perturbations_with_effect_avg_cossim_top20_unionde",
                    "goal": "minimize",
                },
                #"method": "bayes",
                "method": "random",
                "parameters": {
                    "training.max_epochs": {"values": [1000]},
                    #"data.batch_size": {"values": [256]},
                    "data.batch_size": {"values": [256]},
                    "model.class": {
                        "values": [  # "WeightedDeepSetOnPathwaysContrastive",
                            "WeightedDeepSetOnPathwaysContrastive_Refactored",
                            # "WeightedSetOnPathwaysContrastive_Refactored",
                        ]
                    },
                    #"model.n_nmf_embedding": {"values": [32,64,128]},
                    "model.n_nmf_embedding": {"values": [128]},
                    "model.node2vec_walk_length": {"values": [100]},
                    "model.node2vec_context_size": {"values": [5]},
                    "model.node2vec_walks_per_node": {"values": [3]},
                    "model.node2vec_num_negative_samples": {"values": [3]},
                    "model.node2vec_p": {"values": [1.0]},
                    "model.node2vec_q": {"values": [2.0]},
                    "model.node2vec_batchsize": {"values": [32]},
                    "model.n_neigh_prune": {"values": [5]},
                    "model.min_genes_per_kg": {"values": [10]},
                    "model.gex_coexpression_style": {"values": ["coexpression"]},
                    # "model.min_cos":{"values":single_datasets[args['data.dataset']]['min_cos']},
                    # "model.min_cos":{"min":0.1,"max":0.8},
                    "model.min_cos": {"values": [-1.0]},
                    "model.pathway_item_hidden_size": {"values": [128]},
                    "model.pathway_item_nlayers": {"values": [2]},
                    # "model.pathway_pool_type":{"values":["mean","max",'PMA']},
                    "model.pathway_pool_type": {"values": ["sum"]},
                    # "model.pathway_weight_type":{"values":["linear_map","vector","gat", "grouplasso", "grouplassodynamic", "grouplasso_top_k", "transformer"]},#,"linear_map"]}, # alternative is vector, and attention
                    # "model.pathway_weight_type":{"values":["linear_map","vector","gat", "grouplasso","transformer",]},#,"linear_map"]},  "grouplassodynamic", "grouplasso_top_k",
                    "model.pathway_weight_type": {
                        "values": ["gat"]
                    },  # ,"linear_map"]},  "grouplassodynamic", "grouplasso_top_k",
                    # "model.pathway_weight_type":{"values":["vector"]},
                    # "model.pathway_weight_type": {"values":single_datasets[args['data.dataset']]['model.pathway_weight_type']},
                    "model.pool_nlayers": {"values": [2]},
                    #"model.pool_nlayers": {"values": [2]},
                    # "model.pathway_weight_type": {
                    #    "values": ["grouplasso"]
                    # },  # ,"linear_map"]}, # alternative is vector, and attention
                    # "model.pathway_weight_type":{"values":["attention"]},#,"linear_map"]}, # alternative is vector, and attention
                    # "model.softmax_temperature": {
                    #    "max": 1.0,
                    #    "min": 1e-2,
                    #    "distribution": "log_uniform_values",
                    # },
                    # "model.softmax_temperature":{"values":single_datasets[args['data.dataset']]['model.softmax_temperature']},
                    "model.softmax_temperature": {"values": [0.1]},
                    #"model.softmax_temperature": {
                    #    "max": 5e-1,
                    #    "min": 1e-3,
                    #    "distribution": "log_uniform_values",
                    #},
                    #"model.gat_weight": {"values": [0.85,0.9,0.95]},
                    "model.gat_weight": {"values": [0.85]},
                    #"model.gat_weight":{
                    #   "max":1.0,
                    #   "min":5e-1,
                        #"distribution":"log_uniform_values",
                    #},
                    # "model.softmax_temperature": {"values":single_datasets[args['data.dataset']]['model.softmax_temperature']},
                    "model.pathway_pma_layer_norm": {"values": [False]},
                    "model.batch_norm": {"values": [True]},
                    "model.learnable_gene_embedding": {"values": [False]},
                    # "model.learnable_perturbation_and_tf": {"values": [False]},
                    # "model.perturb_cluster_loss_scale":{"values":[0.0]},
                    # "model.perturb_cluster_loss_decay":{"values":[1.0]},
                    # "data.nperturb_clusters":{"values":["None"]},
                    "model.dim_red_alg": {"values": ["Node2Vec"]},
                    "model.pathway_files": {
                        "values": [ 
                            "./sample_files/prior_files/sample.knowledge_experimental.txt",
                        ]
                    },
                    "hyperparameter_tune.tag": {
                        "values": [  # f"{args['data.dataset']}_tune_mincos_1",
                            
                            "tune_presage"
                        ]
                    },
                    "model.embedding_files": {"values": ["None"]},
                    #"model.lr": {"values": [1.2e-3]},
                    "model.lr": {
                        "min": 5e-4,
                        "max": 1.5e-3,
                        "distribution": "log_uniform_values",
                    },
                    #"model.lr":{"values":single_datasets[args['data.dataset']]['model.lr']},
                    #"model.weight_decay": {
                    #    "min": 1e-20,
                    #    "max": 1e-10,
                    #    "distribution": "log_uniform_values",
                    #},
                    #"model.weight_decay":{"values":[10**-i for i in range(10, 20)]},
                    #"model.weight_decay":{"values":single_datasets[args['data.dataset']]['model.weight_decay']},
                    "model.weight_decay":{"values":[1e-15]},
                    "data.use_pseudobulk": {"values": [True]},
                    "data.noisy_pseudobulk": {"values": [False]},
                    "data.preprocessing_zscore": {"values": [False]},
                    "model.contrastive_loss_scale": {"values": [0]},
                    "model.contrastive_nneigh": {"values": [5]},
                    "model.contrastive_neigh_metric": {"values": ["minkowski"]},
                    # "model.univariate_cosine_loss_scale": {"values": [1]},
                    # "model.mse_loss_scale": {"values": [0]},
                    # "model.vector_norm_loss_scale": {"values": [1e-6,1e-3,1e-2,1e-1,1,10]},
                    "data.source": {"values": ["gears"]},
                    "data.seed": {
                        "values": [  
                            f"./splits/{seed}",
                        ]
                    },
                    "data.dataset": {"values": [args["data.dataset"]]},
                    # "model.item_hidden_size": {"values": [64]},
                    # "model.item_hidden_size":{"values":single_datasets[args['data.dataset']]['model.item_hidden_size']},
                    "model.item_hidden_size": {"values": [512]},
                    # "model.item_nlayers":{"values":single_datasets[args['data.dataset']]['model.item_nlayers']},
                    "model.item_nlayers": {"values": [0]},
                    # "model.item_nlayers": {"min":0,"max":1},
                    # "model.item_nlayers": {"values": [0]},
                    # "model.do_regression":{"values":[True]},
                    # "data.latent_dim": {"values": ["50"]},
                    "data.latent_dim": {"values": ["None"]},
                    "training.eval_test":{"values":[False]},
                    "training.predictions_file": {"values": ["None"]},
                },
                "program": "src/train_presage.py",
            }

        else:
            print(f"{args['data.dataset']} not in sweep config possibilities")
            quit()

        entity = ""
        project = ""
        sweep_id = wandb.sweep(sweep=sweep_config, entity=entity, project=project)
        wandb.agent(sweep_id, count=500, entity=entity, project=project)

        # Retrieve the best hyperparameters
        # best_hyperparameters = get_best_hyperparameters(sweep_id, "avg_normalized_mse_top200_de")
        # print("Best Hyperparameters: ", best_hyperparameters)

    else:
        from pprint import pprint

        pprint(args)
        train(args)
