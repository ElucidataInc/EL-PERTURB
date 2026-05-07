import torch
from torch import nn
import numpy as np
import pandas as pd
import copy
from sklearn.decomposition import PCA

from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn.models import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
#from .baseline import MLP, MLPBaseline
#from .pca import PCAWithTrainableDecoder
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import scanpy as sc

from collections import Counter
from torch_geometric.nn.conv import GATConv
from scipy.stats import pearsonr

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping


from scipy.spatial.distance import pdist, squareform

#from grn.buildGRNHelpers import grnBuilder

import torch.nn.functional as F
import math

# import torchsort

import os
import pickle as pkl


class PrepareInputs:
    def __init__(self, datamodule, config):
        self.datamodule = datamodule
        self.config = config

        self.input_preparation_functions = {
            "prep_gene_embeddings": prep_gene_embeddings,
        }

    def _prep_inputs(self):
        return self.input_preparation_functions[self.config["input_preparation"]](
            self.datamodule, self.config
        )

class MLP(nn.Module):
    def __init__(self, input_size, output_size, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        nlayers = config["nlayers"]
        l1_norm = config['l1norm']
        batch_norm = config["batch_norm"]
        if l1_norm:
            linear = SpaRedLinear
        else:
            linear = nn.Linear
        layers = []
        for hidden_size in [hidden_size] * nlayers:
            layers.append(linear(input_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            #layers.append(nn.ReLU())
            layers.append(nn.LeakyReLU())
            input_size = hidden_size
        layers.append(linear(hidden_size, output_size,bias=True))
        self.mlp = nn.Sequential(*layers)
        self.config = config

    def forward(self, x, cov=None):
        return self.mlp(x)#, None

    def compute_loss(self, pred, tgt,pred_clust=None,expr_clust=None):
        return F.mse_loss(pred, tgt)


class PRESAGE_ORIG(nn.Module):
    def __init__(self, config, datamodule, input_dimension, output_dimension):
        super(PRESAGE, self).__init__()

        # prepare variables
        self.batch_size = datamodule.batch_size
        self.config = config
        self.validation_step_outputs = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datamodule = datamodule
        self.genes = datamodule.train_dataset.adata.var.index.to_numpy().ravel()
        self.add_singles_loss_scale = config["added_singles_loss_scale"]
        self.n_nmf_embeddings = config["n_nmf_embedding"]
        self.num_celltypes = len(datamodule.train_dataset.cell_types)
        self.num_genes = output_dimension
        self.pca_dim = config["pca_dim"]
        

        # get gene embeddings
        self.gene_embeddings = PrepareInputs(datamodule, config)._prep_inputs()

        norms = np.linalg.norm(self.gene_embeddings, axis=1, keepdims=True)

        norms = np.median(norms, axis=(0, 1), keepdims=True)

        # fixing areas with norm of 0
        norms[norms == 0] = 1

        self.gene_embeddings = self.gene_embeddings / (norms)

        self.gene_embeddings = (
            torch.tensor(self.gene_embeddings).type(torch.float32).to(self._device)
        )

        self.mask = torch.sum(self.gene_embeddings, dim=1, keepdims=True) != 0

        if config["learnable_gene_embedding"]:
            self.learnable_embedding = nn.Embedding(
                self.gene_embeddings.shape[0], config["item_hidden_size"]
            )

        ngenes, _, n_pathways = self.gene_embeddings.shape

        self.activation = nn.LeakyReLU()
        self.temperature = config["softmax_temperature"]

        # Map from raw gene embeddings to aligned gene embeddings
        # pathway encoder
        config["num_heads"] = 4  # hidden_dim must be divisible by num_heads
        pathway_encoder_function = "MLP"  # MLP MOE MHA

        self.pathway_encoder = GeneEmbeddingTransformation(
            self.gene_embeddings, config, pathway_encoder_function
        )

        if self.pca_dim is not None:
            self.pca = PCAWithTrainableDecoder(
                self.pca_dim, trainable_decoder=True
            )  # TODO add trainable_decoder to config
            self.pca.fit(
                torch.tensor(self.datamodule.train_dataset.X)
                .type(torch.float32)
                .to(self._device)
            )

        # Pool type to perturbation latent space
        config["num_genes"] = self.pca_dim or self.num_genes  # self.num_genes
        self.pool = Pool(n_pathways, config)
        # self.pool.KG_weights = None

        # map from latent to output (logFC)
        config["num_celltypes"] = self.num_celltypes
        self.item_net = ItemNet("MLP", self.pca_dim or self.num_genes, config)

        # getting the location of the singles in the training data
        # this is so we can use them in the combo
        singles_in_training_binary = self.datamodule.train_dataset.indmtx[
            np.sum(self.datamodule.train_dataset.indmtx, axis=1) == 1
        ]

        if self.datamodule.X_train_pca is not None:
            singles_in_training = self.datamodule.X_train_pca[
                np.sum(self.datamodule.train_dataset.indmtx, axis=1) == 1
            ]
        else:
            singles_in_training = self.datamodule.train_dataset.X[
                np.sum(self.datamodule.train_dataset.indmtx, axis=1) == 1
            ]

        self.single_inds_in_training_data = {}
        locs = np.where(singles_in_training_binary)
        for i, g in zip(locs[0], locs[1]):
            self.single_inds_in_training_data[g] = torch.tensor(
                singles_in_training[i, :]
            ).to(self._device)

    def forward_to_emb(self, locs_gene, locs_combos):
        # get the gene embeddings for the perturbed gene
        emb = self.gene_embeddings[locs_gene, :, :]

        mask = self.mask[locs_gene, :, :]

        # maps KG specific embeddings to KG shared latent space
        # uses GeneEmbeddingTransformation class
        emb_h = self.pathway_encoder(emb)

        emb_h = self.activation(emb_h)
        emb_h_temp = emb_h

        self.emb_h = emb_h

        # pool to latent space uses the Pool cass
        emb_h_final = self.pool(
            emb_h, locs_combos, mask
        ) 

        self.emb_h_final = emb_h_final

        # pool is nested here
        if hasattr(self.pool.pool, "p_weight_vec"):
            self.pathway_weight_vector = self.pool.pool.p_weight_vec
        if hasattr(self.pool.pool, "attention_weights"):
            self.attention_weights = self.pool.pool.attention_weights

        return emb_h_temp, emb_h_final

    def emb_to_out(self, emb_h, locs_gene, locs_combos):

        # transformation to output dimensions uses ItemNet class
        out = self.item_net(emb_h)
        
        if self.pca_dim is not None:
            out = self.pca.inverse_transform(out)

        return out

    def forward(self, pert_inds, ctype_inds, cov=None, update_node_embeddings=True):
        self.pert_inds = pert_inds
        locs = torch.nonzero(pert_inds)
        # indices of perturbed genes
        locs_gene = locs[:, 1]
        # indices of perturbations within a batch
        locs_combos = locs[:, 0]

        locs_ctype = torch.nonzero(ctype_inds)[:, 1]
        
        self.locs_gene = locs_gene
        self.locs_combos = locs_combos
        self.locs_ctype = locs_ctype

        # logic for singles START
        # gene indices to latent space
        emb_h_temp, emb_h = self.forward_to_emb(locs_gene, locs_combos)

        # concatenate cell type indicator vector with emb_h
        emb_h_plus = torch.hstack((emb_h, ctype_inds))
        print(ctype_inds[0:5, :])

        # latent space to out dimension
        out = self.emb_to_out(emb_h_plus, locs_gene, locs_combos)

        return out, emb_h_temp, "None"

    def compute_loss(self, pred, expr, tensor, pred_clust=None, expr_clust=None):
        loss = torch.nn.functional.mse_loss(pred, expr)

        return loss

class PRESAGE(nn.Module):
    def __init__(self, config, datamodule, input_dimension, output_dimension):
        super(PRESAGE, self).__init__()

        self.batch_size = datamodule.batch_size
        self.config = config
        self.validation_step_outputs = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datamodule = datamodule
        self.genes = datamodule.train_dataset.adata.var.index.to_numpy().ravel()
        self.add_singles_loss_scale = config["added_singles_loss_scale"]
        self.n_nmf_embeddings = config["n_nmf_embedding"]
        self.num_celltypes = len(datamodule.train_dataset.cell_types)
        self.num_genes = output_dimension
        self.pca_dim = config["pca_dim"]
        self.prior_type = config["prior_type"]
        self.prior_by_group = config["prior_by_group"]

        group_col = config.get("prior_by_group", None)
        self.is_grouped = group_col is not None and group_col != "None"

        # get gene embeddings
        raw_embeddings = PrepareInputs(datamodule, config)._prep_inputs()

        if self.is_grouped:
            # raw_embeddings is {group_name: (n_genes, n_emb, n_sources)}
            # sort to get a deterministic ordering that matches ctype_inds columns
            self.group_names = list(datamodule.train_dataset.cell_types)
            missing = [g for g in self.group_names if g not in raw_embeddings]
            if missing:
                raise ValueError(
                    f"Groups {missing} found in cell_types but not in computed embeddings. "
                    f"Available groups: {list(raw_embeddings.keys())}"
                )
            normed = []
            masks = []
            for g in self.group_names:
                emb = raw_embeddings[g]
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms = np.median(norms, axis=(0, 1), keepdims=True)
                norms[norms == 0] = 1
                emb = emb / norms
                emb_t = torch.tensor(emb).type(torch.float32).to(self._device)
                normed.append(emb_t)
                masks.append(torch.sum(emb_t, dim=1, keepdims=True) != 0)

            # (n_groups, n_genes, n_emb, n_sources)
            self.gene_embeddings = torch.stack(normed, dim=0)
            self.mask = torch.stack(masks, dim=0)

            # all groups have same shape — use first for dimension inference
            reference_emb = normed[0]
        else:
            norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
            norms = np.median(norms, axis=(0, 1), keepdims=True)
            norms[norms == 0] = 1
            raw_embeddings = raw_embeddings / norms
            self.gene_embeddings = (
                torch.tensor(raw_embeddings).type(torch.float32).to(self._device)
            )
            self.mask = torch.sum(self.gene_embeddings, dim=1, keepdims=True) != 0
            reference_emb = self.gene_embeddings

        if config["learnable_gene_embedding"]:
            self.learnable_embedding = nn.Embedding(
                reference_emb.shape[0], config["item_hidden_size"]
            )

        ngenes, _, n_pathways = reference_emb.shape

        self.activation = nn.LeakyReLU()
        self.temperature = config["softmax_temperature"]

        config["num_heads"] = 4
        pathway_encoder_function = "MLP"

        # shared across groups — only needs shape info
        self.pathway_encoder = GeneEmbeddingTransformation(
            reference_emb, config, pathway_encoder_function
        )

        if self.pca_dim is not None:
            self.pca = PCAWithTrainableDecoder(self.pca_dim, trainable_decoder=True)
            self.pca.fit(
                torch.tensor(self.datamodule.train_dataset.X)
                .type(torch.float32)
                .to(self._device)
            )

        config["num_genes"] = self.pca_dim or self.num_genes
        self.pool = Pool(n_pathways, config)

        config["num_celltypes"] = self.num_celltypes
        self.item_net = ItemNet("MLP", self.pca_dim or self.num_genes, config)

        #update self.config
        self.config = config
        
        singles_in_training_binary = self.datamodule.train_dataset.indmtx[
            np.sum(self.datamodule.train_dataset.indmtx, axis=1) == 1
        ]

        if self.datamodule.X_train_pca is not None:
            singles_in_training = self.datamodule.X_train_pca[
                np.sum(self.datamodule.train_dataset.indmtx, axis=1) == 1
            ]
        else:
            singles_in_training = self.datamodule.train_dataset.X[
                np.sum(self.datamodule.train_dataset.indmtx, axis=1) == 1
            ]

        self.single_inds_in_training_data = {}
        locs = np.where(singles_in_training_binary)
        for i, g in zip(locs[0], locs[1]):
            self.single_inds_in_training_data[g] = torch.tensor(
                singles_in_training[i, :]
            ).to(self._device)

        
    def forward_to_emb(self, locs_gene, locs_combos, ctype_inds=None):
        if self.is_grouped and ctype_inds is not None:
            # ctype_inds is one-hot: (batch_size, n_celltypes)
            # convert to integer group index per sample
            group_idx_per_sample = torch.argmax(ctype_inds, dim=1)  # (batch_size,)

            # locs_combos tells us which sample each perturbation belongs to
            # so group_idx_per_sample[locs_combos] gives group index per perturbation
            group_per_pert = group_idx_per_sample[locs_combos]  # (n_perturbations,)

            # self.gene_embeddings: (n_groups, n_genes, n_emb, n_sources)
            # advanced indexing: for perturbation i, pick
            #   gene_embeddings[group_per_pert[i], locs_gene[i], :, :]
            emb = self.gene_embeddings[group_per_pert, locs_gene, :, :]
            mask = self.mask[group_per_pert, locs_gene, :, :]
        else:
            # ungrouped: gene_embeddings is (n_genes, n_emb, n_sources)
            emb = self.gene_embeddings[locs_gene, :, :]
            mask = self.mask[locs_gene, :, :]

        emb_h = self.pathway_encoder(emb)
        emb_h = self.activation(emb_h)
        emb_h_temp = emb_h

        self.emb_h = emb_h

        emb_h_final = self.pool(emb_h, locs_combos, mask)

        self.emb_h_final = emb_h_final

        if hasattr(self.pool.pool, "p_weight_vec"):
            self.pathway_weight_vector = self.pool.pool.p_weight_vec
        if hasattr(self.pool.pool, "attention_weights"):
            self.attention_weights = self.pool.pool.attention_weights

        return emb_h_temp, emb_h_final

    def emb_to_out(self, emb_h, locs_gene, locs_combos):
        out = self.item_net(emb_h)
        if self.pca_dim is not None:
            out = self.pca.inverse_transform(out)
        return out

    def forward(self, pert_inds, ctype_inds, cov=None, update_node_embeddings=True):
        self.pert_inds = pert_inds
        locs = torch.nonzero(pert_inds)
        locs_gene = locs[:, 1]
        locs_combos = locs[:, 0]

        locs_ctype = torch.nonzero(ctype_inds)[:, 1]

        self.locs_gene = locs_gene
        self.locs_combos = locs_combos
        self.locs_ctype = locs_ctype

        # gene embeddings are now group-aware via ctype_inds
        emb_h_temp, emb_h = self.forward_to_emb(
            locs_gene, locs_combos, ctype_inds=ctype_inds
        )

        # still concatenate ctype_inds so ItemNet can condition on cell type
        
        if self.config['num_celltypes'] > 1:
            emb_h_plus = torch.hstack((emb_h, ctype_inds))
        else:
            emb_h_plus = emb_h

        out = self.emb_to_out(emb_h_plus, locs_gene, locs_combos)

        return out, emb_h_temp, "None"

    def compute_loss(self, pred, expr, tensor, pred_clust=None, expr_clust=None):
        loss = torch.nn.functional.mse_loss(pred, expr)

        return loss


class ItemNet(nn.Module):
    def __init__(self, inet_type, out_dim, config):
        super(ItemNet, self).__init__()
        item_net_type = {"MLP": ItemMLP}

        self.item_net = item_net_type[inet_type](out_dim, config)

    def forward(self, tensor):
        return self.item_net(tensor)


class ItemMLP(nn.Module):
    def __init__(self, out_dim, config):
        super(ItemMLP, self).__init__()
        if config["item_nlayers"] == 0:
            if config["num_celltypes"] > 1:
                self.item_net = nn.Linear(config["item_hidden_size"] + config["num_celltypes"], out_dim)
            else:
                self.item_net = nn.Linear(config["item_hidden_size"], out_dim)
        else:
            if config["num_celltypes"] > 1:
                self.item_net = MLP(
                    input_size=config["item_hidden_size"] + config["num_celltypes"],
                    output_size=out_dim,
                    config=dict(
                        hidden_size=config["item_hidden_size"],
                        nlayers=config["item_nlayers"],
                        l1norm=False,
                        batch_norm=config["batch_norm"],
                    ),
                )
            else:
                self.item_net = MLP(
                    input_size=config["item_hidden_size"],
                    output_size=out_dim,
                    config=dict(
                        hidden_size=config["item_hidden_size"],
                        nlayers=config["item_nlayers"],
                        l1norm=False,
                        batch_norm=config["batch_norm"],
                    ),
                )
                

    def forward(self, tensor):
        return self.item_net(tensor)

class GeneEmbeddingTransformation(nn.Module):
    def __init__(self, gene_embeddings, config, transformation_type):
        super(GeneEmbeddingTransformation, self).__init__()

        # classes here must have init and forward defined
        transformations = {"MLP": PathwayMLP}

        ngenes, n_feature_in, n_pathways = gene_embeddings.shape

        self.transformation_obj = transformations[transformation_type](
            n_feature_in,
            config["item_hidden_size"],
            config["pathway_item_hidden_size"],
            config["pathway_item_nlayers"],
            config["batch_norm"],
            n_pathways,
            config,
        )

    def forward(self, tensor):
        return self.transformation_obj(tensor)


class PathwayMLP(nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_dim, n_layers, batch_norm, n_pathways, config
    ):
        super(PathwayMLP, self).__init__()

        # We will build a separate MLP for each pathway.
        self.pathway_mlps = nn.ModuleList()

        for _ in range(n_pathways):
            mlp_layers = []
            if n_layers == 0:  # if no hidden layers
                mlp_layers.append(nn.Linear(in_dim, out_dim))
                nn.init.xavier_uniform_(mlp_layers[-1].weight)
            elif n_layers == 1:  # if one hidden layer
                mlp_layers.append(nn.Linear(in_dim, hidden_dim))
                nn.init.xavier_uniform_(mlp_layers[-1].weight)
                mlp_layers.append(nn.LeakyReLU())
                mlp_layers.append(nn.Linear(hidden_dim, out_dim))
                nn.init.xavier_uniform_(mlp_layers[-1].weight)
            else:  # if more than one hidden layer
                mlp_layers.append(nn.Linear(in_dim, hidden_dim))
                nn.init.xavier_uniform_(mlp_layers[-1].weight)
                if batch_norm:
                    mlp_layers.append(nn.BatchNorm1d(hidden_dim))
                for _ in range(n_layers - 2):  # add hidden layers
                    mlp_layers.append(nn.LeakyReLU())
                    mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
                    nn.init.xavier_uniform_(mlp_layers[-1].weight)
                    if batch_norm:
                        mlp_layers.append(nn.BatchNorm1d(hidden_dim))
                mlp_layers.append(nn.LeakyReLU())
                mlp_layers.append(nn.Linear(hidden_dim, out_dim))  # output layer
                nn.init.xavier_uniform_(mlp_layers[-1].weight)

            mlp = nn.Sequential(*mlp_layers)
            self.pathway_mlps.append(mlp)

    def forward(self, pathway_tensor):
        output = []
        # Apply separate MLP to each pathway
        for i in range(pathway_tensor.shape[-1]):
            pathway_output = self.pathway_mlps[i](pathway_tensor[:, :, i])
            output.append(pathway_output.unsqueeze(-1))

        # Combine outputs from all pathways
        if len(output) > 1:
            output_tensor = torch.cat(output, dim=-1)
        else:
            output_tensor = output[0]

        return output_tensor


class Pool(nn.Module):
    def __init__(self, n_pathways, config):
        super(Pool, self).__init__()
        pool_type = {
            "mean": MeanPool,
            "vector": LearnableWeightPool,
            "gat": GATPool,
        }

        self.pool = pool_type[config["pathway_weight_type"]](
            n_pathways,
            config,
        )
        # print("pool type", config["pathway_weight_type"])
        if np.isin(
            config["pathway_weight_type"],
            ["grouplasso", "grouplassodynamic", "grouplasso_top_k"],
        ):
            self.grouplasso = True
            # print("grouplasso true")
        else:
            self.grouplasso = False

    def forward(self, tensor, locs_combos, mask):
        if self.grouplasso:
            res, _ = self.pool(tensor, locs_combos)
            return res
        else:
            # print("after pool dimension", self.pool(tensor).shape)
            return self.pool(tensor, mask)

class MeanPool(nn.Module):
    def __init__(self, n_pathways, config):
        super(MeanPool, self).__init__()

    def forward(self, tensor):

        return torch.mean(tensor, dim=-1)


class LearnableWeightPool(nn.Module):
    def __init__(self, n_pathways, config):
        super(LearnableWeightPool, self).__init__()
        eps = 1e-3
        self.pathway_weight_vector = nn.Parameter(torch.randn(1, n_pathways) * eps)
        self.temperature = config["softmax_temperature"]

    def forward(self, tensor):
        p_weights = torch.nn.functional.softmax(
            self.pathway_weight_vector / self.temperature, dim=-1
        ).unsqueeze(0)
        # print(torch.sort(p_weights))
        h = tensor * p_weights
        self.kg_weights = p_weights
        # print(self.kg_weights)

        return torch.sum(h, dim=-1)



class GATPool(nn.Module):
    def __init__(self, n_pathways, config):
        super(GATPool, self).__init__()
        self.nlayers = config["pool_nlayers"]
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # all KGs to a common vector
        # plus one so that we also add in the pooled weighted vector
        self.edge_index = torch.tensor(
            np.array([[i, n_pathways] for i in range(n_pathways)]).T
        ).to(self._device)

        eps = 1e-3
        
        self.gatconv = GATConv(
            config["item_hidden_size"], config["item_hidden_size"], 5, concat=False,
        )

        #self.gatconv_list = nn.ModuleList([
        #    GATv2Conv(
        #    config["item_hidden_size"], config["item_hidden_size"], 5, concat=False,
        #    ) for _ in range(self.nlayers)
        #])
        self.activation = nn.LeakyReLU()

        
        self.pathway_weight_vector = nn.Parameter(torch.randn(1, n_pathways) * eps)
        self.pooled_vec = nn.Parameter(torch.randn(1, config['item_hidden_size']) * eps)
        self.temperature = config["softmax_temperature"]
        self.gat_weight = config["gat_weight"]

    def forward(self, tensor, mask):
        # this tensor is needed for the GAT pool
        tensor = tensor * mask

        # we need to do a masked - weighted average here
        masked_weight_vec = self.pathway_weight_vector.unsqueeze(0) * mask

        p_weights = torch.nn.functional.softmax(
            masked_weight_vec / self.temperature, dim=-1
        ).unsqueeze(0)

        self.p_weight_vec = torch.nn.functional.softmax(
            self.pathway_weight_vector / self.temperature, dim=-1
        )

        # skip connection. scaling is to change the average so we don't upscale things with more KGs
        h = (tensor * p_weights).squeeze(0)
        denom = mask.sum(-1).unsqueeze(-1)
        denom[denom == 0] = 1

        h = h * tensor.shape[-1] / (denom)

        tensor_skip = torch.sum(h, dim=-1)

        # tensor = torch.concat([tensor, tensor_skip], dim=-1)

        pooled_vec = self.pooled_vec.repeat(tensor.shape[0], 1)
        pooled_vec = pooled_vec.unsqueeze(-1)
        
        #pooled_vec = torch.zeros((tensor.shape[0], tensor.shape[1], 1)).to(self._device)
        tensor = torch.concat([tensor, pooled_vec], dim=-1)

        self.attention_weights = []
        for l in range(self.nlayers):
            resulting_tensor = []
            for b in range(tensor.shape[0]):
                #temp, attention_weight = self.gatconv_list[l](
                temp, attention_weight = self.gatconv(
                    tensor[b, :, :].T, self.edge_index, return_attention_weights=True
                )
                
                temp = temp.T
                resulting_tensor.append(temp.unsqueeze(0))

                # save first layer attention weights
                if l == 0:
                    attention_weight = attention_weight[1]
                    attention_weight = attention_weight[
                        : attention_weight.shape[0] // 2, :
                    ]

                    attention_weight = torch.mean(attention_weight, dim=1)#.unsqueeze(-1)
                    attention_weight = attention_weight / torch.sum(attention_weight)
                    
                    #attention_weight = torch.mean(attention_weight, dim=1).unsqueeze(-1)
                    
                    self.attention_weights.append(attention_weight.unsqueeze(0))

            tensor = torch.concat(resulting_tensor, dim=0)
            tensor = self.activation(tensor)

        self.attention_weights = torch.concat(self.attention_weights, dim=0)

        # return tensor[:, :, -1]
        return (tensor[:, :, -1] * self.gat_weight) + (
            tensor_skip * (1 - self.gat_weight)
        )


def prep_gene_embeddings(datamodule, config):
    """
    This gives a dictionary of gene embeddings based on the knowledgebase or other pathways instead of concatenating
    It also gives the weights
    """

    pathway_f = config["pathway_files"]
    embedding_f = config["embedding_files"]
    n_nmf_embeddings = config["n_nmf_embedding"]

    gene_names = datamodule.train_dataset.adata.var.index.to_numpy().ravel()

    # knn not done now
    gene_embeddings = read_and_embed_new(
        pathway_f, gene_names, n_nmf_embeddings, config, datamodule
    )

    return gene_embeddings  # , weights, embeddings_from_transpose_matrix


def read_and_embed_new(pathway_file, genes_to_keep, n_nmf_embeddings, config, datamodule):
    split_path = datamodule.split_path.split("/")[-1].split(".json")[0]

    pref = (
        pathway_file.split("/")[-1].split(".txt")[0]
        + "_"
        + str(n_nmf_embeddings)
        + "_"
        + config["dim_red_alg"]
        + "_"
        + config["source"]
        + "_"
        + config["dataset"]
        + "_"
        + config["prior_type"]
        + "_"
        + str(config["prior_by_group"])
    )

    sub_config = {
        "n_nmf_embedding": config["n_nmf_embedding"],
        "node2vec_walk_length": config["node2vec_walk_length"],
        "node2vec_context_size": config["node2vec_context_size"],
        "node2vec_walks_per_node": config["node2vec_walks_per_node"],
        "node2vec_num_negative_samples": config["node2vec_num_negative_samples"],
        "node2vec_p": config["node2vec_p"],
        "node2vec_q": config["node2vec_q"],
        "node2vec_batchsize": config["node2vec_batchsize"],
        "dim_red_alg": config["dim_red_alg"],
        "prior_by_group": config["prior_by_group"],
    }

    pref = pref + "." + ".".join([str(v) for k, v in sub_config.items()])

    n_neigh_prune = config["n_neigh_prune"]
    if n_neigh_prune != "None":
        n_neigh_prune = int(n_neigh_prune)
    else:
        n_neigh_prune = None

    cache_dir = "./cache/pathway_embeddings/"
    os.makedirs(cache_dir, exist_ok=True)

    group_col = config.get("prior_by_group", None)
    is_grouped = group_col is not None and group_col != "None"

    # --- data-derived embeddings (unique per group when grouped) ---
    # stored as dict {group: [list of 3d arrays]} when grouped,
    # or a flat list when not grouped
    if is_grouped:
        obs = datamodule.train_dataset.adata.obs
        group_order = obs[group_col].unique().tolist()
        coex_per_group = {g: [] for g in group_order}
    else:
        group_order = None
        coex_dfs = []

    if config['prior_type'] in ['all', 'dp']:
        # --- transpose matrix embeddings ---
        emb_result = get_embeddings_from_training_gex_new(
            datamodule,
            cache_dir,
            "transpose_matrix",
            sub_config,
            True,
            config["dataset"] + "_" + split_path,
        )

        if is_grouped:
            for g in group_order:
                emb = _filter_to_genes(emb_result[g], genes_to_keep)
                coex_per_group[g].append(
                    emb.values.reshape(emb.shape[0], emb.shape[1], 1)
                )
        else:
            emb = _filter_to_genes(emb_result, genes_to_keep)
            coex_dfs.append(emb.values.reshape(emb.shape[0], emb.shape[1], 1))

        # --- coexpression embeddings ---
        emb_result = get_embeddings_from_training_gex_new(
            datamodule,
            cache_dir,
            "coexpression",
            sub_config,
            False,
            config["dataset"] + "_" + split_path,
        )

        if is_grouped:
            for g in group_order:
                emb = _filter_to_genes(emb_result[g], genes_to_keep)
                coex_per_group[g].append(
                    emb.values.reshape(emb.shape[0], emb.shape[1], 1)
                )
        else:
            emb = _filter_to_genes(emb_result, genes_to_keep)
            coex_dfs.append(emb.values.reshape(emb.shape[0], emb.shape[1], 1))

    # --- pathway / KG embeddings (same across all groups) ---
    pref = "WeightedDeepset." + pref + "." + str(n_neigh_prune)
    cache_file = f"{cache_dir}{pref}"
    cache_file1 = cache_file + ".embeddings.pkl"

    uniq_pert = datamodule.train_dataset.adata.obs.perturbation.unique()
    temp_pert = []
    for p in uniq_pert:
        if "_" in p:
            p_split = p.split("_")
            temp_pert.append(p_split[0])
            temp_pert.append(p_split[1])
        else:
            temp_pert.append(p)
    uniq_pert = np.unique(temp_pert)

    if pathway_file != "None":
        with open(pathway_file, "r") as f_in:
            li = f_in.readlines()

    pathway_tensor = None
    if config['prior_type'] in ['kp', 'all']:
        all_dfs = []
        if pathway_file != "None":
            with open(pathway_file, "r") as f_in:
                gene_embedding_mat = pd.DataFrame(
                    np.zeros((len(genes_to_keep), n_nmf_embeddings))
                )
                gene_embedding_mat.index = genes_to_keep

                weights_vec = pd.DataFrame(np.zeros((len(genes_to_keep), 1))).copy()
                weights_vec.index = genes_to_keep
                for i, line in enumerate(f_in):
                    emb_source = line.rstrip()
                    print(emb_source)

                    if emb_source.split(".")[-1] == "pkl":
                        with open(emb_source, "rb") as f:
                            df = pkl.load(f)
                        shared_genes = np.intersect1d(df.index, genes_to_keep)
                        df = df.loc[shared_genes, :]

                        df = df.loc[:, np.var(df, axis=0) != 0]
                        n_emb = min(df.shape[1] - 1, n_nmf_embeddings)
                        df.loc[:, :] = (
                            df.values
                            + np.random.randn(df.shape[0], df.shape[1]).astype(
                                np.float32
                            )
                            * 1e-6
                        )
                        df.loc[:, :] = df.values - np.mean(
                            df.values, axis=0, keepdims=True
                        )
                        df.loc[:, :] = df.values / np.std(
                            df.values, axis=0, keepdims=True
                        )

                        pc = PCA(n_components=n_emb).fit(df.T.values)
                        emb = pc.components_.T
                        emb = pd.DataFrame(emb)
                        emb.index = df.index

                        if emb.shape[1] < n_nmf_embeddings:
                            zero_pad = pd.DataFrame(
                                np.zeros(
                                    (emb.shape[0], n_nmf_embeddings - emb.shape[1])
                                )
                            )
                            zero_pad.index = emb.index
                            emb = pd.concat([emb.T, zero_pad.T]).T
                    else:
                        df = read_sparse_dataframe(emb_source)
                        emb = process_kg(df, cache_dir, emb_source, sub_config)
                        emb = pd.DataFrame(emb)
                        emb.index = df.index
                        emb = emb.loc[emb.index.isin(gene_embedding_mat.index)]

                    gene_embedding_mat.loc[emb.index, :] = emb.values

                    all_dfs.append(
                        gene_embedding_mat.values.reshape(
                            gene_embedding_mat.shape[0],
                            gene_embedding_mat.shape[1],
                            1,
                        ).copy()
                    )

            pathway_tensor = np.concatenate(all_dfs, -1)

            with open(cache_file1, "wb") as f:
                pkl.dump(pathway_tensor, f, protocol=pkl.HIGHEST_PROTOCOL)

    # --- assemble final output ---
    if is_grouped:
        result = {}
        for g in group_order:
            parts = coex_per_group[g]
            if pathway_tensor is not None:
                parts.append(pathway_tensor)
            result[g] = np.concatenate(parts, -1)
        return result
    else:
        if pathway_tensor is not None:
            coex_dfs.append(pathway_tensor)
        return np.concatenate(coex_dfs, -1)


def _filter_to_genes(emb, genes_to_keep):
    """Filter and reorder an embedding DataFrame to match genes_to_keep."""
    emb = emb.loc[emb.index.isin(genes_to_keep)]
    emb = emb.loc[genes_to_keep, :]
    return emb

def read_and_embed(pathway_file, genes_to_keep, n_nmf_embeddings, config, datamodule):
    split_path = datamodule.split_path.split("/")[-1].split(".json")[0]

    pref = (
        pathway_file.split("/")[-1].split(".txt")[0]
        + "_"
        + str(n_nmf_embeddings)
        + "_"
        + config["dim_red_alg"]
        + "_"
        + config["source"]
        + "_"
        + config["dataset"]
    )

    # if config['dim_red_alg'] == "Node2Vec":
    sub_config = {
        "n_nmf_embedding": config["n_nmf_embedding"],
        "node2vec_walk_length": config["node2vec_walk_length"],
        "node2vec_context_size": config["node2vec_context_size"],
        "node2vec_walks_per_node": config["node2vec_walks_per_node"],
        "node2vec_num_negative_samples": config["node2vec_num_negative_samples"],
        "node2vec_p": config["node2vec_p"],
        "node2vec_q": config["node2vec_q"],
        "node2vec_batchsize": config["node2vec_batchsize"],
        "dim_red_alg": config["dim_red_alg"],
    }

    pref = pref + "." + ".".join([str(v) for k, v in sub_config.items()])

    # sub_config['dataset'] = config['dataset']

    n_neigh_prune = config["n_neigh_prune"]
    if n_neigh_prune != "None":
        n_neigh_prune = int(n_neigh_prune)
    else:
        n_neigh_prune = None

    cache_dir = "./cache/pathway_embeddings/"
    os.makedirs(cache_dir, exist_ok=True)

    coex_dfs = []
    # compute node2vec on coperturbation effects on genes (transpose matrix)
    emb = get_embeddings_from_training_gex(
        datamodule,
        cache_dir,
        "transpose_matrix",
        sub_config,
        True,
        config["dataset"] + "_" + split_path,
    )

    emb = emb.loc[emb.index.isin(genes_to_keep)]
    emb = emb.loc[genes_to_keep, :]
    coex_dfs.append(emb.values.reshape(emb.shape[0], emb.shape[1], 1))

    # compute node2vec on coexpression matrix
    emb = get_embeddings_from_training_gex(
        datamodule,
        cache_dir,
        "coexpression",
        sub_config,
        False,
        config["dataset"] + "_" + split_path,
    )
    emb = emb.loc[emb.index.isin(genes_to_keep)]
    emb = emb.loc[genes_to_keep, :]
    coex_dfs.append(emb.values.reshape(emb.shape[0], emb.shape[1], 1))

    pref = "WeightedDeepset." + pref + "." + str(n_neigh_prune)

    cache_file = f"{cache_dir}{pref}"

    cache_file1 = cache_file + ".embeddings.pkl"
    if os.path.isfile(cache_file1):
        with open(cache_file1, "rb") as f:
            emb = pkl.load(f)

        coex_dfs.append(emb)
        emb_tensor = np.concatenate(coex_dfs, -1)

        return emb_tensor

    # not sure why this is neccesary
    # has something to do with io
    if pathway_file != "None":
        with open(pathway_file, "r") as f_in:
            li = f_in.readlines()

    # used to filter KG with few genes in perturbation set
    uniq_pert = datamodule.train_dataset.adata.obs.perturbation.unique()
    temp_pert = []
    for p in uniq_pert:
        if "_" in p:
            p_split = p.split("_")
            temp_pert.append(p_split[0])
            temp_pert.append(p_split[1])
        else:
            temp_pert.append(p)
    uniq_pert = np.unique(temp_pert)

    all_dfs = []
    if pathway_file != "None":
        with open(pathway_file, "r") as f_in:
            gene_embedding_mat = pd.DataFrame(
                np.zeros((len(genes_to_keep), n_nmf_embeddings))
            )
            gene_embedding_mat.index = genes_to_keep

            weights_vec = pd.DataFrame(np.zeros((len(genes_to_keep), 1))).copy()
            weights_vec.index = genes_to_keep
            for i, line in enumerate(f_in):
                emb_source = line.rstrip()
                print(emb_source)

                # if extension is pkl then read embedding and compute PCA
                if emb_source.split(".")[-1] == "pkl":
                    with open(emb_source, "rb") as f:
                        df = pkl.load(f)
                    shared_genes = np.intersect1d(df.index, genes_to_keep)
                    df = df.loc[shared_genes, :]

                    # remove features with no variance
                    df = df.loc[:, np.var(df, axis=0) != 0]
                    n_emb = min(df.shape[1] - 1, n_nmf_embeddings)
                    df.loc[:, :] = (
                        df.values
                        + np.random.randn(df.shape[0], df.shape[1]).astype(np.float32)
                        * 1e-6
                    )
                    df.loc[:, :] = df.values - np.mean(df.values, axis=0, keepdims=True)
                    df.loc[:, :] = df.values / np.std(df.values, axis=0, keepdims=True)

                    # reduce dimensionality of read in embeddings
                    pc = PCA(n_components=n_emb).fit(df.T.values)
                    emb = pc.components_.T

                    emb = pd.DataFrame(emb)
                    emb.index = df.index

                    if emb.shape[1] < n_nmf_embeddings:
                        zero_pad = pd.DataFrame(
                            np.zeros((emb.shape[0], n_nmf_embeddings - emb.shape[1]))
                        )
                        zero_pad.index = emb.index
                        emb = pd.concat([emb.T, zero_pad.T]).T
                # if line is directory with prefix then it's a KG that can be read in and have node2vec performed
                else:
                    # WARNING: theres a bug here. Need to fix.
                    # if the emb is computed the indices don't work for some reason
                    # it only works the second time when it's read in.
                    df = read_sparse_dataframe(emb_source)
                    # print(df)
                    emb = process_kg(df, cache_dir, emb_source, sub_config)

                    emb = pd.DataFrame(emb)

                    emb.index = df.index
                    # print(emb)
                    emb = emb.loc[emb.index.isin(gene_embedding_mat.index)]

                # frac_perturbations_in_emb = np.mean(np.isin(uniq_pert, emb.index))

                gene_embedding_mat.loc[emb.index, :] = emb.values

                all_dfs.append(
                    gene_embedding_mat.values.reshape(
                        gene_embedding_mat.shape[0], gene_embedding_mat.shape[1], 1
                    ).copy()
                )

        emb_tensor = np.concatenate(all_dfs, -1)

        with open(cache_file1, "wb") as f:
            pkl.dump(emb_tensor, f, protocol=pkl.HIGHEST_PROTOCOL)

        coex_dfs.append(emb_tensor)
    emb_tensor = np.concatenate(coex_dfs, -1)

    return emb_tensor


def process_kg(df_in, c_dir, f_in, sub_config):
    kg_processing_funcs = {
        "Node2Vec": node_2_vec,
    }

    # logic for saving and loading emb here

    emb = kg_processing_funcs[sub_config["dim_red_alg"]](df_in, c_dir, f_in, sub_config)

    return emb

def get_embeddings_from_training_gex_new(
    datamodule, cache_dir, emb_source, config, do_coexpression, dataset_split
):
    """
    Using training data and the relationship between perturbations and genes to get
    gene coexpression.
    If config['prior_by_group'] is provided, the adata is grouped by that obs column
    and a dictionary of {group_value: embedding_dataframe} is returned. Otherwise,
    a single embedding dataframe is returned.
    """
    adata = datamodule.train_dataset.adata
    group = config.get('prior_by_group', None)

    if group is not None and group != "None":
        if group not in adata.obs.columns:
            raise ValueError(f"Column '{group}' not found in adata.obs")
        emb_dict = {}
        for group_name, group_df in adata.obs.groupby(group):
            adata_sub = adata[group_df.index, :]
            emb_dict[group_name] = _compute_embeddings(
                adata_sub, datamodule, emb_source, config
            )
        return emb_dict

    return _compute_embeddings(adata, datamodule, emb_source, config)

def _compute_embeddings(adata, datamodule, emb_source, config):
    """Core embedding logic for a single adata slice."""
    if emb_source == "transpose_matrix":
        print("Computing embeddings on Transpose matrix...")
        avg_perts = [
            (tup[0], np.mean(adata[tup[1].index, :].X, axis=0))
            for tup in adata.obs.groupby(datamodule.perturb_field)
        ]
        avg_mat = pd.DataFrame(
            np.array([avg_perts[i][1] for i in range(len(avg_perts))])
        )
        avg_mat.index = [i[0] for i in avg_perts]
        avg_mat.columns = adata.var.index
        size = list(avg_mat.shape)
        size.append(config["n_nmf_embedding"])
        pc = PCA(n_components=np.min(size)).fit(avg_mat)
        emb = pc.components_.T
        emb = np.pad(
            emb,
            ((0, 0), (0, config["n_nmf_embedding"] - emb.shape[1])),
            "constant",
            constant_values=(0),
        )
        emb = pd.DataFrame(emb, index=avg_mat.columns)
    else:
        print("Computing embeddings on Coexpression...")
        size = list(adata.shape)
        size.append(config["n_nmf_embedding"])
        pc = PCA(n_components=np.min(size)).fit(adata.X)
        emb = pc.components_.T
        emb = np.pad(
            emb,
            ((0, 0), (0, config["n_nmf_embedding"] - emb.shape[1])),
            "constant",
            constant_values=(0),
        )
        emb = pd.DataFrame(emb, index=adata.var.index)
    return emb
    

def get_embeddings_from_training_gex(
    datamodule, cache_dir, emb_source, config, do_coexpression, dataset_split
):
    """
    using training data and the relationship between perturbations and genes to get
    gene coexpression
    """

    adata = datamodule.train_dataset.adata

    if emb_source == "transpose_matrix":
        print("Computing embeddings on Transpose matrix...")
        # pseudobulk for training perturbations
        avg_perts = [
            (tup[0], np.mean(adata[tup[1].index, :].X, axis=0))
            for tup in adata.obs.groupby(datamodule.perturb_field)
        ]
        avg_mat = pd.DataFrame(
            np.array([avg_perts[i][1] for i in range(len(avg_perts))])
        )
        avg_mat.index = [i[0] for i in avg_perts]
        avg_mat.columns = adata.var.index

        size = list(avg_mat.shape)
        size.append(config["n_nmf_embedding"])

        pc = PCA(n_components=np.min(size)).fit(avg_mat)

        emb = pc.components_.T

        emb = np.pad(
            emb,
            ((0, 0), (0, config["n_nmf_embedding"] - emb.shape[1])),
            "constant",
            constant_values=(0),
        )
        emb = pd.DataFrame(emb, index=avg_mat.columns)

        # cor_mat = np.corrcoef(avg_mat)
    else:
        print("Computing embeddings on Coexpression...")
        # adata = datamodule.train_dataset.adata

        size = list(adata.shape)
        size.append(config["n_nmf_embedding"])

        pc = PCA(n_components=np.min(size)).fit(datamodule.train_dataset.adata.X)
        emb = pc.components_.T
        emb = np.pad(
            emb,
            ((0, 0), (0, config["n_nmf_embedding"] - emb.shape[1])),
            "constant",
            constant_values=(0),
        )
        emb = pd.DataFrame(emb, index=datamodule.train_dataset.adata.var.index)
        # cor_mat = np.corrcoef(datamodule.train_dataset.adata.X.T)

    return emb


def node_2_vec(df_in, c_dir, f_in, node2vec_config):
    # get parameters for node2vec
    n_emb = node2vec_config["n_nmf_embedding"]
    walk_length = node2vec_config["node2vec_walk_length"]
    context_size = node2vec_config["node2vec_context_size"]
    if walk_length < context_size:
        walk_length = context_size
    walks_per_node = node2vec_config["node2vec_walks_per_node"]
    neg_samples = node2vec_config["node2vec_num_negative_samples"]
    node2vec_p = node2vec_config["node2vec_p"]
    node2vec_q = node2vec_config["node2vec_q"]
    batch_size = node2vec_config["node2vec_batchsize"]

    f_in = f_in.rstrip()
    f_pref = f_in.split("/")[-1].split(".txt")[0]

    f_pref = f_pref + "." + ".".join([str(v) for k, v in node2vec_config.items()])

    f_out = f"{c_dir}Node2Vec.{f_pref}.ind_file.pkl"

    if os.path.isfile(f_out):
        with open(f_out, "rb") as f_load:
            return pkl.load(f_load)

    df_in.index = np.arange(df_in.shape[0])
    df_in.columns = np.arange(df_in.shape[1]) + 1 + np.max(df_in.index)

    model = GraphEmbedding(
        df_in,
        n_emb=n_emb,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        neg_samples=neg_samples,
        node2vec_p=node2vec_p,
        node2vec_q=node2vec_q,
        batch_size=batch_size,
    )

    early_stop_callback = EarlyStopping(
        monitor="train_loss", min_delta=0, patience=3, verbose=False, mode="min"
    )

    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    trainer = Trainer(
        max_epochs=1000,
        callbacks=[early_stop_callback],
        accelerator=accelerator,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,  # Disable checkpointing
        # profiler="simple",
    )
    trainer.fit(model)

    embeddings = model().detach().numpy()

    # only get embeddings for the genes
    embeddings = embeddings[np.arange(df_in.shape[0]), :]

    with open(f_out, "wb") as f_load:
        pkl.dump(embeddings, f_load, protocol=pkl.HIGHEST_PROTOCOL)

    return embeddings


class GraphEmbedding(pl.LightningModule):
    def __init__(
        self,
        df_in,
        n_emb,
        walk_length,
        context_size,
        walks_per_node,
        neg_samples,
        node2vec_p,
        node2vec_q,
        batch_size,
    ):
        super(GraphEmbedding, self).__init__()

        edge_index = []
        locs = np.where(df_in.values)

        for i, j in zip(locs[0], locs[1]):
            edge_index.append([df_in.index[i], df_in.columns[j]])
            edge_index.append([df_in.columns[j], df_in.index[i]])

        edge_index = torch.tensor(edge_index).T
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        edge_index.to(device)

        self.model = Node2Vec(
            edge_index,
            embedding_dim=n_emb,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=neg_samples,
            p=node2vec_p,
            q=node2vec_q,
            sparse=True,
        ).to(device)

        self.loader = self.model.loader(
            batch_size=batch_size, shuffle=True, num_workers=0
        )

        self.current_epoch_loss = 0.0
        self.last_loss_per_epoch = 0.0

        self.save_hyperparameters()

    def forward(self):
        with torch.no_grad():
            return self.model()

    def training_step(self, batch, batch_idx):
        pos_rw, neg_rw = batch
        loss = self.model.loss(pos_rw, neg_rw)
        self.current_epoch_loss += loss.item()

        return loss

    def on_train_epoch_end(self):

        avg_loss = self.current_epoch_loss / len(self.loader)
        self.log("train_loss", avg_loss, on_epoch=True, prog_bar=False, logger=False)

        if self.current_epoch == 0:
            self.last_loss_per_epoch = avg_loss

        loss_diff = self.last_loss_per_epoch - avg_loss

        print(
            f"Epoch {self.current_epoch}: loss={avg_loss}, loss difference={loss_diff}"
        )

        self.current_epoch_loss = 0
        self.last_loss_per_epoch = avg_loss

    def configure_optimizers(self):
        optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=0.01)
        return optimizer

    def train_dataloader(self):
        return self.loader

    def save_checkpoint(self, checkpoint, filepath, storage_options=None):
        with open(filepath, "wb") as f:
            pkl.dump(checkpoint, f, protocol=pkl.HIGHEST_PROTOCOL)


def read_sparse_dataframe(input_file):
    col_file = input_file + ".columns.txt"
    row_file = input_file + ".rows.txt"
    mat_file = input_file + ".mat.npz"

    with open(col_file, "r") as f:
        columns = [line.strip() for line in f.readlines()]

    with open(row_file, "r") as f:
        rows = [line.strip() for line in f.readlines()]

    sparse_matrix = sparse.load_npz(mat_file)
    df = pd.DataFrame.sparse.from_spmatrix(
        sparse_matrix, index=rows, columns=columns
    ).sparse.to_dense()

    return df


