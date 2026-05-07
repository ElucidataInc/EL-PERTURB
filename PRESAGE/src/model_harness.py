from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
#import pandas as pd
import torch
from torch import optim

from evaluator import Evaluator
from collections import defaultdict
#import scanpy as sc
import torch


def prepend_to_keys(string, dictionary):
    if string is None:
        return dictionary
    return {f"{string}_{key}": val for key, val in dictionary.items()}


def indicator_groupby(inds, cinds, arr, agg=np.mean):
    assert inds.shape[0] == arr.shape[0]
    assert cinds.shape[0] == arr.shape[0]

    keys, positions = np.unique(np.hstack([inds, cinds]), axis=0, return_inverse=True)

    m = len(keys)
    vals = np.zeros((m, arr.shape[1]))
    for i in range(m):
        vals[i] = agg(arr[positions == i], axis=0, keepdims=True)
    return keys[:, :inds.shape[1]], keys[:, inds.shape[1]:], vals
    # return keys[positions,:], vals[positions,:]


class ModelHarness(pl.LightningModule):
    """Model harness for Lightning."""

    def __init__(
        self,
        module,
        datamodule,
        config,
        encoder=None,
        decoder=None,
    ):
        super().__init__()
        self.module = module
        self.module.current_batch = 0
        self.var_names = datamodule.var_names
        self.degs = datamodule.degs

        self.config = config

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.train_step_outputs = []
        self.test_set_keys = getattr(datamodule, "test_set_keys", [""])
        self.encoder = encoder
        self.decoder = decoder

        datamodule.encoder = encoder
        self.do_test_eval = getattr(datamodule, "do_test_eval", True)

        train_adata = datamodule.train_dataset.adata

        ctrl_cells = {}
        for ct in datamodule.cell_types:
            ctrl_idx = (train_adata.obs[datamodule.perturb_field] == datamodule.control_key) & (train_adata.obs[datamodule.celltype_key] == ct)
            ctrl_cells[ct] = train_adata[ctrl_idx]

        
        control_mean = {ct:np.mean(ctrl_cells[ct].X, axis=0, keepdims=True) for ct in ctrl_cells}
        
        train_keys = datamodule.splits["train"]

        # ctrl keys used for sphering

        # 1 / 0
        adata = datamodule.load_preprocessed()
        for ct in datamodule.cell_types:
            ct_pert_cells = adata.obs[datamodule.celltype_key] == ct
            adata.X[ct_pert_cells, :] -= control_mean[ct]

        all_ctrl_cells = train_adata[(train_adata.obs[datamodule.perturb_field] == datamodule.control_key), :]
        self.evaluator = Evaluator(
            self.var_names,
            self.degs,
            all_ctrl_cells,
            train_keys,
            adata,
            geneset_file=datamodule.gs_file,
            # perturbation_cluster_file=datamodule.pclust_file,
            ncells_per_perturbation_file=datamodule.ncells_per_perturbation_file,
            dataset=datamodule.dataset,
            seed=datamodule.seed,
        )

        if hasattr(datamodule, "degs_combo_deviation"):
            self.second_evaluator = Evaluator(
                self.var_names,
                datamodule.degs_combo_deviation,
                all_ctrl_cells,
                train_keys,
                kvals=(10, 20, 40),
                geneset_file=datamodule.gs_file,
                perturbation_cluster_file=datamodule.pclust_file,
            )

        if datamodule.nperturb_clusters != "None":
            self.train_perturb_labels = datamodule.train_perturb_labels
        else:
            self.train_perturb_labels = "None"

        # perturbation embeddings from the model after pathway pooling
        self.all_embh = []
        self.all_embh_final = []
        # linear coefficients for combos from the model
        self.all_coef = []

        # saving attention weights
        self.attention_weights = []
        self.transformed_embs = []

        # these will be for piecing together all_embh and all_coef
        self.all_locs_gene = []
        self.all_locs_ind = []
        self.all_locs_ctype = []

        self.cell_types = datamodule.cell_types

    def ind_to_pert(self, ind, cind) -> str:
        if hasattr(ind, "numpy"):
            ind = ind.numpy()
        ind = ind > 0

        cind = np.where(cind > 0)[0]
        #print(cind, self.cell_types)
        key = self.cell_types[cind[0]] + ":" + "_".join(self.var_names[ind])

        if key not in self.degs:
            key = self.cell_types[cind[0]] + ":" + "_".join(reversed(self.var_names[ind]))

        assert key in self.degs, f"Could not find perturb key {key}."
        return key

    def unpack_batch(self, batch):
        src, cov, tgt, ctype = batch["inds"], batch["cov"], batch["expr"], batch["ctype"]
        if self.encoder is not None:
            tgt = torch.tensor(self.encoder(tgt.cpu().numpy())).to(tgt.device)
            ctype = torch.tensor(self.encoder(ctype.cpu().numpy())).to(ctype.device)
        return src, cov, tgt, ctype

    def forward(self, inds, ctype, cov):
        return self.module(inds, ctype, cov)

    def _step(self, batch, batch_idx):
        src, cov, tgt, ctype = self.unpack_batch(batch)
        self.train_perturb_labels = None
        if self.train_perturb_labels is not None:
            keys = np.array([self.ind_to_pert(ind, cind) for (ind, cind) in zip(src.cpu(), ctype.cpu())])

            pred, tensor, pred_clust = self(src, ctype, cov)
            tgt_clust = torch.tensor([self.train_perturb_labels[k] for k in keys]).to(
                pred_clust.get_device()
            )
            # loss, ce_loss, accuracy = self.module.compute_loss(pred, tgt, pred_clust,tgt_clust) # new loss from Russell
            loss, ce_loss, accuracy = self.module.compute_loss(pred, tgt, tensor)
        else:
            pred, tensor, pred_clust = self(src, ctype, cov)  # what are src, cov?
            # loss = self.module.compute_loss(pred, tgt, pred_clust, None)
            loss = self.module.compute_loss(pred, tgt, tensor)
            ce_loss = None
            accuracy = None

        return loss, ce_loss, accuracy, src, pred, ctype

    def training_step(self, batch, batch_idx):
        self.module.current_batch += 1
        loss, ce_loss, accuracy, src, pred, ctype = self._step(batch, batch_idx)

        """src, cov, tgt = self.unpack_batch(batch)

        if self.train_perturb_labels != "None":
            keys = np.array([self.ind_to_pert(ind) for ind in src.cpu()])
            
            pred, pred_clust = self(src, cov)
            tgt_clust = torch.tensor([self.train_perturb_labels[k] for k in keys]).to(pred_clust.get_device())
            loss, ce_loss = self.module.compute_loss(pred, tgt,pred_clust,tgt_clust)
        else:
            pred = self(src, cov)
            loss = self.module.compute_loss(pred, tgt)"""

        self.log("train_loss", loss)
        if self.train_perturb_labels is not None:
            self.log("train_classification_loss", ce_loss)
            self.log("train_classification_accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):

        # src, cov, tgt = self.unpack_batch(batch)
        # pred = self(src, cov)
        # loss = self.module.compute_loss(pred, tgt)

        loss, ce_loss, accuracy, src, pred, ctype = self._step(batch, batch_idx)

        self.log("val_loss", loss)
        self.validation_step_outputs.append(
            {"src": src, "tgt": batch["expr"], "pred": pred, "ctype":ctype}
        )
        # self.train_perturb_labels = None
        if self.train_perturb_labels is not None:
            self.log("val_classification_loss", ce_loss)
            self.log("val_classification_accuracy", accuracy)

    def on_validation_epoch_end(self):
        self.module.current_batch = 0
        # collect validation step outputs
        src, tgt, pred, ctype = [], [], [], []
        for x in self.validation_step_outputs:
            src.append(x["src"].cpu())
            tgt.append(x["tgt"].cpu())
            ctype.append(x["ctype"].cpu())
            if x["pred"] is not None:
                pred.append(x["pred"].cpu())
            else:
                pred.append(x["pred"])
        src = np.concatenate(src)
        tgt = np.concatenate(tgt)
        ctype = np.concatenate(ctype)
        
        if x["pred"] is not None:
            pred = np.concatenate(pred)
            if self.decoder is not None:
                pred = self.decoder(pred)

            # compute pseudobulks
            keys, ckeys, mean_preds = indicator_groupby(src, ctype, pred)
            _, _, mean_tgts = indicator_groupby(src, ctype, tgt)
            
            keys = np.array([self.ind_to_pert(key, ckey) for (key, ckey) in zip(keys, ckeys)])

            # pass to eval suite and log results
            # self.log_dict(self.evaluator(keys, mean_tgts, mean_preds, False))

            # clean up
            self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # src, cov, tgt = self.unpack_batch(batch)
        # pred = self(src, cov)
        # loss = self.module.compute_loss(pred, tgt)

        # this is a temporary hack since k_classifier not implemented for test
        self.train_perturb_labels = None
        loss, ce_loss, accuracy, src, pred, ctype = self._step(batch, batch_idx)

        self.log("test_loss", loss)
        self.test_step_outputs.append(
            {"src": src, "tgt": batch["expr"], "pred": pred, "dl_idx": dataloader_idx, "ctype": ctype}
        )
        if self.train_perturb_labels is not None:
            self.log("test_classification_loss", ce_loss)
            self.log("test_classification_accuracy", accuracy)

    def on_test_epoch_end(self):

        src, tgt, pred, ctype = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        for x in self.test_step_outputs:
            src[x["dl_idx"]].append(x["src"].cpu())
            tgt[x["dl_idx"]].append(x["tgt"].cpu())
            if x["pred"] is not None:
                pred[x["dl_idx"]].append(x["pred"].cpu())
            else:
                pred[x["dl_idx"]].append(x["pred"])
                
            ctype[x["dl_idx"]].append(x["ctype"].cpu())

        if x["pred"] is not None:
            if self.decoder is not None:
                for dl_idx in pred:
                    pred[dl_idx] = [self.decoder(x) for x in pred[dl_idx]]

            for dl_idx in src:

                keys, ckeys, mean_preds = indicator_groupby(
                    np.concatenate(src[dl_idx]), np.concatenate(ctype[dl_idx]), np.concatenate(pred[dl_idx])
                )
                _, _, mean_tgts = indicator_groupby(
                    np.concatenate(src[dl_idx]), np.concatenate(ctype[dl_idx]), np.concatenate(tgt[dl_idx])
                )

                keys = np.array([self.ind_to_pert(key, ckey) for (key, ckey) in zip(keys, ckeys)])

                current_test_set = self.test_set_keys[dl_idx]

                # fixes instabilities before eval
                # mean_preds[np.isnan(mean_preds)] = 0

                if self.do_test_eval:
                    self.log_dict(
                        prepend_to_keys(
                            current_test_set,
                            self.evaluator(keys, mean_tgts, mean_preds, True),
                        )
                    )
                    if hasattr(self, "second_evaluator"):
                        print("Doing non-linear DEG eval")
                        if current_test_set != "unseen_single":
                            self.log_dict(
                                prepend_to_keys(
                                    "deviation_" + current_test_set,
                                    self.second_evaluator(
                                        keys, mean_tgts, mean_preds, True
                                    ),
                                )
                            )
                else:
                    self.log_dict(
                        prepend_to_keys(
                            current_test_set,
                            self.evaluator(keys, mean_tgts, mean_preds, False),
                        )
                    )
        self.test_step_outputs.clear()
            

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # I need to fix this because of the classification
        src, cov, tgt, ctype = self.unpack_batch(batch)
        preds, tensor, pred_clust = self(src, ctype, cov)

        if hasattr(self.module, "emb_h"):
            self.all_embh.append(self.module.emb_h.detach().cpu().numpy())

        if hasattr(self.module, "emb_h_final"):
            self.all_embh_final.append(self.module.emb_h_final.detach().cpu().numpy())
            
            # self.all_coef.append(self.module.linear_weights.detach().cpu().numpy())
        self.all_locs_gene.append(self.module.locs_gene.detach().cpu().numpy())
        self.all_locs_ind.append(self.module.locs_combos.detach().cpu().numpy())
        self.all_locs_ctype.append(self.module.locs_ctype.detach().cpu().numpy())

        if hasattr(self.module, "pathway_weight_vector"):
            self.pathway_weight_vector = self.module.pathway_weight_vector
        if hasattr(self.module, "attention_weights"):
            self.attention_weights.append(self.module.attention_weights)

        self.transformed_embs.append(self.module.emb_h.cpu().numpy())

        if preds is not None:
            if self.decoder is not None:
                preds = self.decoder(preds.cpu())
            else:
                print('self.decoder is none')
            # keys, mean_preds = indicator_groupby(src.cpu(), pred)
            keys = np.array([self.ind_to_pert(ind, cind) for (ind, cind) in zip(src.cpu(), ctype.cpu())])
            return keys, preds

    def configure_optimizers(self):
        # only SGD w/ momentum and Adam implemented
        if self.config["optimizer"] == "SGD":
            opt = optim.SGD(
                self.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
                momentum=self.config["momentum"],
            )
        else:
            opt = optim.Adam(
                self.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )

        scheduler = optim.lr_scheduler.StepLR(opt, step_size=200, gamma=1)
        # scheduler = optim.lr_scheduler.StepLR(opt, step_size=5,gamma=0.5)

        return [opt], [scheduler]
