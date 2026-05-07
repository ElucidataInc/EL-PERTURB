import gzip
import json
import os
import shutil
from copy import deepcopy
from multiprocessing import cpu_count
from typing import List, Mapping

import numpy as np
import numpy.typing
import pandas as pd
import pytorch_lightning as pl
import requests
import scanpy as sc
import torch
from anndata import AnnData
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import (
    DataLoader,
    Dataset,
    SubsetRandomSampler,
    WeightedRandomSampler,
)
from tqdm import tqdm


def compute_pseudobulk(
    adata: AnnData, condition_field: str = "perturbation"
) -> pd.DataFrame:

    final_index = adata.obs[condition_field].tolist()
    #print(final_index)

    outdf = pd.DataFrame(
        adata.X, index=final_index, columns=adata.var_names
    ).pipe(lambda df: df.groupby(df.index).mean())

    #print("from pseudobulk fn:")
    #print(outdf.head())
    return outdf


class scPerturbData(Dataset):
    """Interface for a preprocessed AnnData object derived from scPerturb.org

    Preprocessing defined in `prepare_data` and/or `setup` methods of `scPerturbDataModule`.

    Needs to identify and separate control cells.

    Needs to compute pseudobulk and implement option to generate samples from
    either pseudobulk or single cells

    Needs to implement an invertible mapping between
    perturbation keys and indicators over adata.var_names
    """

    def __init__(
        self,
        adata,
        pert_covariates=None,
        perturb_field="perturbation",
        control_key="non-targeting",
        celltype_key="cell_type",
        perturb_group_key="perturbation_group",
        cell_types=[],
        use_pseudobulk=True,
        z_score=False,
        pre_pseudobulked=False,
    ):
        self.adata = adata
        self.pre_pseudobulked = pre_pseudobulked

        self.pert_covariates = pert_covariates

        self.perturb_field = perturb_field # CHMP3
        self.perturb_group_key = perturb_group_key # k562:CHMP3
        self.control_key = control_key # control

        # separate control cells
        self.cell_types = pd.Series(cell_types)
        print(self.cell_types)
        
        self.controls = {}
        for ct in self.cell_types:
            ctrl_idx = (adata.obs[perturb_field] == control_key) & (adata.obs[celltype_key] == ct)
            self.controls[ct] = adata[ctrl_idx]

        all_ctrl_idx = adata.obs[perturb_field] == control_key
        self.perturbs = adata[~all_ctrl_idx]

        self.control_mean = {ct:np.mean(self.controls[ct].X, axis=0, keepdims=True) for ct in self.controls}
        # TODO pseudobulks should be computed on counts in scPerturbDataModule.preprocess(),
        # between filtering and log transform, and stored in adata

        # create data and indicator matrices to serve as dataset
        perturb_keys = None
        self.X = None
        if pre_pseudobulked:
            # Data is already pseudobulked — each row IS a pseudobulk sample
            print("Pre-pseudobulked mode: skipping aggregation...")
            self.X = self.perturbs.X
            if hasattr(self.X, "toarray"):
                self.X = self.X.toarray()
            perturb_keys = self.perturbs.obs[self.perturb_group_key]
        elif use_pseudobulk:
            print("Computing pseudobulk...")
            pseudobulk = compute_pseudobulk(self.perturbs, self.perturb_group_key)

            self.X = pseudobulk.values
            perturb_keys = pseudobulk.index
        else:
            self.X = self.perturbs.X
            perturb_keys = self.perturbs.obs[self.perturb_group_key]

        print(perturb_keys)
        only_perturb_keys = perturb_keys.map(lambda x:x.split(":")[-1])

        #self.X = (X - self.control_mean).astype(np.float32) 
        #changed below to accomodate multiple cell types
        for ct in self.control_mean:
            Xctmu = self.control_mean[ct]
            mask = perturb_keys.str.startswith(ct+":")
            self.X[mask, :] -= Xctmu

        # this is only used for z-score
        self.control_std = {ct:np.std(self.controls[ct].X, axis=0, keepdims=True) for ct in self.controls}
        for ct in self.controls:
            temp = self.control_std[ct].copy()
            locs = np.where(temp == 0)
            temp[locs[0], locs[1]] = 1

            if z_score:
                mask = perturb_keys.str.startswith(ct+":")
                self.X[mask, :] = np.divide(self.X[mask, :], temp)

        if z_score:
            np.clip(self.X, a_min=-10, a_max=10, out=self.X)

        
        self.n_ntc = {ct:self.controls[ct].shape[0] for ct in self.controls}

        self.control_std = pd.concat(
            [pd.DataFrame(v).assign(cell_type=k) for k, v in self.control_std.items()],
            ignore_index=True
        )
        #self.control_std = pd.DataFrame(self.control_std)
        self.control_std.columns = self.controls[self.cell_types[0]].var.index.tolist() + [celltype_key]
        #print(self.control_std.head())

        self.control_mean = pd.concat(
            [pd.DataFrame(v).assign(cell_type=k) for k, v in self.control_mean.items()],
            ignore_index=True
        )
        #self.control_mean = pd.DataFrame(self.control_mean)
        self.control_mean.columns = self.controls[self.cell_types[0]].var.index.tolist() + [celltype_key]
        #print(self.control_mean.head())
        
        self.perturb_keys = perturb_keys.to_numpy()
        self.only_perturb_keys = only_perturb_keys.to_numpy()
        self.celltype_keys = perturb_keys.map(lambda x:x.split(":")[0]).to_numpy()

        self.var_names = adata.var_names

        #print(self.only_perturb_keys)
        #print(self.pert_to_ind(self.only_perturb_keys[0]))
        self.indmtx = np.vstack(
            [self.pert_to_ind(key) for key in self.only_perturb_keys]
        ).astype(np.float32)
        
        self.ctindmtx = np.vstack(
            [self.ctype_to_ind(key) for key in self.celltype_keys]
        ).astype(np.float32)
        
        # for now, drop perturbations of genes that aren't measured
        not_observed = self.indmtx.sum(1) == 0
        if not_observed.any():
            not_observed_keys = set(self.perturb_keys[not_observed])
            print(
                f"WARNING: Data contain perturbations for {len(not_observed_keys)} genes "
                f"for which there is no mRNA expression measurement: {not_observed_keys}.\n"
                "They will be removed because they have all-0 indicator variables."
            )
            observed = ~not_observed
            self.X = self.X[observed]
            self.perturb_keys = self.perturb_keys[observed]
            self.only_perturb_keys = self.only_perturb_keys[observed]
            self.celltype_keys = self.celltype_keys[observed]
            self.indmtx = self.indmtx[observed]
            self.ctindmtx = self.ctindmtx[observed]

        print(f"CTINDMTX: {self.ctindmtx.shape}")

        # this was only for debugging. Can be removed
        # perturb_keys_df = pd.DataFrame(perturb_keys)
        # perturb_keys_df.columns = ['perturbation']

        # Handle additional single-perturbation level covariates
        if self.pert_covariates is not None:
            # Make sure order matches indices
            self.pert_covariates = self.pert_covariates.loc[self.var_names, :]

            cov_len = self.pert_covariates.shape[1]
            self.covmtx = np.zeros((self.indmtx.shape[0], cov_len * 2))
            for i, row in enumerate(self.indmtx):
                for j, pert in enumerate(np.where(row)[0]):
                    self.covmtx[i, (j * cov_len) : ((j + 1) * cov_len)] = (
                        self.pert_covariates.iloc[pert, :].values
                    )
        else:
            self.covmtx = np.zeros((self.indmtx.shape[0], 0))
        self.covmtx = self.covmtx.astype(np.float32)

    def pert_to_ind(self, pert) -> np.typing.NDArray[np.bool_]:
        ind = self.var_names.isin(pert.split("_"))
        return ind

    def ctype_to_ind(self, ct) -> np.typing.NDArray[np.bool_]:
        ind = self.cell_types.isin([ct])
        #print(ind)
        return ind

    def ind_to_pert(self, ind) -> str:
        if hasattr(ind, "numpy"):
            ind = ind.numpy()
        ind = ind > 0
        key = "_".join(self.var_names[ind])
        
        if key not in self.only_perturb_keys:
            key = "_".join(reversed(self.var_names[ind]))
        assert key in self.only_perturb_keys, f"Could not find perturb key {key}"
        return key

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):

        return dict(
            inds=torch.tensor(self.indmtx[i]),
            expr=torch.tensor(self.X[i]),
            cov=torch.tensor(self.covmtx[i]),
            ctype=torch.tensor(self.ctindmtx[i])
        )


class NoisyPseudobulkData(scPerturbData):
    def __getitem__(self, i):
        rows = np.where(self.perturb_keys == self.perturb_keys[i])[0]
        sampled = np.random.choice(rows, size=int(0.95 * len(rows)), replace=False)
        return dict(
            inds=torch.tensor(self.indmtx[i]),
            expr=torch.tensor(np.mean(self.X[sampled], axis=0)),
            cov=torch.tensor(self.covmtx[i]),
            ctype=torch.tensor(self.ctindmtx[i])
        )


class scPerturbDataModule(pl.LightningDataModule):
    """Orchestrate creating dataloaders from scPerturb datasets."""

    urls = {
        "replogle": "https://zenodo.org/record/7041849/files/ReplogleWeissman2022_K562_gwps.h5ad",
        "replogle_k562_essential_pbase":"k562_essen_genes_raw_hgnc.h5ad",
        "replogle_k562_gwps_vcc":"k562_gwps_hvgs_5k.h5ad",
        "vcc_paper_dataset":"merged_lognorm_single_cell.h5ad"
    }

    @classmethod
    def from_config(cls, config):
        config = deepcopy(config)
        noisy_pseudobulk = config.pop("noisy_pseudobulk", False)
        pre_pseudobulked = config.pop("pre_pseudobulked", False)
        config["dataset_class"] = (
            NoisyPseudobulkData if noisy_pseudobulk else scPerturbData
        )
        config["pre_pseudobulked"] = pre_pseudobulked
        return cls(**config)

    def __init__(
        self,
        dataset,
        batch_size=64,
        nperturb_clusters=None,
        use_pseudobulk=False,
        preprocessing_zscore=False,
        allow_list=None,
        allow_list_out_genes=None,
        perturb_field="perturbation",
        control_key="non-targeting",
        celltype_key="cell_type",
        perturb_group_key="perturbation_group",
        dataset_class=scPerturbData,
        data_dir="./data/",
        pre_pseudobulked=False,
    ):
        super().__init__()
        '''
        assert (
            dataset in self.urls
        ), f"`dataset` must be one of {list(self.urls)}, not {dataset}"
        '''

        if dataset not in self.urls:
            self.urls[dataset] = dataset + '.h5ad'
            
        self.dataset = dataset
        self.batch_size = batch_size
        self.nperturb_clusters = nperturb_clusters
        self.use_pseudobulk = use_pseudobulk
        self.pre_pseudobulked = pre_pseudobulked

        # zscore is a tuple for some reason
        self.preprocessing_zscore = (preprocessing_zscore,)
        self.preprocessing_zscore = self.preprocessing_zscore[0]

        self.n_genes: int = None
        self.degs: Mapping[str, List[str]] = None
        self.var_names: pd.Index = None
        self.allow_list = allow_list
        self.allow_list_out_genes = allow_list_out_genes
        self.perturb_field = perturb_field
        self.control_key = control_key
        self.celltype_key = celltype_key
        self.perturb_group_key = perturb_group_key
        self.dataset_class = dataset_class
        self.train_dataset: scPerturbData = None
        self.val_dataset: scPerturbData = None
        self.test_dataset: scPerturbData = None
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.deg_dir, exist_ok=True)

        self._data_prepared = False
        self._data_setup = False
        self.X_train_pca = None

    @staticmethod
    def download(url, save_path) -> None:
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024

        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(save_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

    @property
    def url(self) -> str:
        return self.urls[self.dataset]

    @property
    def download_path(self) -> str:
        url = self.urls[self.dataset]
        filename = url.split("/")[-1]
        return os.path.join(self.data_dir, filename)

    @property
    def preprocessed_path(self) -> str:
        if self.allow_list is not None:
            fname = self.allow_list.split("/")[-1]
            fname = fname.split(".txt")[0]
            preprocessed_path = self.download_path.replace(
                ".h5ad", f"_{fname}_preprocessed.h5ad"
            )
        else:
            preprocessed_path = self.download_path.replace(
                ".h5ad", "_preprocessed.h5ad"
            )
        if self.allow_list_out_genes:
            fname = self.allow_list_out_genes.split("/")[-1]
            fname = fname.split(".txt")[0]
            preprocessed_path = preprocessed_path.replace(
                "_preprocessed.h5ad", f"_{fname}_preprocessed.h5ad"
            )

        return preprocessed_path

    @property
    def deg_dir(self) -> str:
        return os.path.join(self.data_dir, self.dataset, "degs")

    @property
    def merged_deg_file(self) -> str:
        return os.path.join(self.deg_dir, "merged.degs.json")

    def preprocess(self, adata: AnnData) -> AnnData:
        if self.allow_list != "None":
            allowed_perturbations = (
                pd.read_csv(self.allow_list, header=None).to_numpy().ravel()
            )
            adata = adata[adata.obs[self.perturb_field].isin(allowed_perturbations)]

        adata.var_names_make_unique()
        sc.pp.filter_genes(adata, min_cells=100)
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        if self.allow_list_out_genes != "None":
            genes = (
                pd.read_csv(self.allow_list_out_genes, header=None).to_numpy().ravel()
            )
            bool_vec = np.zeros(adata.var.shape[0])
            locs = np.where(np.isin(adata.var.index, genes))[0]
            bool_vec[locs] = 1
            adata.var["highly_variable"] = bool_vec.astype(bool)
        else:
            sc.pp.highly_variable_genes(adata, n_top_genes=5000)

        adata.var["in_perturbations"] = adata.var.index.isin(
            adata.obs[self.perturb_field].unique()
        )

        return adata[:, adata.var.highly_variable | adata.var.in_perturbations]

    def compute_degs(self, adata) -> None:
        if (adata.obs[self.perturb_field] == self.control_key).sum() > 5000:
            control_locs = (
                adata.obs[adata.obs[self.perturb_field] == self.control_key]
                .sample(5000, random_state=0)
                .index
            )
        else:
            control_locs = adata.obs[
                adata.obs[self.perturb_field] == self.control_key
            ].index

        degs = {}
        perts = set(adata.obs[self.perturb_field]) - {self.control_key}
        for pert in tqdm(perts, desc="compute_degs"):
            target = os.path.join(self.deg_dir, f"{pert}.json")
            if os.path.isfile(target):
                with open(target) as fp:
                    degs[pert] = json.load(fp)
                continue
            temp = adata[
                control_locs.union(
                    adata.obs[adata.obs[self.perturb_field] == pert].index
                )
            ].copy()
            sc.tl.rank_genes_groups(
                temp,
                groupby=self.perturb_field,
                reference=self.control_key,
                method="wilcoxon",
                rankby_abs=True,
            )
            result = (
                pd.DataFrame(temp.uns["rank_genes_groups"]["names"])
                .loc[:999, pert]
                .to_list()
            )
            degs[pert] = result
            with open(target, "w") as fp:
                json.dump(result, fp)
        with open(self.merged_deg_file, "w") as fp:
            json.dump(degs, fp)

    def create_dataset(
        self, adata: AnnData, train: bool
    ):  # -> scPerturbData | NoisyPseudobulkData:
        dataset_class = self.dataset_class if train else scPerturbData

        return dataset_class(
            adata,
            use_pseudobulk=self.use_pseudobulk,
            z_score=self.preprocessing_zscore,
            pert_covariates=self.pert_covariates,
            perturb_field=self.perturb_field,
            control_key=self.control_key,
            celltype_key=self.celltype_key,
            perturb_group_key=self.perturb_group_key,
            cell_types=self.cell_types,
            pre_pseudobulked=self.pre_pseudobulked,
        )

    def prepare_data(self) -> None:
        if not self._data_prepared:
            if self.pre_pseudobulked:
                # In pseudobulk mode, data is already preprocessed.
                # Just ensure the preprocessed file path points to the actual data.
                if not os.path.exists(self.preprocessed_path):
                    # The pseudobulk h5ad IS the preprocessed file — 
                    # copy/symlink the download_path to preprocessed_path if needed
                    if os.path.exists(self.download_path):
                        print(f"Pseudobulk mode: linking {self.download_path} -> {self.preprocessed_path}")
                        shutil.copy2(self.download_path, self.preprocessed_path)
                    else:
                        raise FileNotFoundError(
                            f"Pre-pseudobulked data not found at {self.download_path}. "
                            f"Ensure the h5ad file exists."
                        )
                else:
                    print(f"Found local preprocessed data file {self.preprocessed_path}")

                if not os.path.exists(self.merged_deg_file):
                    raise FileNotFoundError(
                        f"DEG file not found at {self.merged_deg_file}. "
                        f"DEGs must be pre-computed for pseudobulk mode."
                    )
                else:
                    print(f"Found local DEG file {self.merged_deg_file}")

                self._data_prepared = True
                return

            # Download, preprocess (incl log transform), save to local h5ad
            '''
            if not os.path.exists(self.download_path):
                print(f"Downloading from {self.url}")
                self.download(self.url, self.download_path)
            else:
                print(f"Found local data file {self.download_path}")
            '''
            if not os.path.exists(self.preprocessed_path):
                print("Preprocessing...")
                adata = sc.read(self.download_path)
                adata = self.preprocess(adata)
                sc.write(self.preprocessed_path, adata)
                print(f"Saved preprocessed data to {self.preprocessed_path}")
            else:
                print(f"Found local preprocessed data file {self.preprocessed_path}")

            if not os.path.exists(self.merged_deg_file):
                print("Computing DEGs...")
                adata = sc.read(self.preprocessed_path)
                adata.uns["log1p"][
                    "base"
                ] = None  # This is a known bug that requires this (https://github.com/scverse/scanpy/issues/2239)
                self.compute_degs(adata)
            else:
                print(f"Found local preprocessed data file {self.merged_deg_file}")

            self._data_prepared = True

    def load_preprocessed(self) -> AnnData:
        print("Loading adata...")
        adata = sc.read(self.preprocessed_path)
        
        self.cell_types = sorted(adata.obs[self.celltype_key].unique().tolist())

        if hasattr(adata.X, "toarray"):
            adata.X = adata.X.toarray()
        self.n_genes = adata.shape[1]
        print("Loading DEGs...")
        with open(self.merged_deg_file) as fp:
            self.degs = json.load(fp)
        deg_dir = "/".join(self.merged_deg_file.split("/")[:-1])

        parent_data_dir = "/".join(deg_dir.split("/")[:-1]) + "/"

        seed = self.split_path.split("/")[-1].split(".json")[0]
        self.seed = seed

        # perturbation cluster file for eval
        self.pclust_file = parent_data_dir + "eval.stratification.clusters.json"
        # genesets for virtual screen
        self.gs_file = parent_data_dir + "virtual.screen.genesets.json"

        self.ncells_per_perturbation_file = (
            f"{parent_data_dir}/ncells_per_perturbation_{seed}.json"
        )

        if not self.pre_pseudobulked:
            # Only compute ncells_per_perturbation from SC data — with pseudobulk
            # each perturbation_group has 1 row, making these counts meaningless
            if not os.path.isfile(self.ncells_per_perturbation_file):
                cells_per_perturbation = dict(adata.obs.value_counts(self.perturb_group_key))
                cells_per_perturbation_temp = {
                    i: int(j) for i, j in cells_per_perturbation.items()
                }
                with open(self.ncells_per_perturbation_file, "w") as f:
                    json.dump(cells_per_perturbation_temp, f)
        else:
            # In pseudobulk mode, set to "None" so compute_norms is skipped
            # (unless a pre-existing ncells file was created from a prior SC run)
            if not os.path.isfile(self.ncells_per_perturbation_file):
                self.ncells_per_perturbation_file = "None"

        self.var_names = adata.var_names
        print(adata.X.max())
        self.pseudobulk = compute_pseudobulk(adata, self.perturb_group_key)

        self.centered_pseudobulk = self.pseudobulk.copy()
        control_keys = [k for k in self.pseudobulk.index.tolist() if self.control_key in k]
        
        for ck in control_keys:
            self.centered_pseudobulk.loc[self.centered_pseudobulk.index.str.startswith(ck.split(":")[0]), :] -= self.pseudobulk.loc[ck, :] 

        print(adata.X.max())

        return adata

    def setup(self, stage: str) -> None:
        if not self._data_setup:
            # Load preprocessed adata and split into train/val/test
            adata = self.load_preprocessed()

            # TODO: This is a placeholder empty covariate dataframe
            self.pert_covariates = pd.DataFrame(index=self.var_names)

            with open(self.split_path, "r") as f:
                splits = json.load(f)
            self.splits = splits

            # remove pertubations from splits that got removed from adata
            for key in splits:
                splits[key] = list(
                    np.array(splits[key])[
                        np.isin(splits[key], adata.obs[self.perturb_group_key])
                    ]
                )

            if stage == "fit":
                subsets = {"train": splits["train"], "val": splits["val"]}

                for name, subset in subsets.items():
                    setattr(
                        self,
                        f"{name}_dataset",
                        self.create_dataset(
                            adata[
                                (adata.obs[self.perturb_group_key].isin(subset)) | (adata.obs[self.perturb_field].isin([self.control_key]))
                            ],
                            train=(name == "train"),
                        ),
                    )

                if self.nperturb_clusters is not None:
                    k_classifier, scaler, pc = self.cluster_perturbations(
                        self.nperturb_clusters
                    )

                    train_labels = self.get_clusters(
                        self.train_dataset.adata, k_classifier, scaler, pc
                    )

                    val_labels = self.get_clusters(
                        self.val_dataset.adata, k_classifier, scaler, pc
                    )
                    self.train_perturb_labels = {**train_labels, **val_labels}
                else:
                    self.train_perturb_labels = None

            if stage == "test":
                self.nperturb_clusters = None
                self.train_perturb_labels = None
                print(5)
                # train are added here so it computes the geometric eval on train
                self.test_dataset = self.create_dataset(
                    adata[
                        (adata.obs[self.perturb_group_key].isin(splits["test"] + splits["train"])) | (adata.obs[self.perturb_field].isin([self.control_key]))
                    ],
                    train=False,
                )

                # k_classifier = self.cluster_perturbations(self.nperturb_clusters)

                # k_classifier does not get implemented here
                # if self.nperturb_clusters != "None":
                #    test_labels = self.get_clusters(self.test_dataset.adata, k_classifier)
                #    self.test_perturb_labels = test_labels

            # set up data loaders so we can to val
            # sanity checks
            self.on_epoch_start()
            self._data_setup = True

    def pca(self, latent_dims):
        dim = min(self.train_dataset.X.shape[0], latent_dims)
        pca = PCA(dim).fit(self.train_dataset.X)

        # pca = FastICA(dim).fit(self.train_dataset.X)
        self.X_train_pca = pca.transform(self.train_dataset.X)

        return pca.transform, pca.inverse_transform

    def avg_exp_pert(self, ad):
        avg_tuples = [
            (tup[0], np.mean(ad[tup[1].index].X, axis=0))
            for tup in ad.obs.groupby(self.perturb_group_key)
        ]
        x_avg = np.array([i[1] for i in avg_tuples])
        x_avg_columns = np.array([i[0] for i in avg_tuples])
        return x_avg, x_avg_columns

    def cluster_perturbations(self, nclust):
        # clust = AgglomerativeClustering(n_clusters=nclust,linkage='complete',metric='cosine')
        clust = AgglomerativeClustering(
            n_clusters=2, linkage="ward", metric="euclidean"
        )  # changed nclust to 2.
        x_avg, x_avg_columns = self.avg_exp_pert(self.train_dataset.adata)

        scaler = StandardScaler().fit(x_avg)
        x_avg = scaler.transform(x_avg)
        # pc = PCA(n_components=4).fit(x_avg)
        pc = PCA(n_components=2).fit(x_avg)
        x_pcs = pc.transform(x_avg)

        clust.fit(x_pcs)

        k_classifier = KNeighborsClassifier(n_neighbors=1, metric="cosine").fit(
            x_pcs, clust.labels_
        )

        return k_classifier, scaler, pc

    def get_clusters(self, ad, k_classifier, scaler, pc):
        x_avg, x_avg_columns = self.avg_exp_pert(ad)
        x_avg = scaler.transform(x_avg)
        x_pc = pc.transform(x_avg)
        labels = k_classifier.predict(x_pc)
        # from collections import Counter
        # print(Counter(labels))
        return {i: l for i, l in zip(x_avg_columns, labels)}

    def _make_weights_for_balanced_classes(self, labels, n_classes):
        count = [0] * n_classes
        for label in labels:
            count[self.train_perturb_labels[label]] += 1

        weight_per_class = [0.0] * n_classes
        N = float(sum(count))
        for i in range(n_classes):
            weight_per_class[i] = N / float(count[i])

        weight = [0] * len(labels)
        for idx, label in enumerate(labels):
            weight[idx] = weight_per_class[self.train_perturb_labels[label]]
        return weight

    def on_epoch_start(self):
        self.train_length = self.train_dataset.X.shape[0]

        self.val_length = self.train_dataset.X.shape[0]
        self.subset_size_train = int(0.2 * self.train_length)

        train_indices = torch.randperm(self.train_length)[: self.subset_size_train]

        self.subset_size_val = int(0.3 * self.val_length)
        val_indices = torch.randperm(self.val_length)[: self.subset_size_val]

        train_labels = [
            label
            for label in self.train_dataset.adata[train_indices.numpy(), :].obs[
                self.perturb_group_key
            ]
        ]

        # only used if we want to rebalance classes
        # train_weights = self._make_weights_for_balanced_classes(train_labels,self.nperturb_clusters)

        self.train_indices = train_indices
        self.val_indices = val_indices

        # self.train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        self.train_sampler = SubsetRandomSamplerWithLength(train_indices)
        self.val_sampler = SubsetRandomSamplerWithLength(val_indices)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        self.on_epoch_start()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            # num_workers=cpu_count(),
            num_workers=28,
            pin_memory=True,
            # sampler=self.train_sampler
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            # num_workers=cpu_count(),
            num_workers=28,
            pin_memory=True,
            # sampler=self.val_sampler
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            # num_workers=cpu_count(),
            num_workers=28,
            pin_memory=True,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            # num_workers=cpu_count(),
            num_workers=28,
            pin_memory=True,
        )


class SubsetRandomSamplerWithLength(SubsetRandomSampler):
    """
    This is a helper class that allows you to retrieve the number of samples within a Dataset.
    """

    def __iter__(self):
        self.indices = torch.randperm(len(self.indices), generator=self.generator)
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)


def nested_valmap(func, nested_dict):
    """Apply func to each element of every list in a nested_dict"""
    if isinstance(nested_dict, list):
        return [nested_valmap(func, item) for item in nested_dict]
    elif isinstance(nested_dict, dict):
        return {k: nested_valmap(func, v) for k, v in nested_dict.items()}
    else:
        return func(nested_dict)


class ReplogleDataModule(scPerturbDataModule):
    test_set_keys = ["test"]

    def compute_degs(self, adata) -> None:
        degs = {}
        perts = set(adata.obs[self.perturb_field]) - {self.control_key}
        for pert in tqdm(perts, desc="compute_degs"):
            target = os.path.join(self.deg_dir, f"{pert}.json")
            if os.path.isfile(target):
                with open(target) as fp:
                    degs[pert] = json.load(fp)
                continue

        with open(self.merged_deg_file, "w") as fp:
            json.dump(degs, fp)

    def set_seed(self, seed: str):
        self.split_path = seed
