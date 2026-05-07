import os
from zipfile import ZipFile
import scanpy as sc
import json
import re
import pandas as pd
import numpy as np
from scipy import sparse
import scipy.sparse as sp

import pytorch_lightning as pl
from datamodule_multicell import scPerturbDataModule, ReplogleDataModule, compute_pseudobulk
from torch.utils.data import DataLoader, Dataset


class PRESAGEDataModule(scPerturbDataModule):
    urls = {
        "adamson": "https://dataverse.harvard.edu/api/access/datafile/6154417",
        "dixit": "https://dataverse.harvard.edu/api/access/datafile/6154416",
        "replogle_k562_essential": "https://dataverse.harvard.edu/api/access/datafile/7458695",
        "replogle_rpe1_essential": "https://dataverse.harvard.edu/api/access/datafile/7458694",
        "wessels_2023":"perturb_processed.h5ad",
        "replogle_2020":"perturb_processed.h5ad",
        "replogle_k562_essential_unfiltered":"perturb_processed.h5ad",
        "replogle_rpe1_essential_unfiltered":"perturb_processed.h5ad",
        "replogle_k562_gw":"perturb_processed.h5ad",
        "nadig_hepg2":"perturb_processed.h5ad",
        "nadig_jurkat":"perturb_processed.h5ad",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(self.data_dir)
        os.makedirs(self.dataset_dir, exist_ok=True)

    @property
    def preprocessed_path(self) -> str:
        return os.path.join(self.dataset_dir, f"{self.dataset}_processed.h5ad")

    @property
    def dataset_dir(self) -> str:
        return os.path.join(self.data_dir, self.dataset)

    @property
    def deg_dir(self) -> str:
        return os.path.join(self.dataset_dir, "degs")

    @property
    def download_path(self) -> str:
        url = self.urls[self.dataset]
        filename = url.split("/")[-1]
        return os.path.join(self.dataset_dir, filename)

    @property
    def raw_path(self) -> str:
        return os.path.join(self.dataset_dir, "perturb_processed.h5ad")

    def prepare_data(self) -> None:
        # Download archive
        print(self.download_path)
        if not os.path.exists(self.download_path):
            if self.url == "perturb_processed.h5ad":     
                print(f"{self.dataset} path is not downloadable. This must be downloaded separately")
                quit()
            print(f"Downloading from {self.url}")
            self.download(self.url, self.download_path)
        else:
            print(f"Found local data file {self.download_path}")

        # Extract pre-processed data
        if not os.path.exists(self.raw_path):
            with ZipFile(self.download_path, "r") as f:
                f.extractall(path=self.data_dir)
        else:
            print(f"Found local extracted data file {self.raw_path}")
        
        

        if not os.path.exists(self.preprocessed_path):
            # read normally (option: use backed='r' for very large files - see notes below)
            adata = sc.read(self.raw_path)
        
            # 1) keep indices small and stringified (no big copies)
            adata.obs.index = adata.obs.index.astype(str)
        
            # 2) harmonize perturbation column in a vectorized way (no apply)
            # remove old perturbation col if present, then rename condition -> perturbation
            if 'perturbation' in adata.obs.columns:
                adata.obs = adata.obs.drop(columns=['perturbation'])
            adata.obs.rename(columns={'condition': 'perturbation'}, inplace=True)
        
            # fillna and vectorized replacements
            adata.obs['perturbation'] = adata.obs['perturbation'].astype(str)
            pert = adata.obs['perturbation'].fillna('control').astype(str)
            pert = (pert.str.replace('+', '_', regex=False)
                        .str.replace('ctrl', '', regex=False)
                        .str.strip('_'))
            pert = pert.replace('', 'control')
            # use categorical to save memory if there are many repeated values
            adata.obs['perturbation'] = pd.Categorical(pert)
        
            # 3) nperts vectorized
            # if there's an underscore it's 2+ perturbations -> 1 + ("_" in x) in original code
            adata.obs['nperts'] = (adata.obs['perturbation'].astype(str).str.contains('_').astype(np.int8) + 1)
            adata.obs.loc[adata.obs['perturbation'] == 'control', 'nperts'] = 0
            adata.obs['nperts'] = adata.obs['nperts'].astype(np.int8)
        
            # 4) var / gene_name handling (avoid resetting index multiple times)
            if 'gene_name' not in adata.var.columns:
                adata.var['gene_name'] = adata.var.index.astype(str)
            adata.var['gene_name'] = adata.var['gene_name'].astype(str)
            # set gene_name as index (unique names expected)
            adata.var = adata.var.set_index('gene_name', drop=False)
            adata.var_names_make_unique()
        
            # 5) find perturbations that don't appear as measured genes
            # (these will become additional columns/genes with all-zero expression)
            # avoid building an intermediate large boolean array where possible
            measured_genes = set(adata.var.index)
            obs_perts = pd.Series(adata.obs['perturbation'].astype(str).unique())
            missing_perturbations = np.array([p for p in obs_perts if p not in measured_genes])
        
            # 6) if there are missing perturbations, expand adata.X using sparse matrix hstack
            if missing_perturbations.size > 0:
                n_cells = adata.n_obs
                n_missing = missing_perturbations.size
        
                # build zero sparse block (cheap)
                zeros_block = sp.csr_matrix((n_cells, n_missing), dtype=np.float32)
        
                # If adata.X is sparse, use sparse.hstack (does not densify)
                if sp.issparse(adata.X):
                    new_X = sp.hstack([adata.X, zeros_block], format='csr')
                else:
                    # If dense, add dense zeros (this may be necessary if your X is already dense)
                    new_X = np.hstack([adata.X, np.zeros((n_cells, n_missing), dtype=np.float32)])
        
                # build new var DataFrame (concatenate indices, avoid copies)
                old_var_index = list(adata.var.index)
                new_index = old_var_index + list(missing_perturbations)
                var_df = pd.DataFrame(index=new_index)
                var_df['gene_name'] = var_df.index.astype(str)
                var_df['measured_gene'] = [True] * len(old_var_index) + [False] * n_missing
        
                # create new AnnData with existing obs (no extra copies)
                adata = sc.AnnData(X=new_X, obs=adata.obs, var=var_df)
        
            # 7) ensure correct numeric dtype (sparse astype keeps sparse)
            if sp.issparse(adata.X):
                adata.X = adata.X.astype(np.float32)
            else:
                adata.X = adata.X.astype(np.float32, copy=False)
        
            # write final file
            adata.write(self.preprocessed_path)
        else:
            print(f"Found local preprocessed data file {self.preprocessed_path}")

        # Compute DEGs
        if not os.path.exists(self.merged_deg_file):
            print("Computing DEGs...")
            print(self.raw_path)
            adata = sc.read(self.raw_path)
            adata.obs.index = np.arange(adata.shape[0]).astype(str)

            adata.var.index.name = None
            adata.var['gene_name'] = adata.var.index.tolist()
            if 'gene_name' in adata.var.columns:
                id2name_table = adata.var.reset_index().set_index("index")['gene_name']
                #adata.var.columns = ['gene_name']
            else:
                id2name_table = adata.var.reset_index().set_index("gene_id")["gene_name"]
                #adata.var.columns = ['gene_name']

            
            degs = adata.uns["rank_genes_groups_cov_all"]
            
            # Harmonize to scPerturb perturbation keys
            degs = {
                re.sub(
                    r"^.*?_(.*)",
                    r"\1",
                    k.replace("_1+1", "").replace("+", "_").replace("ctrl", ""),
                )
                .strip("_"): id2name_table[v]
                .tolist()
                for k, v in degs.items()
            }
            with open(self.merged_deg_file, "w") as f:
                json.dump(degs, f)
        else:
            print(f"Found local preprocessed data file {self.merged_deg_file}")


class ReploglePRESAGEDataModule(PRESAGEDataModule, ReplogleDataModule):
    pass
