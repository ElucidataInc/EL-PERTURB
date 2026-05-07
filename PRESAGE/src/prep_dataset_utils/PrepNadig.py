# %%
import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys

# %%


# %%
in_dir = "./data/raw_data/"
# %%
ds = sys.argv[1]

# %%
if ds == "hepg2":
    adata = sc.read(f"{in_dir}/GSE264667_hepg2_raw_singlecell_01.h5ad")
elif ds == "jurkat":
    adata = sc.read(f"{in_dir}/GSE264667_jurkat_raw_singlecell_01.h5ad")

# %%
from collections import Counter

gene_count = Counter(adata.var.gene_name)
new_index = []
for i, g in zip(adata.var.index, adata.var.gene_name):
    if gene_count[g] > 1:
        new_index.append(g + "_" + i)
    else:
        new_index.append(g)
adata.var.index = new_index

# %%
vec = adata.obs.gene.values.ravel().astype(str)
vec[vec == "non-targeting"] = "control"
adata.obs["perturbation"] = vec

cond = []

for p in adata.obs.perturbation:
    if p == "control":
        cond.append("ctrl")
    else:
        cond.append(p + "+ctrl")
adata.obs["condition"] = cond

# %%
pert_genes = adata.obs.perturbation.unique()

# %%
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata_ctrl = adata[adata.obs.condition == "ctrl"]
sc.pp.highly_variable_genes(adata_ctrl, n_top_genes=5000)

# %%


# %%
# pert_genes

# %%
new_hvg = adata_ctrl.var.highly_variable.copy()
new_hvg[adata_ctrl.var.index.isin(pert_genes)] = True
adata.var["highly_variable"] = new_hvg
adata = adata[:, adata.var.highly_variable]

# %%
cells_per_perturbation = adata.obs.condition.value_counts() >= 10
cells_per_perturbation = cells_per_perturbation[cells_per_perturbation.values]
adata = adata[adata.obs.condition.isin(cells_per_perturbation.index)].copy()

# %%
adata.shape

# %%
from scipy.sparse import csr_matrix

adata.X = np.asarray(adata.X).astype(np.float32)
adata.X = csr_matrix(np.asarray(adata.X))


# %%
os.makedirs(f"./data/nadig_{ds}/", exist_ok=True)
adata.write(f"./data/nadig_{ds}/perturb_processed.h5ad")
