import pandas as pd
import scanpy as sc
import sys

ds = sys.argv[1]
adata_file = f"./{ds}/perturb_processed.h5ad"
adata = sc.read(adata_file)
adata.obs['condition'] = [p+"+ctrl" for p in adata.obs['perturbation']]
adata.write(adata_file)
