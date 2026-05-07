import scanpy as sc
import pandas as pd
import anndata as ad
from pathlib import Path
import sys
from tqdm import tqdm

study_folder = sys.argv[1]
ref_adata = sys.argv[2]

files = list(Path(study_folder).glob("FINAL_*/final_results_0.csv"))

rdata = sc.read_h5ad(ref_adata, backed = 'r')

for file in tqdm(files, desc = 'Pre-Processing results files for Evaluation...'):
    print(file)
    pred_df = pd.read_csv(file, index_col = 0)

    pdata = ad.AnnData(X = pred_df, obs = pd.DataFrame(index = pred_df.index.tolist()), var = pd.DataFrame(index = list(pred_df.columns)))
    print(f"\nRead in predicted perturbations:\n {pdata}")

    split_name = "_".join(file.parent.name.split('_')[2:]).split("_2026")[0] # FINAL_kpdp_K562_TF_10_UF_10_rs_1_random_2026-04-26-09-45-23
    split_name_deci = split_name.replace('10', '0.1').replace('30', '0.3').replace('50', '0.5') 

    print(split_name, split_name_deci)

    target_screen = split_name.split('_')[0]

    pdata.obs['perturbation_group'] = pdata.obs_names.tolist()
    pdata = pdata[pdata.obs.perturbation_group.str.startswith(target_screen), :].copy()
    pdata.obs['context'] = target_screen
    
    #print(pdata) 

    pdata.obs['target_gene'] = pdata.obs.perturbation_group.map(lambda x:x.split(":")[-1])
    
    test_seen_targets = rdata.obs.loc[rdata.obs[split_name_deci].isin(['test_seen']), 'target_gene'].unique().tolist()
    test_unseen_targets = rdata.obs.loc[rdata.obs[split_name_deci].isin(['test_unseen']), 'target_gene'].unique().tolist()
    pdata = pdata[pdata.obs.target_gene.isin(test_seen_targets + test_unseen_targets), :].copy()
    pdata.obs['test_split'] = 'test_seen'

    pdata.obs.loc[pdata.obs.target_gene.isin(test_unseen_targets), 'test_split'] = 'test_unseen'
    
    #print(pdata.obs.target_gene)

    #print(rdata.obs[split_name_deci].value_counts())
    
    
    pdata.uns['training_params'] = {
                                        'split_column':split_name_deci, 
                                        'target_screen':target_screen
                                }

    print(f"\nSaving Final Adata for Evaluation:\n {pdata}")

    pdata.write_h5ad(file.parent / "test_genes_pred_anndata.h5ad")
    
    #break
    
'''
params = adata_pred.uns["training_params"]
        split_col = params.get("split_column")
        target_screen = params.get("target_screen")
adata_pred.obs["target_gene"]
'''