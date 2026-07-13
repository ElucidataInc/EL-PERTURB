"""
Perturb-Prediction MLP Concat Pipeline
------------------------------------------
This script implements an MLP-based pipeline for predicting transcriptional responses
to genetic perturbations in a target cellular context. Multiple prior sources—biological
embeddings, cross-context data priors, and context-dependent PCA loadings—are concatenated
into a single input vector and passed through a feed-forward network.

Key Features:
1. Multi-Source Concatenation: Joins knowledge priors (ESM, GenePT, pathways) and
   cross-context data priors into one high-dimensional latent vector per perturbation.
2. Dimensionality-Matched Priors: External embeddings and cross-context responses are
   PCA-reduced or zero-padded to a shared prior dimension before fusion.
3. Feature Fusion: A linear projection maps the concatenated prior vector into a shared
   embedding space, followed by an MLP decoder for expression prediction.
4. Weighted MSE (WMSE): Optional per-gene loss weighting derived from significance scores
   stored in the input AnnData object.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import pickle

# --- 1. Data Handling ---

class PerturbationDataset(Dataset):
    def __init__(self, 
                 x_pert: torch.Tensor, 
                 x_context: torch.Tensor, 
                 y: torch.Tensor,
                 weights: Optional[torch.Tensor] = None,
                 obs_indices: Optional[List[str]] = None):
        self.x_pert = x_pert
        self.x_context = x_context
        self.y = y
        self.weights = weights
        self.obs_indices = obs_indices

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        item = {
            'pert': self.x_pert[idx],
            'y': self.y[idx],
            'idx': idx
        }
        if self.weights is not None:
            item['weight'] = self.weights[idx]
        return item

class DataModule:
    def __init__(self, h5ad_path: str, target_screen: str, split_column: str = 'split_random'):
        self.h5ad_path = h5ad_path
        self.target_screen = target_screen
        self.split_column = split_column
        
        self.adata = None
        self.pca_model = None
        self.vocab_pert = {}
        self.vocab_context = {}
        
        self.knowledge_prior_tensor = None
        self.contextwise_loadings = None
        self.data_prior_tensor = None
        self.output_dim = 0
        
    def load_and_preprocess(self, 
                            use_pca: bool = False, 
                            pca_dim: int = 50, 
                            embedding_paths: Optional[List[str]] = None,
                            embedding_pca_dim: int = 128,
                            use_data_prior: bool = True,
                            seed: int = 42):
        print(f"Loading data from {self.h5ad_path}...")
        full_adata = sc.read_h5ad(self.h5ad_path)
        
        # 1. Initialize Vocabulary based on TARGET genes in the dataset
        valid_splits = ['train', 'test_seen', 'test_unseen']
        ignored_names = {'control', 'non-targeting', 'Control', 'Non-targeting'}
        mask_target = (full_adata.obs['context'] == self.target_screen) & (full_adata.obs[self.split_column].isin(valid_splits))
        mask_target &= ~full_adata.obs['target_gene'].isin(ignored_names)
        
        self.adata = full_adata[mask_target].copy()
        print(f"Kept {len(self.adata)} samples from {self.target_screen} screen.")
        
        all_perts = sorted(self.adata.obs['target_gene'].unique())
        self.vocab_pert = {p: i for i, p in enumerate(all_perts)}
        self.vocab_context = {c: i for i, c in enumerate(sorted(self.adata.obs['context'].unique()))}
        
        n_perts = len(all_perts)
        target_genes = all_perts
        
        # 2. Load cross-context data priors
        cross_context_priors = []
        if use_data_prior:
            print(f"Computing Cross-Context Data Priors (PB-LFC per screen other than {self.target_screen})...")
            other_contexts = [c for c in full_adata.obs['context'].unique() if c != self.target_screen]
            
            for context in other_contexts:
                context_mask = (full_adata.obs['context'] == context) & (full_adata.obs[self.split_column].isin(valid_splits))
                adata_c = full_adata[context_mask].copy()
                
                if len(adata_c) == 0:
                    continue
                    
                gene_means = {}
                for gene in adata_c.obs['target_gene'].unique():
                    if gene in ignored_names: continue
                    gene_mask = adata_c.obs['target_gene'] == gene
                    gene_X = adata_c[gene_mask].X
                    if hasattr(gene_X, 'toarray'): gene_X = gene_X.toarray()
                    gene_means[gene] = np.mean(gene_X, axis=0)
                
                if not gene_means:
                    continue
                
                matrix_c = np.zeros((n_perts, full_adata.shape[1]))
                global_mean_c = np.mean(list(gene_means.values()), axis=0)
                
                for i, gene in enumerate(target_genes):
                    if gene in gene_means:
                        matrix_c[i] = gene_means[gene]
                    else:
                        matrix_c[i] = global_mean_c
                
                pca_c = PCA(n_components=min(embedding_pca_dim, matrix_c.shape[0], matrix_c.shape[1]), random_state=seed)
                reduced_c = pca_c.fit_transform(matrix_c)
                
                tensor_c = torch.tensor(reduced_c, dtype=torch.float32)
                if tensor_c.shape[1] < embedding_pca_dim:
                    padding = torch.zeros((n_perts, embedding_pca_dim - tensor_c.shape[1]))
                    tensor_c = torch.cat([tensor_c, padding], dim=1)
                
                cross_context_priors.append(tensor_c)

        # 3. Load Knowledge Prior (External Embeddings)
        knowledge_sources = []
        if embedding_paths:
            kp_tensor = self.load_knowledge_embeddings(embedding_paths, embedding_pca_dim)
            if kp_tensor is not None:
                for s in range(kp_tensor.shape[0]):
                    knowledge_sources.append(kp_tensor[s])
        
        # 4. Merge knowledge sources and cross-context data priors
        all_sources = knowledge_sources + cross_context_priors
        if all_sources:
            self.knowledge_prior_tensor = torch.stack(all_sources)
            print(f"Total Prior Sources Integrated: {len(all_sources)} (Knowledge: {len(knowledge_sources)}, Data: {len(cross_context_priors)})")
        else:
            self.knowledge_prior_tensor = None

        # 5. Compute Context-Dependent PCA Feature Loadings (Additional Prior)
        print(f"Computing Context-Dependent PCA Feature Loadings (target_dim={embedding_pca_dim})...")
        n_contexts = len(self.vocab_context)
        loadings_tensor = torch.zeros((n_contexts, n_perts, embedding_pca_dim))

        train_mask = self.adata.obs[self.split_column] == 'train'
        var_to_idx = {name: i for i, name in enumerate(self.adata.var_names)}
        
        for context_name, c_idx in self.vocab_context.items():
            context_train_mask = train_mask & (self.adata.obs['context'] == context_name)
            if context_train_mask.sum() < 2:
                continue
                
            try:
                sub_y = self.adata[context_train_mask].X.toarray()
            except:
                sub_y = self.adata[context_train_mask].X
                
            pca_load = PCA(n_components=min(embedding_pca_dim, sub_y.shape[0], sub_y.shape[1]), random_state=seed)
            pca_load.fit(sub_y)
            
            loadings = pca_load.components_.T 
            
            for pert_name, p_idx in self.vocab_pert.items():
                if pert_name in var_to_idx:
                    feat_idx = var_to_idx[pert_name]
                    v = torch.tensor(loadings[feat_idx], dtype=torch.float32)
                    if v.shape[0] < embedding_pca_dim:
                        v = torch.cat([v, torch.zeros(embedding_pca_dim - v.shape[0])])
                    loadings_tensor[c_idx, p_idx] = v
        
        self.contextwise_loadings = loadings_tensor
        
        # 6. PCA on target screen targets if requested
        try:
            raw_y = self.adata.X.toarray()
        except:
            raw_y = self.adata.X
            
        if use_pca:
            print(f"Fitting PCA (n_components={pca_dim}) on TRAIN data only...")
            train_X = raw_y[train_mask]
            self.pca_model = PCA(n_components=pca_dim, random_state=seed)
            self.pca_model.fit(train_X)
            y_data = self.pca_model.transform(raw_y)
            self.output_dim = pca_dim
        else:
            y_data = raw_y
            self.output_dim = y_data.shape[1]
            
        self.all_x_pert = torch.tensor(self.adata.obs['target_gene'].map(self.vocab_pert).values, dtype=torch.long)
        self.all_x_context = torch.tensor(self.adata.obs['context'].map(self.vocab_context).values, dtype=torch.long)
        self.all_y = torch.tensor(y_data, dtype=torch.float32)
        self.all_obs_names = self.adata.obs_names.tolist()
        
    def load_knowledge_embeddings(self, embedding_paths: List[str], target_p_dim: int = 128):
        if not embedding_paths:
            return None
            
        print(f"Loading {len(embedding_paths)} knowledge prior files...")
        processed_sources = []
        target_genes = sorted(self.vocab_pert.keys(), key=lambda x: self.vocab_pert[x])
        
        for path in embedding_paths:
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as f:
                data = pickle.load(f)
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame.from_dict(data, orient='index')
            
            df_aligned = df.reindex(target_genes)
            df_aligned = df_aligned.fillna(df.mean(axis=0)) if not df.empty else df_aligned.fillna(0)
            
            raw_vals = df_aligned.values
            n_perts, d_orig = raw_vals.shape
            
            if d_orig > target_p_dim:
                pca = PCA(n_components=target_p_dim, random_state=42)
                reduced = pca.fit_transform(raw_vals)
                source_tensor = torch.tensor(reduced, dtype=torch.float32)
            else:
                source_tensor = torch.tensor(raw_vals, dtype=torch.float32)
                if d_orig < target_p_dim:
                    padding = torch.zeros((n_perts, target_p_dim - d_orig))
                    source_tensor = torch.cat([source_tensor, padding], dim=1)
            
            processed_sources.append(source_tensor)
        
        return torch.stack(processed_sources) if processed_sources else None
        
    def get_split_indices(self) -> Dict[str, np.ndarray]:
        indices = np.arange(len(self.adata))
        splits = {}
        for label in ['train', 'test_seen', 'test_unseen']:
            mask = self.adata.obs[self.split_column] == label
            splits[label] = indices[mask]
        return splits

    def create_dataset(self, indices: np.ndarray, weights: Optional[torch.Tensor] = None) -> PerturbationDataset:
        return PerturbationDataset(
            self.all_x_pert[indices],
            self.all_x_context[indices],
            self.all_y[indices],
            weights[indices] if weights is not None else None,
            [self.all_obs_names[i] for i in indices]
        )

# --- 2. Model Architecture (Concatenation Strategy) ---

class MLPConcatModel(nn.Module):
    def __init__(self, 
                 n_perts: int, 
                 output_dim: int,
                 emb_dim: int = 64, 
                 hidden_dim: int = 256,
                 dropout: float = 0.2,
                 knowledge_prior_tensor: Optional[torch.Tensor] = None,
                 context_prior_tensor: Optional[torch.Tensor] = None): 
        super(MLPConcatModel, self).__init__()
        
        self.use_priors = (knowledge_prior_tensor is not None or context_prior_tensor is not None)
        
        if self.use_priors:
            self.pert_embedding = None
            num_knowledge_sources = 0
            prior_dim = 0
            
            if knowledge_prior_tensor is not None:
                self.register_buffer('knowledge_priors', knowledge_prior_tensor)
                num_knowledge_sources = knowledge_prior_tensor.shape[0]
                prior_dim = knowledge_prior_tensor.shape[2]
            else:
                self.knowledge_priors = None
                
            if context_prior_tensor is not None:
                self.register_buffer('context_priors', context_prior_tensor)
                num_context_sources = 1 # Consistent with ALT pipeline (target screen prior)
                if prior_dim == 0: prior_dim = context_prior_tensor.shape[2]
            else:
                self.context_priors = None
                num_context_sources = 0

            self.num_total_sources = num_knowledge_sources + num_context_sources
            # Concatenation projection: (num_sources * prior_dim) -> emb_dim
            self.fusion_projection = nn.Linear(self.num_total_sources * prior_dim, emb_dim)
        else:
            self.pert_embedding = nn.Embedding(n_perts, emb_dim)
            self.fusion_projection = None
            self.knowledge_priors = None
            self.context_priors = None
            
        self.pert_encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x_pert):
        if self.use_priors:
            priors_list = []
            
            if self.knowledge_priors is not None:
                # knowledge_priors shape: (num_k, n_perts, dim)
                for i in range(self.knowledge_priors.shape[0]):
                    priors_list.append(self.knowledge_priors[i, x_pert, :])
                
            if self.context_priors is not None:
                # context_priors shape: (n_contexts, n_perts, dim)
                priors_list.append(self.context_priors[0, x_pert, :])
                
            # Concatenate all sources along feature dimension
            concatenated = torch.cat(priors_list, dim=1) # (batch, num_sources * prior_dim)
            prior_emb = self.fusion_projection(concatenated)
        else:
            prior_emb = self.pert_embedding(x_pert)
            
        h_pert = self.pert_encoder(prior_emb) 
        return self.decoder(h_pert)

# --- 3. Training & Evaluation ---

class Trainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss(reduction='none')
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_samples = 0
        for batch in dataloader:
            xp = batch['pert'].to(self.device)
            y = batch['y'].to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(xp)
            loss_elements = self.criterion(preds, y)
            if 'weight' in batch:
                w = batch['weight'].to(self.device)
                loss = (loss_elements * w).sum() / xp.size(0)
            else:
                loss = loss_elements.mean()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
        return total_loss / total_samples

    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_targets = [], []
        total_weighted_loss, total_mse_loss, total_samples = 0, 0, 0
        with torch.no_grad():
            for batch in dataloader:
                xp, y = batch['pert'].to(self.device), batch['y'].to(self.device)
                preds = self.model(xp)
                loss_elements = self.criterion(preds, y)
                total_mse_loss += loss_elements.mean().item() * y.size(0)
                if 'weight' in batch:
                    w = batch['weight'].to(self.device)
                    weighted_loss = (loss_elements * w).sum() / xp.size(0)
                    total_weighted_loss += weighted_loss.item() * y.size(0)
                else:
                    total_weighted_loss += loss_elements.mean().item() * y.size(0)
                total_samples += y.size(0)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        preds_np, targets_np = np.concatenate(all_preds), np.concatenate(all_targets)
        
        # Pearson per sample
        preds_mean = preds_np - preds_np.mean(axis=1, keepdims=True)
        targets_mean = targets_np - targets_np.mean(axis=1, keepdims=True)
        preds_norm = np.linalg.norm(preds_mean, axis=1)
        targets_norm = np.linalg.norm(targets_mean, axis=1)
        mask = (preds_norm > 1e-9) & (targets_norm > 1e-9)
        corrs = np.zeros(preds_np.shape[0])
        if np.any(mask):
            corrs[mask] = np.sum(preds_mean[mask] * targets_mean[mask], axis=1) / (preds_norm[mask] * targets_norm[mask])
        
        return {'mse': total_mse_loss / total_samples, 
                'weighted_loss': total_weighted_loss / total_samples, 
                'pearson': np.mean(corrs)}

    def predict(self, dataloader) -> np.ndarray:
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                preds = self.model(batch['pert'].to(self.device))
                all_preds.append(preds.cpu().numpy())
        return np.concatenate(all_preds)

def compute_dynamic_weights(dm: DataModule, train_indices: np.ndarray, all_indices: np.ndarray, use_wmse: bool = True) -> torch.Tensor:
    y_full = dm.all_y
    if not use_wmse:
        return torch.ones_like(y_full) / y_full.shape[1]
    
    suffix = dm.split_column.split('_', 1)[1] if '_' in dm.split_column else dm.split_column
    layer_name = f"t_scores_{suffix}"
    scores = dm.adata.layers[layer_name]
    if hasattr(scores, 'toarray'): scores = scores.toarray()
    weights = torch.abs(torch.tensor(scores, dtype=torch.float32))
    min_v, max_v = weights.min(dim=1, keepdim=True)[0], weights.max(dim=1, keepdim=True)[0]
    range_v = torch.clamp(max_v - min_v, min=1e-12)
    normalized = (weights - min_v) / range_v
    weights_squared = normalized**2
    return weights_squared / (torch.sum(weights_squared, dim=1, keepdim=True) + 1e-12)

def run_kfold_cv(dm, config, use_wmse=True, use_context_prior=True, device='cpu', n_folds=5, n_epochs=20, verbose=False):
    splits = dm.get_split_indices()
    train_indices = splits['train']
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores, fold_epochs = [], []
    all_indices = np.arange(len(dm.adata))
    
    for fold_i, (t_idx, v_idx) in enumerate(kf.split(train_indices)):
        global_t, global_v = train_indices[t_idx], train_indices[v_idx]
        weights = compute_dynamic_weights(dm, global_t, all_indices, use_wmse=use_wmse)
        train_loader = DataLoader(dm.create_dataset(global_t, weights=weights), batch_size=config['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(dm.create_dataset(global_v, weights=weights), batch_size=config['batch_size'], shuffle=False)
        
        model = MLPConcatModel(
            n_perts=len(dm.vocab_pert), output_dim=dm.output_dim,
            emb_dim=config['emb_dim'], hidden_dim=config['hidden_dim'], dropout=config['dropout'],
            knowledge_prior_tensor=dm.knowledge_prior_tensor,
            context_prior_tensor=dm.contextwise_loadings if use_context_prior else None
        )
        trainer = Trainer(model, device)
        for g in trainer.optimizer.param_groups: g['lr'] = config['lr']
        
        min_v_err, best_ep = 1e9, n_epochs
        pbar = tqdm(range(n_epochs), desc=f"Fold {fold_i+1}/{n_folds}", leave=False)
        for epoch in pbar:
            t_loss = trainer.train_epoch(train_loader)
            v_err = trainer.evaluate(val_loader)['weighted_loss']
            if v_err < min_v_err: 
                min_v_err, best_ep = v_err, epoch
            pbar.set_postfix({'T_Loss': f"{t_loss:.4f}", 'V_Err': f"{v_err:.4f}"})
        
        fold_scores.append(min_v_err)
        fold_epochs.append(best_ep+1)
        if verbose:
            print(f"    Fold {fold_i+1}: Best Val Error = {min_v_err:.4f} | Best Epoch = {best_ep+1}")

    return np.mean(fold_scores), np.max(fold_epochs)

def optimize_and_train(h5ad_path, target_screen, split_column, param_grid=None, 
                       use_pca=False, pca_dim=50, embedding_paths=None, 
                       embedding_pca_dim=128, use_wmse=True, 
                       use_data_prior=True, use_context_prior=True, device='cpu', 
                       n_folds=5, n_epochs=20, verbose=True):
    dm = DataModule(h5ad_path, target_screen, split_column)
    dm.load_and_preprocess(use_pca=use_pca, pca_dim=pca_dim, embedding_paths=embedding_paths, 
                            embedding_pca_dim=embedding_pca_dim, use_data_prior=use_data_prior)
    
    if not param_grid: param_grid = [{'emb_dim': 64, 'hidden_dim': 128, 'dropout': 0.2, 'lr': 1e-3, 'batch_size': 32}]
    
    best_score, best_config, best_ep = 1e9, None, n_epochs
    for i, config in enumerate(param_grid):
        print(f"[{i+1}/{len(param_grid)}] Testing Config: {config}")
        score, ep = run_kfold_cv(dm, config, use_wmse=use_wmse, use_context_prior=use_context_prior, 
                                  device=device, n_folds=n_folds, n_epochs=n_epochs, verbose=verbose)
        print(f"  -> Avg Val Error: {score:.4f}")
        if score < best_score: 
            best_score, best_config, best_ep = score, config, ep
            print("  -> New Best!")
    
    print(f"\nBest Config: {best_config}")
    print(f"Best CV Error: {best_score:.4f} | Best Epochs: {best_ep}")

    # Final Retraining
    print(f"\n--- Final Retraining on Full Train Set with Best Config over {best_ep} epochs ---")
    splits = dm.get_split_indices()
    train_idx = splits['train']
    weights = compute_dynamic_weights(dm, train_idx, np.arange(len(dm.adata)), use_wmse=use_wmse)
    train_loader = DataLoader(dm.create_dataset(train_idx, weights=weights), batch_size=best_config['batch_size'], shuffle=True, drop_last=True)
    
    final_model = MLPConcatModel(
        n_perts=len(dm.vocab_pert), output_dim=dm.output_dim,
        emb_dim=best_config['emb_dim'], hidden_dim=best_config['hidden_dim'], dropout=best_config['dropout'],
        knowledge_prior_tensor=dm.knowledge_prior_tensor,
        context_prior_tensor=dm.contextwise_loadings if use_context_prior else None
    )
    trainer = Trainer(final_model, device)
    for g in trainer.optimizer.param_groups: g['lr'] = best_config['lr']
    for _ in tqdm(range(best_ep), desc="Retraining"): trainer.train_epoch(train_loader)
    
    # Save Predictions
    test_idx = np.concatenate([splits['test_seen'], splits['test_unseen']])
    preds_all = trainer.predict(DataLoader(dm.create_dataset(test_idx), batch_size=best_config['batch_size']))
    final_preds = dm.pca_model.inverse_transform(preds_all) if dm.pca_model else preds_all
    
    test_obs = dm.adata.obs.iloc[test_idx][['target_gene', 'context', split_column]].copy().rename(columns={split_column: 'test_split'})
    pred_adata = sc.AnnData(X=final_preds, obs=test_obs, var=dm.adata.var.copy())
    
    # --- Metadata Tracking ---
    pred_adata.uns['best_hyperparams'] = best_config
    pred_adata.uns['training_params'] = {
        'n_epochs': best_ep,
        'pca_dim': pca_dim if use_pca else None,
        'n_folds': n_folds,
        'batch_size': best_config.get('batch_size', 32),
        'target_screen': target_screen,
        'split_column': split_column,
        'loss_type': 'WeightedMSE' if use_wmse else 'MSE',
        'use_knowledge_prior': (embedding_paths is not None and len(embedding_paths) > 0),
        'use_data_prior': use_data_prior,
        'use_context_prior': use_context_prior
    }

    tag = "WMSE" if use_wmse else "MSE"
    # Tagging: KP = Knowledge (Prior), DP = Data (Prior), KDP = both
    has_kp = (embedding_paths is not None and len(embedding_paths) > 0)
    has_dp = (dm.knowledge_prior_tensor is not None and dm.knowledge_prior_tensor.shape[0] > (len(embedding_paths) if embedding_paths else 0))
    
    if has_kp and not has_dp: tag2 = "KP"
    elif has_dp and not has_kp: tag2 = "DP"
    else: tag2 = "KDP"
        
    output_filename = f"./mlp_results/MLP_Concat_predictions__{split_column}__{tag}__{tag2}.h5ad"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    pred_adata.write_h5ad(output_filename)
    print(f"Saved predictions to {output_filename}")
    return output_filename

if __name__ == "__main__":
    param_grid = [
        {'emb_dim': 128,
         'hidden_dim': 128,
         'dropout': 0.0,
         'lr': 1e-3,
         'batch_size': 32},
    ]
    embedding_paths = [
        "presage_data/other_embeddings/GenePT_protein_embedding.pkl",
        "presage_data/other_embeddings/esm_emb_gene2esm.pkl",
        "presage_data/pathway_embeddings/c5.go.bp.v2023.2.Hs.symbols.pkl",
        "presage_data/pathway_embeddings/c2.cp.reactome.v2023.2.Hs.symbols.pkl",
        "presage_data/pathway_embeddings/c2.cp.pid.v2023.2.Hs.symbols.pkl",
        "presage_data/pathway_embeddings/c2.cp.wikipathways.v2023.2.Hs.symbols.pkl",
        "presage_data/other_embeddings/HeLa_HPLM.pkl",
        "presage_data/other_embeddings/HeLa_DMEM.pkl",
        "presage_data/other_embeddings/A549_CP186.pkl",
        "presage_data/other_embeddings/CRISPRGeneEffectDepMap.pkl",
        "presage_data/pathway_embeddings/c5.hpo.v2023.2.Hs.symbols.pkl",
        "presage_data/pathway_embeddings/h.all.v2023.2.Hs.symbols.pkl",
        "presage_data/pathway_embeddings/stringdb.human.highest.pkl",
    ]
    embedding_pca_dim = 128

    res_path = optimize_and_train(
        h5ad_path="./dataset.h5ad",
        target_screen="k562",
        split_column="k562_TF_0.1_UF_0.1_rs_1_random",
        param_grid=param_grid,
        embedding_paths=embedding_paths,
        embedding_pca_dim=embedding_pca_dim,
        use_wmse=False,
        use_data_prior=True,
        use_context_prior=True,
        n_epochs=100,
        n_folds=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
