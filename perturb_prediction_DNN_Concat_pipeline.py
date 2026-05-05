"""
Perturb-Prediction DNN Concat Pipeline
------------------------------------------
This script implements an 'Integrated' Deep Neural Network (DNN) pipeline for predicting 
transcriptional responses to genetic perturbations across multiple cellular contexts.

Key Features:
1. Multi-Context Learning: Trains on data from multiple screening contexts simultaneously.
2. FiLM-based Architecture: Uses Feature-wise Linear Modulation (FiLM) to inject context-specific 
   information into the gene perturbation embeddings.
3. Balanced Batch Sampling: Implements a stratified sampling strategy to ensure the model 
   sees a consistent ratio of target vs. non-target screen data during training.
4. Flexible Prior Integration: All gene-level knowledge priors are stacked into one tensor
   ``(num_sources, n_contexts, n_perts, D)``. Static paths are **replicated** across contexts;
   paths with ``{context}`` / ``{CONTEXT}`` load per-screen slices. Context-dependent PCA loadings
   remain a separate prior block in the fusion MLP.
5. Weighted MSE (WMSE): Optimized for predicting gene expression changes using a loss function 
   that can be weighted by the significance or magnitude of the perturbation-specific effects.
"""

import os, json
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

CONTEXT_PATH_MARKERS = ("{context}", "{CONTEXT}")


def _path_is_contextual(template: str) -> bool:
    return any(m in template for m in CONTEXT_PATH_MARKERS)


def _format_context_embedding_path(template: str, context_name: str) -> str:
    return template.replace("{CONTEXT}", str(context_name)).replace("{context}", str(context_name))


def _embedding_path_to_tensor(path: str, target_genes: List[str], target_p_dim: int) -> Optional[torch.Tensor]:
    """One json/pkl file → (n_perts, target_p_dim), genes aligned to ``target_genes``."""
    if not os.path.exists(path):
        return None
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame.from_dict(data, orient="index")
    df_aligned = df.reindex(target_genes)
    if not df.empty:
        df_aligned = df_aligned.fillna(df.mean(axis=0))
    else:
        df_aligned = df_aligned.fillna(0)
    raw_vals = df_aligned.values
    n_perts, d_orig = raw_vals.shape
    if d_orig > target_p_dim:
        pca = PCA(n_components=target_p_dim, random_state=42)
        reduced = pca.fit_transform(raw_vals)
        return torch.tensor(reduced, dtype=torch.float32)
    source_tensor = torch.tensor(raw_vals, dtype=torch.float32)
    if d_orig < target_p_dim:
        padding = torch.zeros((n_perts, target_p_dim - d_orig))
        source_tensor = torch.cat([source_tensor, padding], dim=1)
    return source_tensor


# --- 1. Data Handling ---

class PerturbationDataset(Dataset):
    def __init__(self, 
                 x_pert: torch.Tensor, 
                 x_context: torch.Tensor, 
                 y: torch.Tensor,
                 weights: Optional[torch.Tensor] = None,
                 obs_indices: Optional[List[str]] = None):
        """
        Args:
            x_pert: Tensor of shape (N,) matching indices in vocab.
            x_context: Tensor of shape (N,) matching indices in vocab.
            y: Tensor of shape (N, output_dim) - the response vector.
            weights: Tensor of shape (N, output_dim) - per-element weights.
            obs_indices: List of observation names (optional, for tracking).
        """
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
            'context': self.x_context[idx],
            'y': self.y[idx],
            'idx': idx
        }
        if self.weights is not None:
            item['weight'] = self.weights[idx]
        return item

class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, target_indices: List[int], other_indices: List[int], batch_size: int, target_frac: float):
        self.target_indices = np.array(target_indices)
        self.other_indices = np.array(other_indices)
        self.batch_size = batch_size
        self.target_frac = target_frac
        
        self.n_target_per_batch = int(batch_size * target_frac)
        self.n_other_per_batch = batch_size - self.n_target_per_batch
        
        self.n_batches = int(np.ceil(len(self.target_indices) / max(1, self.n_target_per_batch)))
        
    def __iter__(self):
        t_pool = np.random.permutation(self.target_indices)
        o_pool = np.random.permutation(self.other_indices)
        
        t_ptr = 0
        o_ptr = 0
        
        for _ in range(self.n_batches):
            batch = []
            for _ in range(self.n_target_per_batch):
                if t_ptr >= len(t_pool):
                    t_pool = np.random.permutation(self.target_indices)
                    t_ptr = 0
                batch.append(t_pool[t_ptr])
                t_ptr += 1
            for _ in range(self.n_other_per_batch):
                if o_ptr >= len(o_pool):
                    o_pool = np.random.permutation(self.other_indices)
                    o_ptr = 0
                batch.append(o_pool[o_ptr])
                o_ptr += 1
            yield np.random.permutation(batch).tolist()

    def __len__(self):
        return self.n_batches

class DataModule:
    def __init__(self, h5ad_path: str, target_screen: str, split_column: str = 'split_random'):
        self.h5ad_path = h5ad_path
        self.target_screen = target_screen
        self.split_column = split_column
        
        self.adata = None
        self.pca_model = None
        self.vocab_pert = {}
        self.vocab_context = {}
        
        self.pert_knowledge_tensor: Optional[torch.Tensor] = None
        self.contextwise_loadings = None
        self.output_dim = 0
        
    def load_and_preprocess(self, 
                            use_pca: bool = False, 
                            pca_dim: int = 50, 
                            embedding_paths: Optional[List[str]] = None,
                            embedding_pca_dim: int = 128,
                            in_context_only: bool = False,
                            seed: int = 42):
        print(f"Loading data from {self.h5ad_path}...")
        self.adata = sc.read_h5ad(self.h5ad_path)
        
        valid_splits = ['train', 'test_seen', 'test_unseen']
        mask = self.adata.obs[self.split_column].isin(valid_splits)
        if in_context_only:
            mask &= self.adata.obs['context'] == self.target_screen
        n_excluded = (~mask).sum()
        
        if n_excluded > 0:
            print(f"Discarding {n_excluded} rows from modeling...")
            self.adata = self.adata[mask].copy()
        
        all_perts = self.adata.obs['target_gene'].unique()
        all_contexts = self.adata.obs['context'].unique()
        self.vocab_pert = {p: i for i, p in enumerate(sorted(all_perts))}
        self.vocab_context = {c: i for i, c in enumerate(sorted(all_contexts))}
        
        print(f"Vocab sizes: Pert={len(self.vocab_pert)}, Context={len(self.vocab_context)}")
        
        # Step A: Load external gene embeddings → single (sources, contexts, perts, D) tensor
        if embedding_paths:
            self.pert_knowledge_tensor = self.load_external_embeddings(
                embedding_paths, embedding_pca_dim
            )
        else:
            self.pert_knowledge_tensor = None
        
        pert_indices = self.adata.obs['target_gene'].map(self.vocab_pert).values
        context_indices = self.adata.obs['context'].map(self.vocab_context).values
        
        # Step B: Load target gene expression data
        try:
            raw_y = self.adata.X.toarray()
        except:
            raw_y = self.adata.X
            
        if use_pca:
            print(f"Fitting PCA (n_components={pca_dim}) on TRAIN data only...")
            train_mask = self.adata.obs[self.split_column] == 'train'
            train_X = raw_y[train_mask]
            
            self.pca_model = PCA(n_components=pca_dim, random_state=seed)
            self.pca_model.fit(train_X)
            y_data = self.pca_model.transform(raw_y)
            self.output_dim = pca_dim
        else:
            y_data = raw_y
            self.output_dim = y_data.shape[1]
            
        print(f"Output Matrix Shape: {y_data.shape}")
        
        self.all_x_pert = torch.tensor(pert_indices, dtype=torch.long)
        self.all_x_context = torch.tensor(context_indices, dtype=torch.long)
        self.all_y = torch.tensor(y_data, dtype=torch.float32)
        self.all_obs_names = self.adata.obs_names.tolist()

        # Step C: Compute Context-Dependent PCA Feature Loadings (Additional Prior)
        print(f"Computing Context-Dependent PCA Feature Loadings (target_dim={embedding_pca_dim})...")
        n_perts = len(self.vocab_pert)
        n_contexts = len(self.vocab_context)
        loadings_tensor = torch.zeros((n_contexts, n_perts, embedding_pca_dim))

        train_mask = self.adata.obs[self.split_column] == 'train'
        var_to_idx = {name: i for i, name in enumerate(self.adata.var_names)}
        
        for context_name, c_idx in self.vocab_context.items():
            context_train_mask = train_mask & (self.adata.obs['context'] == context_name)
            if context_train_mask.sum() < 2:
                print(f"  Skipping {context_name}: insufficient training data for PCA.")
                continue
                
            print(f"  PCA for {context_name}...")
            # Use the raw expression data for PCA loadings
            try:
                sub_y = self.adata[context_train_mask].X.toarray()
            except:
                sub_y = self.adata[context_train_mask].X
                
            # Fit PCA to extract feature loadings
            # n_components is the embedding dimension
            pca_load = PCA(n_components=min(embedding_pca_dim, sub_y.shape[0], sub_y.shape[1]), random_state=seed)
            pca_load.fit(sub_y)
            
            # components_ shape: (n_components, n_features)
            # We want each perturbation (if it's a feature) to have a loading vector
            loadings = pca_load.components_.T # (n_features, n_components)
            
            # Map features to the perturbation vocabulary
            for pert_name, p_idx in self.vocab_pert.items():
                if pert_name in var_to_idx:
                    feat_idx = var_to_idx[pert_name]
                    v = torch.tensor(loadings[feat_idx], dtype=torch.float32)
                    if v.shape[0] < embedding_pca_dim:
                        v = torch.cat([v, torch.zeros(embedding_pca_dim - v.shape[0])])
                    loadings_tensor[c_idx, p_idx] = v
        
        self.contextwise_loadings = loadings_tensor
        print(f"Final Context-dependent PCA Loadings Tensor Shape: {loadings_tensor.shape}")
        
    def load_external_embeddings(
        self, embedding_paths: List[str], target_p_dim: int = 128
    ) -> Optional[torch.Tensor]:
        """
        Build one tensor ``(num_sources, n_contexts, n_perts, D)``.

        - **Static** paths: each file → ``(n_perts, D)``, then stacked to ``(S, P, D)`` and
          **expanded** to ``(S, C, P, D)`` (same embedding for every context).
        - **Contextual** paths (``{context}`` / ``{CONTEXT}``): each template →
          ``(C, P, D)``, stacked to ``(K, C, P, D)``.

        The two blocks are concatenated along the source (first) dimension. Missing contextual
        files leave zeros for that (template, context).
        """
        if not embedding_paths:
            return None

        target_genes = sorted(self.vocab_pert.keys(), key=lambda x: self.vocab_pert[x])
        n_c = len(self.vocab_context)
        n_p = len(self.vocab_pert)
        static_paths = [p for p in embedding_paths if not _path_is_contextual(p)]
        ctx_templates = [p for p in embedding_paths if _path_is_contextual(p)]

        processed_static: List[torch.Tensor] = []
        print(
            f"Loading {len(static_paths)} static + {len(ctx_templates)} contextual "
            f"gene embedding paths (PCA/pad to {target_p_dim})...",
            flush=True,
        )

        for path in static_paths:
            if not os.path.exists(path):
                print(f"Warning: Embedding path {path} not found. Skipping.", flush=True)
                continue
            st = _embedding_path_to_tensor(path, target_genes, target_p_dim)
            if st is None:
                continue
            print(f"  Static: {path} → {tuple(st.shape)}", flush=True)
            processed_static.append(st)

        blocks: List[torch.Tensor] = []
        if processed_static:
            static_stacked = torch.stack(processed_static, dim=0)
            static_expanded = static_stacked.unsqueeze(1).expand(-1, n_c, -1, -1).contiguous()
            print(
                f"  Static stack expanded to contexts: {tuple(static_expanded.shape)}",
                flush=True,
            )
            blocks.append(static_expanded)

        if ctx_templates:
            K = len(ctx_templates)
            out = torch.zeros((K, n_c, n_p, target_p_dim), dtype=torch.float32)
            ctx_order = sorted(self.vocab_context.items(), key=lambda kv: kv[1])
            any_loaded = False
            for k, tmpl in enumerate(ctx_templates):
                for ctx_name, c_idx in ctx_order:
                    concrete = _format_context_embedding_path(tmpl, ctx_name)
                    if not os.path.isfile(concrete):
                        print(f"  Missing (zeros): {concrete}", flush=True)
                        continue
                    t = _embedding_path_to_tensor(concrete, target_genes, target_p_dim)
                    if t is None:
                        continue
                    out[k, c_idx, :, :] = t
                    any_loaded = True
                    print(f"  Contextual [{ctx_name}]: {concrete} → {tuple(t.shape)}", flush=True)
            if any_loaded:
                print(f"  Contextual stack: {tuple(out.shape)}", flush=True)
                blocks.append(out)
            else:
                print("No contextual embedding files found; skipping contextual stack.", flush=True)

        if not blocks:
            return None
        merged = torch.cat(blocks, dim=0)
        print(f"Merged pert knowledge tensor: {tuple(merged.shape)} (sources × ctx × perts × dim)", flush=True)
        return merged
        
    def get_split_indices(self) -> Dict[str, np.ndarray]:
        indices = np.arange(len(self.adata))
        splits = {}
        for label in ['train', 'test_seen', 'test_unseen']:
            mask = self.adata.obs[self.split_column] == label
            splits[label] = indices[mask]
        train_indices = splits['train']
        train_mask_global = self.adata.obs[self.split_column] == 'train'
        target_mask = self.adata.obs['context'] == self.target_screen
        target_train_mask = train_mask_global & target_mask
        other_train_mask = train_mask_global & (~target_mask)
        splits['train_target'] = indices[target_train_mask]
        splits['train_other'] = indices[other_train_mask]
        print("\nSplit Sizes:")
        for k, v in splits.items():
            print(f"  {k}: {len(v)}")
        return splits

    def create_dataset(self, indices: np.ndarray, weights: Optional[torch.Tensor] = None) -> PerturbationDataset:
        return PerturbationDataset(
            self.all_x_pert[indices],
            self.all_x_context[indices],
            self.all_y[indices],
            weights[indices] if weights is not None else None,
            [self.all_obs_names[i] for i in indices]
        )

# --- 2. Model Architecture ---

class FiLMConcatModel(nn.Module):
    def __init__(self, 
                 n_perts: int, 
                 n_contexts: int, 
                 output_dim: int,
                 emb_dim: int = 64, 
                 hidden_dim: int = 256,
                 dropout: float = 0.2,
                 pert_knowledge_tensor: Optional[torch.Tensor] = None,
                 context_prior_tensor: Optional[torch.Tensor] = None):
        super(FiLMConcatModel, self).__init__()
        
        self.use_priors = pert_knowledge_tensor is not None or context_prior_tensor is not None
        
        if self.use_priors:
            self.pert_embedding = None
            num_knowledge = 0
            prior_dim = 0
            
            if pert_knowledge_tensor is not None:
                # (num_sources, n_contexts, n_perts, D)
                self.register_buffer("pert_knowledge", pert_knowledge_tensor)
                num_knowledge = int(pert_knowledge_tensor.shape[0])
                prior_dim = int(pert_knowledge_tensor.shape[3])
            else:
                self.pert_knowledge = None
                
            if context_prior_tensor is not None:
                self.register_buffer('context_priors', context_prior_tensor)
                num_context_sources = 1
                if prior_dim == 0:
                    prior_dim = int(context_prior_tensor.shape[2])
                elif int(context_prior_tensor.shape[2]) != prior_dim:
                    raise ValueError(
                        f"context PCA prior dim {context_prior_tensor.shape[2]} != prior_dim {prior_dim}"
                    )
            else:
                self.context_priors = None
                num_context_sources = 0

            self.num_total_sources = num_knowledge + num_context_sources
            self.fusion_projection = nn.Linear(self.num_total_sources * prior_dim, emb_dim)
        else:
            self.pert_embedding = nn.Embedding(n_perts, emb_dim)
            self.pert_knowledge = None
            self.context_priors = None
            self.fusion_projection = None

        # Added context embedding to learn context specific responses
        self.context_embedding = nn.Embedding(n_contexts, emb_dim)
        
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
        
        self.context_encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim) 
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x_pert, x_context):
        # x_pert: (batch_size,)
        # x_context: (batch_size,)
        
        if self.use_priors:
            priors_list = []
            
            if self.pert_knowledge is not None:
                for s in range(self.pert_knowledge.shape[0]):
                    priors_list.append(self.pert_knowledge[s, x_context, x_pert, :])
                
            if self.context_priors is not None:
                # Context-dependent prior: self.context_priors is (C, P, D)
                # Fetch [context_id, pert_id] for each sample in batch
                # Shape: (B, D)
                context_batch = self.context_priors[x_context, x_pert, :]
                priors_list.append(context_batch)
                
            # Concatenate all sources along feature dimension
            concatenated = torch.cat(priors_list, dim=1) # (B, S_total * D)
            pert_emb = self.fusion_projection(concatenated) # (B, emb_dim)
        else:
            # Use learnable latent embedding
            pert_emb = self.pert_embedding(x_pert) # (B, emb_dim)
            
        h_pert = self.pert_encoder(pert_emb)
        
        c_emb = self.context_embedding(x_context)
        film_params = self.context_encoder(c_emb)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        h_gated = (gamma * h_pert) + beta
        output = self.decoder(h_gated)
        return output

# --- 3. Training & Evaluation ---

class Trainer:
    def __init__(self, model, device='cpu', loss_type='mse'):
        self.model = model.to(device)
        self.device = device
        self.loss_type = loss_type.lower()
        
        # Use reduction='none' so we can apply weights manually
        loss_map = {
            'mse': nn.MSELoss(reduction='none'),
            'mae': nn.L1Loss(reduction='none'),
            'huber': nn.HuberLoss(reduction='none', delta=1.0)
        }
        self.criterion = loss_map.get(self.loss_type, nn.MSELoss(reduction='none'))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        for batch in dataloader:
            xp = batch['pert'].to(self.device)
            xc = batch['context'].to(self.device)
            y = batch['y'].to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(xp, xc)
            
            loss_elements = self.criterion(preds, y)
            
            if 'weight' in batch:
                w = batch['weight'].to(self.device)
                # Weighted average per sample, then mean across batch
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
        all_preds = []
        all_targets = []
        total_weighted_loss = 0
        total_mse_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                xp = batch['pert'].to(self.device)
                xc = batch['context'].to(self.device)
                y = batch['y'].to(self.device)
                
                preds = self.model(xp, xc)
                loss_elements = self.criterion(preds, y)
                
                # Standard MSE (always reported for consistency)
                mse_elements = (preds - y)**2
                total_mse_loss += mse_elements.mean().item() * y.size(0)
                
                # Weighted MSE if provided
                if 'weight' in batch:
                    w = batch['weight'].to(self.device)
                    # Weighted average per sample, then mean across batch
                    weighted_loss = (loss_elements * w).sum() / xp.size(0)
                    total_weighted_loss += weighted_loss.item() * y.size(0)
                else:
                    total_weighted_loss += loss_elements.mean().item() * y.size(0)
                
                total_samples += y.size(0)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                
        preds_np = np.concatenate(all_preds)
        targets_np = np.concatenate(all_targets)
        
        mse = total_mse_loss / total_samples
        weighted_loss_final = total_weighted_loss / total_samples
        
        preds_mean = preds_np - preds_np.mean(axis=1, keepdims=True)
        targets_mean = targets_np - targets_np.mean(axis=1, keepdims=True)
        preds_norm = np.linalg.norm(preds_mean, axis=1)
        targets_norm = np.linalg.norm(targets_mean, axis=1)
        mask = (preds_norm > 1e-9) & (targets_norm > 1e-9)
        corrs = np.zeros(preds_np.shape[0])
        if np.any(mask):
            corrs[mask] = np.sum(preds_mean[mask] * targets_mean[mask], axis=1) / (preds_norm[mask] * targets_norm[mask])
        mean_pearson = np.mean(corrs)
        
        return {'mse': mse, 'weighted_loss': weighted_loss_final, 'pearson': mean_pearson}

    def predict(self, dataloader) -> np.ndarray:
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                xp = batch['pert'].to(self.device)
                xc = batch['context'].to(self.device)
                preds = self.model(xp, xc)
                all_preds.append(preds.cpu().numpy())
        return np.concatenate(all_preds)

def compute_dynamic_weights(dm: DataModule, train_indices: np.ndarray, all_indices: np.ndarray, use_wmse: bool = True) -> torch.Tensor:
    """
    Computes per-sample weights based on pre-calculated scores from the h5ad layers.
    The layer name is 't_scores_' + suffix of dm.split_column.
    Steps per sample:
    1. Absolute value transformation: abs(scores)
    2. Min-max normalization per-sample across features.
    3. Squaring normalized weights.
    4. Row-wise sum-to-1 normalization.
    
    If use_wmse=False, returns uniform weights (1/output_dim).
    """
    y_full = dm.all_y
    if not use_wmse:
        print("Using uniform weights in Loss...")
        return torch.ones_like(y_full) / y_full.shape[1]
        '''
        print("Using DEG-restricted subset masking in Loss instead of uniform weights...")
        mask = torch.zeros_like(y_full)
        var_to_idx = {name: i for i, name in enumerate(dm.adata.var_names)}
        
        if 'top100_degs' in dm.adata.uns:
            screens = list(dm.adata.uns['top100_degs'].keys())
            for i in range(len(y_full)):
                pert = dm.adata.obs['target_gene'].iloc[i]
                ctx = str(dm.adata.obs['context'].iloc[i]).lower()
                
                screen_key = ctx if ctx in screens else (screens[0] if len(screens) == 1 else None)
                deg_list = []
                
                if screen_key and 'by_padj_0.05' in dm.adata.uns['top100_degs'][screen_key]:
                    degs_dict = dm.adata.uns['top100_degs'][screen_key]['by_padj_0.05']
                    if pert in degs_dict:
                        deg_list = degs_dict[pert]
                        
                valid_indices = [var_to_idx[g] for g in deg_list if g in var_to_idx]
                
                if valid_indices:
                    mask[i, valid_indices] = 1.0 / len(valid_indices)
                else:
                    mask[i, :] = 1.0 / y_full.shape[1]
        else:
            mask = torch.ones_like(y_full) / y_full.shape[1]
            
        return mask
        '''

    # Dynamically determine the layer name based on the split column
    suffix = dm.split_column.split('_', 1)[1] if '_' in dm.split_column else dm.split_column
    layer_name = f"t_scores_{suffix}"
    
    print(f"Computing weights from adata.layers['{layer_name}']...")
    if layer_name not in dm.adata.layers:
        raise ValueError(f"Layer '{layer_name}' not found in adata.layers. Available layers: {list(dm.adata.layers.keys())}")
    
    # Extract scores corresponding to the samples in dm.adata
    try:
        scores = dm.adata.layers[layer_name]
        if hasattr(scores, 'toarray'):
            scores = scores.toarray()
    except Exception as e:
        raise ValueError(f"Error accessing layer '{layer_name}': {e}")
    
    scores_torch = torch.tensor(scores, dtype=torch.float32)
    
    # 1. Absolute value transformation
    weights = torch.abs(scores_torch)
    
    # 2. Min-Max normalization per sample across features
    min_v = torch.min(weights, dim=1, keepdim=True)[0]
    max_v = torch.max(weights, dim=1, keepdim=True)[0]
    range_v = max_v - min_v
    
    # Handle rows where all scores are identical to avoid division by zero
    mask_constant = (range_v < 1e-12).squeeze()
    safe_range = torch.where(range_v > 1e-12, range_v, torch.ones_like(range_v))
    normalized = (weights - min_v) / safe_range
    
    # For constant rows, assign uniform weights
    if mask_constant.any():
        normalized[mask_constant] = 1.0 / weights.shape[1]
        
    # 3. Squaring the normalized weights
    weights_squared = normalized**2
    
    # 4. Final normalization: sum to 1 per sample
    row_sums = torch.sum(weights_squared, dim=1, keepdim=True)
    final_weights = weights_squared / (row_sums + 1e-12)
    
    return final_weights

def run_kfold_cv(dm: DataModule, 
                 config: Dict, 
                 use_wmse: bool = True,
                 n_folds: int = 5, 
                 n_epochs: int = 20, 
                 batch_size: int = 32, 
                 loss_type: str = 'mse',
                 use_context_prior: bool = True,
                 device: str = 'cpu',
                 verbose: bool = False) -> float:
    splits = dm.get_split_indices()
    target_train_indices = splits['train_target']
    other_train_indices = splits['train_other']
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []
    fold_epochs = []
    
    all_indices = np.arange(len(dm.adata))
    
    for fold_i, (train_idx_local, val_idx_local) in enumerate(kf.split(target_train_indices)):
        global_train_subset = target_train_indices[train_idx_local]
        global_val_subset = target_train_indices[val_idx_local]
        cv_train_indices = np.concatenate([global_train_subset, other_train_indices])
        
        # --- Dynamic Weights Calculation ---
        weights = compute_dynamic_weights(dm, cv_train_indices, all_indices, use_wmse=use_wmse)
        
        train_ds = dm.create_dataset(cv_train_indices, weights=weights)
        val_ds = dm.create_dataset(global_val_subset, weights=weights)
        
        current_batch_size = config.get('batch_size', batch_size)
        target_batch_frac = config.get('target_batch_frac', None)
        
        if target_batch_frac is not None and target_batch_frac > 0:
            rel_target_idx = list(range(len(global_train_subset)))
            rel_other_idx = list(range(len(global_train_subset), len(cv_train_indices)))
            sampler = BalancedBatchSampler(rel_target_idx, rel_other_idx, current_batch_size, target_batch_frac)
            train_loader = DataLoader(train_ds, batch_sampler=sampler)
        else:
            train_loader = DataLoader(train_ds, batch_size=current_batch_size, shuffle=True, drop_last=True)
            
        val_loader = DataLoader(val_ds, batch_size=current_batch_size, shuffle=False)
        
        model = FiLMConcatModel(
            n_perts=len(dm.vocab_pert),
            n_contexts=len(dm.vocab_context),
            output_dim=dm.output_dim,
            emb_dim=config.get('emb_dim', 64),
            hidden_dim=config.get('hidden_dim', 256),
            dropout=config.get('dropout', 0.2),
            pert_knowledge_tensor=dm.pert_knowledge_tensor,
            context_prior_tensor=dm.contextwise_loadings if use_context_prior else None,
        )
        
        trainer = Trainer(model, device, loss_type=loss_type)
        if 'lr' in config:
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = config['lr']
        
        min_val_err = 1e9
        best_epoch = n_epochs
        
        # Progress bar for Epochs
        pbar = tqdm(range(n_epochs), desc=f"Fold {fold_i+1}/{n_folds}", leave=False)
        for epoch in pbar:
            train_loss = trainer.train_epoch(train_loader)
            eval_metrics = trainer.evaluate(val_loader)
            
            val_err = eval_metrics['weighted_loss']
            if val_err < min_val_err:
                min_val_err = val_err
                best_epoch = epoch
            
            pbar.set_postfix({'T_Loss': f"{train_loss:.4f}", 'V_Err': f"{val_err:.4f}"})
        
        fold_scores.append(min_val_err)
        fold_epochs.append(best_epoch+1)
        if verbose:
            print(f"    Fold {fold_i+1}: Best Val Error = {min_val_err:.4f} | Best Epoch = {best_epoch+1}")
            
    return np.mean(fold_scores), np.max(fold_epochs)


def _print_model_io_shapes(dm: DataModule, use_context_prior: bool) -> None:
    """Print fused prior concat width and supervision output shape."""
    prior_d = None
    n_k = 0
    if dm.pert_knowledge_tensor is not None:
        n_k = int(dm.pert_knowledge_tensor.shape[0])
        prior_d = int(dm.pert_knowledge_tensor.shape[3])
    n_c = 1 if (use_context_prior and dm.contextwise_loadings is not None) else 0
    if prior_d is None and n_c:
        prior_d = int(dm.contextwise_loadings.shape[2])
    if prior_d is not None:
        d_in = (n_k + n_c) * prior_d
        print(f"Fused prior input (per batch row): (batch, {d_in})", flush=True)
    else:
        print("Fused prior input (per batch row): (batch, emb_dim) [learnable pert emb]", flush=True)
    print(f"Output target shape: {tuple(dm.all_y.shape)}", flush=True)


def optimize_and_train(h5ad_path: str,
                       target_screen: str = 'K562',
                       split_column: str = 'split_random',
                       param_grid: Optional[List[Dict]] = None,
                       use_pca: bool = False,
                       pca_dim: int = 50,
                       n_folds: int = 5,
                       n_epochs: int = 20,
                       batch_size: int = 32,
                       embedding_paths: Optional[List[str]] = None,
                       embedding_pca_dim: int = 128,
                       in_context_only: bool = False,
                       use_wmse: bool = True,
                       use_context_prior: bool = True,
                       loss_type: str = 'mse',
                       device: str = 'cpu'):
    dm = DataModule(h5ad_path, target_screen, split_column)
    dm.load_and_preprocess(
        use_pca=use_pca, 
        pca_dim=pca_dim,
        embedding_paths=embedding_paths,
        embedding_pca_dim=embedding_pca_dim,
        in_context_only=in_context_only
    )
    _print_model_io_shapes(dm, use_context_prior=use_context_prior)

    if param_grid is None:
        param_grid = [
            {'emb_dim': 64, 'hidden_dim': 128, 'dropout': 0.2, 'lr': 1e-3, 'target_batch_frac': None, 'batch_size': 32},
        ]
        
    print(f"\n--- Starting Hyperparameter Optimization over {len(param_grid)} configs ---")
    
    best_score = 1e9
    best_config = None
    best_epochs = n_epochs
    
    for i, config in enumerate(param_grid):
        print(f"[{i+1}/{len(param_grid)}] Testing Config: {config}")
        score, epochs = run_kfold_cv(dm, config, use_wmse=use_wmse, use_context_prior=use_context_prior, loss_type=loss_type, n_folds=n_folds, n_epochs=n_epochs, batch_size=batch_size, device=device, verbose=True)
        print(f"  -> Avg Val Error: {score:.4f}")
        
        if score < best_score:
            best_score = score
            best_config = config
            best_epochs = epochs
            print("  -> New Best!")
            
    print(f"\nBest Config: {best_config}")
    print(f"Best CV Error: {best_score:.4f} | Best Epochs: {best_epochs}")
    
    # --- Final Retraining ---
    print(f"\n--- Final Retraining on Full Train Set with Best Config over {best_epochs} epochs ---")
    splits = dm.get_split_indices()
    full_train_indices = splits['train']
    
    # Dynamic Weights for Final Run
    all_indices = np.arange(len(dm.adata))
    weights = compute_dynamic_weights(dm, full_train_indices, all_indices, use_wmse=use_wmse)
    
    train_ds = dm.create_dataset(full_train_indices, weights=weights)
    target_batch_frac = best_config.get('target_batch_frac', None)
    current_batch_size = best_config.get('batch_size', batch_size)
    
    if target_batch_frac is not None and target_batch_frac > 0:
        target_train_global = splits['train_target']
        is_target = np.isin(full_train_indices, target_train_global)
        rel_target_idx = np.where(is_target)[0].tolist()
        rel_other_idx = np.where(~is_target)[0].tolist()
        sampler = BalancedBatchSampler(rel_target_idx, rel_other_idx, current_batch_size, target_batch_frac)
        train_loader = DataLoader(train_ds, batch_sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=current_batch_size, shuffle=True, drop_last=True)
    
    final_model = FiLMConcatModel(
        n_perts=len(dm.vocab_pert),
        n_contexts=len(dm.vocab_context),
        output_dim=dm.output_dim,
        emb_dim=best_config.get('emb_dim', 64),
        hidden_dim=best_config['hidden_dim'],
        dropout=best_config['dropout'],
        pert_knowledge_tensor=dm.pert_knowledge_tensor,
        context_prior_tensor=dm.contextwise_loadings if use_context_prior else None,
    )
    
    trainer = Trainer(final_model, device, loss_type=loss_type)
    if 'lr' in best_config:
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = best_config['lr']
    
    pbar = tqdm(range(best_epochs), desc="Retraining", leave=False)
    for epoch in pbar:
        loss = trainer.train_epoch(train_loader)
        pbar.set_postfix({'Loss': f"{loss:.4f}"})
            
    # --- Final Evaluation ---
    print("\n--- Final Evaluation ---")
    test_seen_indices = splits['test_seen']
    test_unseen_indices = splits['test_unseen']
    
    test_seen_ds = dm.create_dataset(test_seen_indices, weights=weights)
    test_unseen_ds = dm.create_dataset(test_unseen_indices, weights=weights)
    
    seen_loader = DataLoader(test_seen_ds, batch_size=current_batch_size, shuffle=False)
    unseen_loader = DataLoader(test_unseen_ds, batch_size=current_batch_size, shuffle=False)
    
    metrics_seen = trainer.evaluate(seen_loader)
    print(f"Test Seen: MSE={metrics_seen['mse']:.4f}, Weighted_Loss={metrics_seen['weighted_loss']:.4f}, Pearson={metrics_seen['pearson']:.4f}")
    
    metrics_unseen = trainer.evaluate(unseen_loader)
    print(f"Test Unseen: MSE={metrics_unseen['mse']:.4f}, Weighted_Loss={metrics_unseen['weighted_loss']:.4f}, Pearson={metrics_unseen['pearson']:.4f}")

    # --- Save Predictions ---
    print("\n--- Saving Predictions ---")
    preds_all_test = trainer.predict(DataLoader(dm.create_dataset(np.concatenate([test_seen_indices, test_unseen_indices])), batch_size=current_batch_size, shuffle=False))
    
    if dm.pca_model is not None:
        print("Inverse transforming predictions from PCA space to gene space...")
        final_preds = dm.pca_model.inverse_transform(preds_all_test)
    else:
        final_preds = preds_all_test
        
    all_test_idx = np.concatenate([test_seen_indices, test_unseen_indices])
    test_obs = dm.adata.obs.iloc[all_test_idx][['target_gene', 'context', split_column]].copy()
    test_obs = test_obs.rename(columns={split_column: 'test_split'})
    if hasattr(test_obs['test_split'], 'cat'):
        test_obs['test_split'] = test_obs['test_split'].cat.remove_unused_categories()
    
    pred_adata = sc.AnnData(X=final_preds, obs=test_obs, var=dm.adata.var.copy())
    pred_adata.uns['best_hyperparams'] = best_config
    pred_adata.uns['training_params'] = {
        'n_epochs': best_epochs,
        'pca_dim': pca_dim if use_pca else None,
        'target_screen': target_screen,
        'split_column': split_column,
        'n_folds': n_folds,
        'batch_size': current_batch_size,
        'target_batch_frac': target_batch_frac,
        'loss_type': f"{'Weighted' if use_wmse else ''}{loss_type.upper()}"
    }
    pred_adata.uns['test_metrics'] = {'seen': metrics_seen, 'unseen': metrics_unseen}
    
    tag = f"{'W' if use_wmse else ''}{loss_type.upper()}"
    
    if in_context_only:
        tag2 = "KP"
    elif embedding_paths is None:
        tag2 = "DP"
    else:
        tag2 = "KDP"
        
    output_filename = f"./mlp_results/integrated_DNN_Concat_predictions__{split_column}__{tag}__{tag2}.h5ad"
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
        'target_batch_frac': None, 
        'batch_size': 32
        },
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
        h5ad_path= "./dataset.h5ad", # 
        target_screen="k562",
        split_column="k562_TF_0.1_UF_0.1_rs_1_random",
        param_grid=param_grid,
        embedding_paths=embedding_paths,
        embedding_pca_dim=embedding_pca_dim,
        in_context_only=False, # False if using cross-context training data
        use_wmse=False, # True for WMSE, False for MSE
        use_context_prior=True,
        loss_type='mse',
        n_epochs=100,
        n_folds=5,
        batch_size=32,
        use_pca=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
