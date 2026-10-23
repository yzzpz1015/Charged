# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/11 2:56
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/11 2:56

"""
Module defining PredictionTrainer for training and evaluating prediction models.

This module provides a comprehensive training framework for both statistical
and neural models for electric vehicle charging demand prediction. It handles
training loops, validation, model checkpointing, and evaluation with multiple
regression metrics.

Key Features:
    - Supports both statistical models (LO, AR, ARIMA) and neural models
    - Implements training and validation loops with early stopping
    - Provides model checkpointing and loading
    - Computes comprehensive regression metrics
    - Saves predictions, labels, and evaluation results

Classes:
    PredictionTrainer: Manages training loops, validation, testing, and metric logging.
"""

# Standard library imports
import json
import os
from typing import Optional, Any

# Third-party imports
import numpy as np
import torch
from tqdm import tqdm

# Local imports
from api.utils import calculate_regression_metrics, convert_numpy


def _inverse_transform_with_scaler(scaler: Any, array: np.ndarray) -> np.ndarray:
    """Apply sklearn-style inverse_transform to 2D or 3D forecast arrays."""
    if array.ndim <= 2:
        return scaler.inverse_transform(array)

    original_shape = array.shape
    reshaped = array.transpose(0, 2, 1).reshape(-1, original_shape[1])
    restored = scaler.inverse_transform(reshaped)
    return restored.reshape(original_shape[0], original_shape[2], original_shape[1]).transpose(0, 2, 1)


class PredictionTrainer(object):
    """
    Trainer class for prediction models on EV charging demand data.

    This class provides a unified interface for training and evaluating both
    statistical and neural models. It handles the complete training pipeline
    including data loading, model training, validation, checkpointing, and
    evaluation with comprehensive metrics.

    The trainer automatically detects model type and configures appropriate
    training strategies. Statistical models use direct prediction methods,
    while neural models use gradient-based optimization with validation.

    Attributes:
        ev_dataset (Any): Dataset object containing DataLoaders or raw arrays.
        ev_model (Any): PredictionModel instance with `.model` and `.model_name`.
        is_train (bool): Flag indicating training mode.
        stat_model (bool): True if using statistical model (no optimizer).
        optim (Optional[torch.optim.Optimizer]): Optimizer for neural models.
        loss_func (Optional[torch.nn.Module]): Loss function for neural models.
        save_path (str): Directory to save checkpoints and outputs.
        test_loader (Any): Raw arrays or DataLoader for test data.
        train_valid_feat (Optional[np.ndarray]): Combined data for statistical models.
    """

    def __init__(
            self,
            dataset: Any,
            model: Any,
            seq_l: int,
            pre_len: int,
            is_train: bool,
            save_path: str,
            lr: float = 1e-3,
            weight_decay: float = 1e-5,
            adj_matrix: Optional[torch.Tensor] = None,
            early_stop_patience: int = 0,
            ev_vlm_mode: str = 'staged',
            use_bf16: bool = True,
    ) -> None:
        """
        Initialize PredictionTrainer with dataset, model, and configuration.

        This method sets up the training environment based on the model type.
        For statistical models, it prepares the data arrays and disables training.
        For neural models, it initializes the optimizer and loss function.

        Args:
            dataset (Any): EVDataset or similar with train_loader, valid_loader, test_loader.
            model (Any): PredictionModel containing `.model` and `.model_name`.
            seq_l (int): Input sequence length for statistical models.
            pre_len (int): Prediction horizon length for statistical models.
            is_train (bool): Flag to enable training mode.
            save_path (str): Directory path to save model and results.
            lr (float): Learning rate for neural optimizers.
            weight_decay (float): Weight decay coefficient for neural optimizers.
            adj_matrix (torch.Tensor, optional): Adjacency matrix for GWNet/ASTGCN models.
        """
        self.ev_dataset = dataset
        self.ev_model = model
        self.save_path = save_path
        self.is_train = is_train
        self.adj_matrix = adj_matrix  # Store adjacency matrix for GWNet/ASTGCN
        self.early_stop_patience = early_stop_patience
        self.ev_vlm_mode = ev_vlm_mode  # 'staged' or 'end_to_end'
        self.use_bf16 = use_bf16  # bf16 autocast for the ev_vlm LLM backbone

        # Set adjacency matrix for GWNet/ASTGCN if it's not already set
        if adj_matrix is not None and model.model_name in ('gwnet', 'astgcn'):
            self.ev_model.model.adj_matrix = adj_matrix
        
        # Load distance matrix for Hyperdist model
        if model.model_name in ('hyperdist', 'frehyper'):
            dataset.load_distance_and_adjacency()
            # Set distance matrix to model
            if hasattr(dataset, 'distance_matrix'):
                dist_matrix = torch.from_numpy(dataset.distance_matrix).float()
                model.model.set_distance_matrix(dist_matrix)

        # Load satellite embeddings for SatHyper
        if model.model_name == 'sathyper':
            import numpy as np
            import pandas as pd
            city = dataset.city if hasattr(dataset, 'city') else 'MEL'
            # Load site coordinates
            sites_info = pd.read_csv(os.path.join(dataset.data_path, 'sites.csv'))
            site_col = 'site' if 'site' in sites_info.columns else 'site_id'
            col_names = [str(c) for c in dataset.feat_columns]
            sites_info[site_col] = sites_info[site_col].astype(str)
            sites_info = sites_info.set_index(site_col)
            sites_info = sites_info.loc[col_names]
            coords = torch.from_numpy(sites_info[['latitude', 'longitude']].values).float()
            # Load satellite embeddings
            embed_path = os.path.join(dataset.tiles_path, city, 'embedding', 'z16', 'sat_embed_dinov2_768.npy')
            sat_embed_np = np.load(embed_path)
            N = len(col_names)
            # Handle size mismatch (some sites may lack satellite tiles)
            if sat_embed_np.shape[0] < N:
                pad = np.zeros((N - sat_embed_np.shape[0], sat_embed_np.shape[1]))
                sat_embed_np = np.concatenate([sat_embed_np, pad], axis=0)
            sat_embed = torch.from_numpy(sat_embed_np[:N]).float()
            device = next(model.model.parameters()).device
            model.model.setup_hypergraph(coords, sat_embed, device)

        # Configure training strategy based on model type
        if model.model_name in ('lo', 'ar', 'arima'):
            # Statistical models: no optimizer or loss function needed
            self.optim = None
            self.loss_func = None
            self.is_train = False
            self.stat_model = True
            self.train_stage = None  # Not applicable

            # Prepare combined training and validation data for statistical prediction
            self.train_valid_feat = np.vstack(
                (dataset.train_feat, dataset.valid_feat,
                 dataset.test_feat[: seq_l + pre_len, :])  # Include initial test data for context
            )
            # Test features and labels for statistical prediction
            self.test_loader = [
                self.train_valid_feat,  # Input features (train + valid + initial test)
                dataset.test_feat[pre_len + seq_l:, :]  # Target labels (remaining test data)
            ]
        elif model.model_name == 'ev_vlm':
            # EV-VLM: single-stage, lower LR + cosine annealing
            self.ev_lr = lr * 0.1  # 1e-4 instead of 1e-3 for stability
            self.optim = torch.optim.Adam(
                model.model.parameters(), lr=self.ev_lr, weight_decay=weight_decay
            )
            self.stat_model = False
            self.loss_func = torch.nn.HuberLoss(delta=1.0)
            self.train_stage = None
            self.scheduler = None  # created in training() once `epoch` is known
        else:
            # Neural models: use Adam optimizer + Huber loss.
            self.optim = torch.optim.Adam(
                model.model.parameters(), lr=lr, weight_decay=weight_decay
            )
            self.loss_func = torch.nn.HuberLoss(delta=1.0)
            self.stat_model = False
            self.train_stage = None

    def _autocast(self):
        """bf16 autocast context for the ev_vlm LLM backbone (no-op when disabled).

        bf16 has the same exponent range as fp32, so no GradScaler is needed --
        the heavy GPT-2 matmuls simply run in bf16, halving activation memory and
        ~1.7x throughput, while the final output is promoted back to fp32.
        """
        return torch.cuda.amp.autocast(enabled=self.use_bf16, dtype=torch.bfloat16)

    def training(self, epoch: int) -> None:
        """
        Train and validate neural model over multiple epochs.

        This method implements the complete training loop for neural models,
        including training epochs, validation, and model checkpointing based
        on validation loss. It saves the best model based on validation performance.

        Args:
            epoch (int): Number of training epochs to run.
        """
        import sys as _sys
        best_val_loss = float('inf')  # Track best validation loss for model saving

        if self.ev_model.model_name == 'satcrossattn':
            import numpy as np
            city = self.ev_dataset.city if hasattr(self.ev_dataset, 'city') else 'MEL'
            embed_path = os.path.join(self.ev_dataset.tiles_path, city, 'embedding', 'z16', 'sat_embed_dinov2_768.npy')
            site_embeddings = np.load(embed_path)
            device = next(self.ev_model.model.parameters()).device
            self.ev_model.model.setup_spatial(site_embeddings, device)

        elif self.ev_model.model_name == 'frets_spatial':
            import pandas as pd
            city = self.ev_dataset.city if hasattr(self.ev_dataset, 'city') else 'MEL'
            sites_info = pd.read_csv(os.path.join(self.ev_dataset.data_path, 'sites.csv'))
            col_names = [str(c) for c in self.ev_dataset.feat_columns]
            site_to_pos = {int(c): i for i, c in enumerate(col_names)}

            # Hilbert curve ordering: map 2D coords to 1D preserving locality
            def xy2d(n, x, y):
                d = 0
                s = n // 2
                while s > 0:
                    rx = 1 if (x & s) > 0 else 0
                    ry = 1 if (y & s) > 0 else 0
                    d += s * s * ((3 * rx) ^ ry)
                    if ry == 0:
                        if rx == 1:
                            x = s - 1 - x
                            y = s - 1 - y
                        x, y = y, x
                    s //= 2
                return d

            import numpy as np
            lat = sites_info['latitude'].values
            lon = sites_info['longitude'].values
            lat_n = ((lat - lat.min()) / (lat.max() - lat.min()) * 15).astype(int).clip(0, 15)
            lon_n = ((lon - lon.min()) / (lon.max() - lon.min()) * 15).astype(int).clip(0, 15)
            sites_info = sites_info.copy()
            sites_info['hilbert_d'] = [xy2d(16, int(x), int(y)) for x, y in zip(lon_n, lat_n)]
            hilbert_sorted = sites_info.sort_values('hilbert_d')
            hilbert_positions = [site_to_pos[sid] for sid in hilbert_sorted['site'].tolist() if sid in site_to_pos]

            device = next(self.ev_model.model.parameters()).device
            self.ev_model.model.setup_spatial(hilbert_positions, device)

        elif self.ev_model.model_name == 'frets_mixer':
            import pandas as pd, numpy as np
            sites_info = pd.read_csv(os.path.join(self.ev_dataset.data_path, 'sites.csv'))
            col_names = [str(c) for c in self.ev_dataset.feat_columns]
            site_to_pos = {int(c): i for i, c in enumerate(col_names)}

            # Hilbert 曲线排序
            def xy2d(n, x, y):
                d = 0
                s = n // 2
                while s > 0:
                    rx = 1 if (x & s) > 0 else 0
                    ry = 1 if (y & s) > 0 else 0
                    d += s * s * ((3 * rx) ^ ry)
                    if ry == 0:
                        if rx == 1:
                            x = s - 1 - x
                            y = s - 1 - y
                        x, y = y, x
                    s //= 2
                return d

            lat = sites_info['latitude'].values
            lon = sites_info['longitude'].values
            lat_n = ((lat - lat.min()) / (lat.max() - lat.min()) * 15).astype(int).clip(0, 15)
            lon_n = ((lon - lon.min()) / (lon.max() - lon.min()) * 15).astype(int).clip(0, 15)
            sites_info_copy = sites_info.copy()
            sites_info_copy['hilbert_d'] = [xy2d(16, int(x), int(y)) for x, y in zip(lon_n, lat_n)]
            hilbert_sorted = sites_info_copy.sort_values('hilbert_d')
            hilbert_positions = [site_to_pos[sid] for sid in hilbert_sorted['site'].tolist() if sid in site_to_pos]

            # 纬度排序
            lat_sorted = sites_info.sort_values('latitude')
            lat_positions = [site_to_pos[sid] for sid in lat_sorted['site'].tolist() if sid in site_to_pos]

            device = next(self.ev_model.model.parameters()).device
            self.ev_model.model.setup_spatial([hilbert_positions, lat_positions], device)

        elif self.ev_model.model_name == 'dlfre_spatial':
            import pandas as pd
            city = self.ev_dataset.city if hasattr(self.ev_dataset, 'city') else 'MEL'
            sites_info = pd.read_csv(os.path.join(self.ev_dataset.data_path, 'sites.csv'))
            col_names = [str(c) for c in self.ev_dataset.feat_columns]
            site_to_pos = {int(c): i for i, c in enumerate(col_names)}

            def xy2d(n, x, y):
                d = 0
                s = n // 2
                while s > 0:
                    rx = 1 if (x & s) > 0 else 0
                    ry = 1 if (y & s) > 0 else 0
                    d += s * s * ((3 * rx) ^ ry)
                    if ry == 0:
                        if rx == 1:
                            x = s - 1 - x
                            y = s - 1 - y
                        x, y = y, x
                    s //= 2
                return d

            import numpy as np
            lat = sites_info['latitude'].values
            lon = sites_info['longitude'].values
            lat_n = ((lat - lat.min()) / (lat.max() - lat.min()) * 15).astype(int).clip(0, 15)
            lon_n = ((lon - lon.min()) / (lon.max() - lon.min()) * 15).astype(int).clip(0, 15)
            sites_info = sites_info.copy()
            sites_info['hilbert_d'] = [xy2d(16, int(x), int(y)) for x, y in zip(lon_n, lat_n)]
            hilbert_sorted = sites_info.sort_values('hilbert_d')
            hilbert_positions = [site_to_pos[sid] for sid in hilbert_sorted['site'].tolist() if sid in site_to_pos]

            device = next(self.ev_model.model.parameters()).device
            self.ev_model.model.setup_spatial(hilbert_positions, device)

        elif self.ev_model.model_name == 'frets_mixer':
            import pandas as pd, numpy as np
            city = self.ev_dataset.city if hasattr(self.ev_dataset, 'city') else 'MEL'
            sites_info = pd.read_csv(os.path.join(self.ev_dataset.data_path, 'sites.csv'))
            col_names = [str(c) for c in self.ev_dataset.feat_columns]
            site_to_pos = {int(c): i for i, c in enumerate(col_names)}

            # Hilbert 曲线排序
            def xy2d(n, x, y):
                d = 0
                s = n // 2
                while s > 0:
                    rx = 1 if (x & s) > 0 else 0
                    ry = 1 if (y & s) > 0 else 0
                    d += s * s * ((3 * rx) ^ ry)
                    if ry == 0:
                        if rx == 1:
                            x = s - 1 - x
                            y = s - 1 - y
                        x, y = y, x
                    s //= 2
                return d

            lat = sites_info['latitude'].values
            lon = sites_info['longitude'].values
            lat_n = ((lat - lat.min()) / (lat.max() - lat.min()) * 15).astype(int).clip(0, 15)
            lon_n = ((lon - lon.min()) / (lon.max() - lon.min()) * 15).astype(int).clip(0, 15)
            sites_info = sites_info.copy()
            sites_info['hilbert_d'] = [xy2d(16, int(x), int(y)) for x, y in zip(lon_n, lat_n)]
            hilbert_sorted = sites_info.sort_values('hilbert_d')
            hilbert_positions = [site_to_pos[sid] for sid in hilbert_sorted['site'].tolist() if sid in site_to_pos]

            # 纬度排序
            lat_sorted = sites_info.sort_values('latitude')
            lat_positions = [site_to_pos[sid] for sid in lat_sorted['site'].tolist() if sid in site_to_pos]

            device = next(self.ev_model.model.parameters()).device
            self.ev_model.model.setup_spatial(hilbert_positions, lat_positions, device)

            import pandas as pd, numpy as np
            sites_info = pd.read_csv(os.path.join(self.ev_dataset.data_path, 'sites.csv'))
            col_names = [str(c) for c in self.ev_dataset.feat_columns]
            site_to_pos = {int(c): i for i, c in enumerate(col_names)}
            N = len(col_names)

            if self.ev_model.model_name == 'frets_multispatial_random':
                # Random orderings (fixed seed for reproducibility)
                rng = np.random.RandomState(42)
                orderings = [rng.permutation(N).tolist() for _ in range(4)]
            else:
                # Geographic orderings
                # Normalize site column name (some cities use 'site_id' instead of 'site')
                if 'site' not in sites_info.columns and 'site_id' in sites_info.columns:
                    sites_info = sites_info.rename(columns={'site_id': 'site'})

                def xy2d(n, x, y):
                    d = 0
                    s = n // 2
                    while s > 0:
                        rx = 1 if (x & s) > 0 else 0
                        ry = 1 if (y & s) > 0 else 0
                        d += s * s * ((3 * rx) ^ ry)
                        if ry == 0:
                            if rx == 1:
                                x = s - 1 - x
                                y = s - 1 - y
                            x, y = y, x
                        s //= 2
                    return d

                lat = sites_info['latitude'].values
                lon = sites_info['longitude'].values
                # Filter to only sites in dataset's feat_columns
                lat_order = [site_to_pos[s] for s in sites_info.sort_values('latitude')['site'].tolist() if s in site_to_pos]
                lon_order = [site_to_pos[s] for s in sites_info.sort_values('longitude')['site'].tolist() if s in site_to_pos]
                lat_n = ((lat - lat.min()) / (lat.max() - lat.min()) * 15).astype(int).clip(0, 15)
                lon_n = ((lon - lon.min()) / (lon.max() - lon.min()) * 15).astype(int).clip(0, 15)
                sites_copy = sites_info.copy()
                sites_copy['hilbert_d'] = [xy2d(16, int(x), int(y)) for x, y in zip(lon_n, lat_n)]
                hilbert_order = [site_to_pos[s] for s in sites_copy.sort_values('hilbert_d')['site'].tolist() if s in site_to_pos]
                centroid_lat, centroid_lon = lat.mean(), lon.mean()
                dists = np.sqrt((lat - centroid_lat)**2 + (lon - centroid_lon)**2)
                sites_info_copy = sites_info.copy()
                sites_info_copy['centroid_dist'] = dists
                anchor_order = [site_to_pos[s] for s in sites_info_copy.sort_values('centroid_dist')['site'].tolist() if s in site_to_pos]
                if self.ev_model.model_name == 'frets_multispatial_2':
                    orderings = [hilbert_order, lat_order]
                else:
                    orderings = [lat_order, lon_order, hilbert_order, anchor_order]

            device = next(self.ev_model.model.parameters()).device
            self.ev_model.model.setup_spatial(orderings, device)

        self.ev_model.model.train()

        # Single-stage: create cosine scheduler now that `epoch` is known
        if self.ev_model.model_name == 'ev_vlm' and hasattr(self, 'ev_lr'):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim, T_max=epoch, eta_min=self.ev_lr * 0.01
            )

        # Training loop over specified epochs
        early_stop_counter = 0
        for current_epoch in tqdm(range(epoch), desc='Training'):
            train_total = 0.0
            train_count = 0
            grad_norms = []
            for batch in self.ev_dataset.train_loader:
                self.optim.zero_grad()

                if self.ev_model.model_name in ('gwnet', 'astgcn'):
                    feat, label = batch[0], batch[1]
                    preds = self.ev_model.model(feat)
                elif self.ev_model.model_name == 'hyperdist':
                    feat, label, extra = batch
                    if self.ev_dataset.extra_feat is None:
                        extra = None
                    # Set distance matrix from dataset
                    if hasattr(self.ev_dataset, 'distance_matrix'):
                        dist_matrix = torch.from_numpy(self.ev_dataset.distance_matrix).float()
                        dist_matrix = dist_matrix.to(feat.device)
                        self.ev_model.model.set_distance_matrix(dist_matrix)
                    preds = self.ev_model.model(feat, extra)
                elif self.ev_model.model_name == 'ev_vlm':
                    # batch = (feat, label, shared_ids, time_ids, time_mask, static_ids, static_mask, stats_ids, stats_mask, vision_emb)
                    feat, label, shared_ids, time_ids, time_mask, static_ids, static_mask, stats_ids, stats_mask, vision_emb = batch
                    device = feat.device
                    feat = feat.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)
                    shared_ids = shared_ids.to(device, non_blocking=True)
                    time_ids = time_ids.to(device, non_blocking=True)
                    time_mask = time_mask.to(device, non_blocking=True)
                    static_ids = static_ids.to(device, non_blocking=True)
                    static_mask = static_mask.to(device, non_blocking=True)
                    stats_ids = stats_ids.to(device, non_blocking=True)
                    stats_mask = stats_mask.to(device, non_blocking=True)
                    vision_emb = vision_emb.to(device, non_blocking=True)
                    with self._autocast():
                        preds = self.ev_model.model(
                            feat,
                            shared_ids=shared_ids,
                            time_ids=time_ids,
                            time_mask=time_mask,
                            static_ids=static_ids,
                            static_mask=static_mask,
                            stats_ids=stats_ids,
                            stats_mask=stats_mask,
                            vision_emb=vision_emb,
                        )
                else:
                    feat, label, extra = batch
                    if self.ev_dataset.extra_feat is None:
                        extra = None
                    preds = self.ev_model.model(feat, extra)

                if preds.shape != label.shape:
                    loss = self.loss_func(preds.unsqueeze(-1), label)
                else:
                    loss = self.loss_func(preds, label)

                loss.backward()

                # Compute gradient norm before optimizer step
                total_norm = 0.0
                for p in self.ev_model.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                grad_norms.append(total_norm)

                self.optim.step()
                train_total += loss.item()
                train_count += 1

            avg_train_loss = train_total / train_count
            avg_grad_norm = sum(grad_norms) / len(grad_norms)

            # Step LR scheduler after each epoch
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step()

            # Validation phase
            self.ev_model.model.eval()
            val_total = 0.0
            val_count = 0
            for batch in self.ev_dataset.valid_loader:

                if self.ev_model.model_name in ('gwnet', 'astgcn'):
                    feat, label = batch[0], batch[1]
                    preds = self.ev_model.model(feat)
                elif self.ev_model.model_name == 'hyperdist':
                    feat, label, extra = batch
                    if self.ev_dataset.extra_feat is None:
                        extra = None
                    if hasattr(self.ev_dataset, 'distance_matrix'):
                        dist_matrix = torch.from_numpy(self.ev_dataset.distance_matrix).float()
                        dist_matrix = dist_matrix.to(feat.device)
                        self.ev_model.model.set_distance_matrix(dist_matrix)
                    preds = self.ev_model.model(feat, extra)
                elif self.ev_model.model_name == 'ev_vlm':
                    # validation
                    feat, label, shared_ids, time_ids, time_mask, static_ids, static_mask, stats_ids, stats_mask, vision_emb = batch
                    device = feat.device
                    feat = feat.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)
                    shared_ids = shared_ids.to(device, non_blocking=True)
                    time_ids = time_ids.to(device, non_blocking=True)
                    time_mask = time_mask.to(device, non_blocking=True)
                    static_ids = static_ids.to(device, non_blocking=True)
                    static_mask = static_mask.to(device, non_blocking=True)
                    stats_ids = stats_ids.to(device, non_blocking=True)
                    stats_mask = stats_mask.to(device, non_blocking=True)
                    vision_emb = vision_emb.to(device, non_blocking=True)
                    with torch.no_grad(), self._autocast():
                        preds = self.ev_model.model(
                                feat,
                                shared_ids=shared_ids,
                                time_ids=time_ids,
                                time_mask=time_mask,
                                static_ids=static_ids,
                                static_mask=static_mask,
                                stats_ids=stats_ids,
                                stats_mask=stats_mask,
                                vision_emb=vision_emb,
                            )
                else:
                    feat, label, extra = batch
                    if self.ev_dataset.extra_feat is None:
                        extra = None
                    preds = self.ev_model.model(feat, extra)

                if preds.shape != label.shape:
                    loss = self.loss_func(preds.unsqueeze(-1), label)
                else:
                    loss = self.loss_func(preds, label)

                val_total += loss.item()
                val_count += 1

            avg_val_loss = val_total / val_count

            # Restore training mode for next epoch
            self.ev_model.model.train()

            print(f"Epoch {current_epoch+1}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}, grad={avg_grad_norm:.2f}")
            _sys.stdout.flush()

            # Periodic checkpoint (every 10 epochs, for crash recovery)
            if (current_epoch + 1) % 10 == 0:
                ckpt = {
                    'epoch': current_epoch + 1,
                    'model': self.ev_model.model.state_dict(),
                    'optimizer': self.optim.state_dict(),
                    'best_val_loss': best_val_loss,
                    'early_stop_counter': early_stop_counter,
                    'train_stage': self.train_stage if hasattr(self, 'train_stage') else None,
                }
                torch.save(ckpt, os.path.join(self.save_path, 'resume.pth'))

            if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stop_counter = 0
                    os.makedirs(self.save_path, exist_ok=True)
                    # 只存「可训练参数 + 全部 buffer」：
                    #   * 可训练参数 —— 冻结的预训练骨干（GPT-2 / SigLIP）不存，保住体积优势；
                    #   * buffer —— **必须存**。BatchNorm 的 running_mean/running_var 是
                    #     buffer 而非 parameter，漏掉它 fresh-process 推理会静默沿用
                    #     mean=0/var=1 的初始化统计，预测系统性偏移（PatchTST 的
                    #     norm_type=batchnorm 就踩了这个坑；训练进程内 test() 看不出来，
                    #     因为内存里的 buffer 是对的）。buffer 都很小，代价可忽略。
                    keep = {n for n, p in self.ev_model.model.named_parameters() if p.requires_grad}
                    keep |= {n for n, _ in self.ev_model.model.named_buffers()}
                    trainable = {k: v for k, v in self.ev_model.model.state_dict().items()
                                 if k in keep}
                    ckpt_path = os.path.join(self.save_path, 'train.pth')
                    torch.save(trainable, ckpt_path)
            elif self.early_stop_patience > 0:
                early_stop_counter += 1
                if early_stop_counter >= self.early_stop_patience:
                    print(f"Early stopping at epoch {current_epoch+1} (val_loss not improved for {self.early_stop_patience} epochs)")
                    _sys.stdout.flush()
                    break

    def test(self, model_path: Optional[str] = None) -> None:
        """
        Evaluate model on test data and save predictions, labels, and metrics.

        This method performs comprehensive evaluation of the trained model on
        test data. It handles both neural and statistical models, computes
        multiple regression metrics, and saves all results for analysis.

        For neural models, it loads saved checkpoint if provided.
        For statistical models, it runs direct prediction on raw arrays.

        Args:
            model_path (str, optional): Path to model weights for neural models.
        """
        preds_list, labels_list = [], []  # Lists to collect predictions and labels

        if not self.stat_model:
            # Neural model evaluation
            if model_path:
                self.ev_model.load_model(model_path)  # Load saved model weights
            self.ev_model.model.eval()  # Set model to evaluation mode

            # Iterate through test data
            for batch in self.ev_dataset.test_loader:

                with torch.no_grad():
                    if self.ev_model.model_name in ('gwnet', 'astgcn'):
                        feat, label = batch[0], batch[1]
                        preds = self.ev_model.model(feat)
                    elif self.ev_model.model_name == 'hyperdist':
                        feat, label, extra = batch
                        if self.ev_dataset.extra_feat is None:
                            extra = None
                        preds = self.ev_model.model(feat, extra)
                    elif self.ev_model.model_name == 'ev_vlm':
                        # test
                        feat, label, shared_ids, time_ids, time_mask, static_ids, static_mask, stats_ids, stats_mask, vision_emb = batch
                        device = feat.device
                        feat = feat.to(device, non_blocking=True)
                        label = label.to(device, non_blocking=True)
                        shared_ids = shared_ids.to(device, non_blocking=True)
                        time_ids = time_ids.to(device, non_blocking=True)
                        time_mask = time_mask.to(device, non_blocking=True)
                        static_ids = static_ids.to(device, non_blocking=True)
                        static_mask = static_mask.to(device, non_blocking=True)
                        stats_ids = stats_ids.to(device, non_blocking=True)
                        stats_mask = stats_mask.to(device, non_blocking=True)
                        vision_emb = vision_emb.to(device, non_blocking=True)
                        with self._autocast():
                            preds = self.ev_model.model(
                                feat,
                                shared_ids=shared_ids,
                                time_ids=time_ids,
                                time_mask=time_mask,
                                static_ids=static_ids,
                                static_mask=static_mask,
                                stats_ids=stats_ids,
                                stats_mask=stats_mask,
                                vision_emb=vision_emb,
                            )
                    else:
                        feat, label, extra = batch
                        if self.ev_dataset.extra_feat is None:
                            extra = None
                        preds = self.ev_model.model(feat, extra)

                    preds = preds.unsqueeze(-1) if preds.shape != label.shape else preds
                    preds = preds.cpu().numpy()

                labels = label.cpu().numpy()
                preds_list.append(preds)
                labels_list.append(labels)
        else:
            # Statistical model prediction
            train_valid, test_feats = self.test_loader  # Unpack test data
            preds = self.ev_model.model.predict(train_valid, test_feats)  # Direct prediction
            preds_list.append(preds)  # Collect predictions
            labels_list.append(test_feats)  # Collect labels

        # Concatenate all predictions and labels
        pred_array = np.concatenate(preds_list, axis=0)  # Combine all predictions
        label_array = np.concatenate(labels_list, axis=0)  # Combine all labels
        
        # Inverse transform if scaler is available
        if self.ev_dataset.scaler:
            pred_array = _inverse_transform_with_scaler(self.ev_dataset.scaler, pred_array)
            label_array = _inverse_transform_with_scaler(self.ev_dataset.scaler, label_array)

        # Save outputs to files
        os.makedirs(self.save_path, exist_ok=True)  # Create save directory
        np.save(os.path.join(self.save_path, 'predict.npy'), pred_array)  # Save predictions
        np.save(os.path.join(self.save_path, 'label.npy'), label_array)  # Save labels

        # Flatten site dimension for metric computation: [T, N, pre_len] -> [T*N, pre_len]
        if pred_array.ndim == 3:
            T, N, P = pred_array.shape
            pred_array = pred_array.reshape(-1, P)
            label_array = label_array.reshape(-1, P)

        # Compute comprehensive regression metrics
        metrics = calculate_regression_metrics(y_true=label_array, y_pred=pred_array)  # Calculate metrics
        metrics = convert_numpy(metrics)  # Convert numpy types to native Python types
        
        # Save metrics to JSON file
        with open(
                os.path.join(self.save_path, 'metrics.json'), 'w', encoding='utf-8'
        ) as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)  # Save metrics with formatting
