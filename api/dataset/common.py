# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/10 17:16
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/10 17:16

"""
Module for loading and processing electric vehicle dataset features.

This module provides comprehensive functionality for loading and preprocessing
electric vehicle charging data, including volume/duration features, pricing
information, weather data, and site metadata. It supports cross-validation
splits and DataLoader creation for deep learning training.

Key Features:
    - Loads and normalizes multiple data sources (volume/duration, pricing, weather)
    - Supports site selection strategies (top, middle, random)
    - Implements cross-validation splits by month
    - Creates PyTorch DataLoaders for training, validation, and testing
    - Handles auxiliary feature construction and normalization
"""

# Standard library imports
import os

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader

# Local imports
from api.utils import (
    CreateDataset,
    EV_VLM_Dataset,
    ASTGCNDataset,
    ASTGCN_RECENT_LEN,
    ASTGCN_DAILY_LEN,
    ASTGCN_DAILY_INTERVAL,
)
from api.utils.spatial import (
    load_distance_matrix,
    construct_adjacency_matrix,
    to_torch_tensor,
)



class EVDataset(object):
    """
    Dataset handler for EV (electric vehicle) features with auxiliary data.

    This class provides a comprehensive interface for loading, preprocessing,
    and managing electric vehicle charging data. It handles multiple data
    sources including main features (volume or duration), pricing information,
    weather data, and site metadata. The class supports various site
    selection strategies and provides methods for cross-validation splitting
    and DataLoader creation.

    Attributes:
        feature (str): Type of feature to load ('volume' or 'duration').
        auxiliary (str): Auxiliary feature modes ('None', 'all', or '+'-separated list).
        data_path (str): Root path to CSV files.
        feat (np.ndarray): Primary feature array [time, sites].
        extra_feat (np.ndarray): Auxiliary feature array [time, sites, features].
        time (pd.DatetimeIndex): Timestamps for each row in feat.
        scaler (StandardScaler): Scaler fitted on training data.
        train_feat, valid_feat, test_feat (np.ndarray): Splits of feat.
        train_extra_feat, valid_extra_feat, test_extra_feat (np.ndarray): Splits of extra_feat.
        train_loader, valid_loader, test_loader (DataLoader): PyTorch DataLoaders.
    """

    def __init__(
            self,
            feature: str,
            auxiliary: str,
            data_path: str = 'data/',
            max_sites: int = 300,
            weather_columns: list[str] = ['temp', 'precip', 'visibility'],
            selection_mode: str = 'top',
            sat_embed_root: str | None = None,
            sat_embed_zoom: str = 'z16',
            sat_embed_dim: int = 256,
            sat_embed_strict: bool = False,
            poi_topk: int = 32,
            poi_min_count: int = 50,
                poi_radius_m: str = '300,800,1500',
                poi_radius_weights: str = '1.0,0.5,0.2',
    ) -> None:
        """
        Initialize EVDataset by loading and preprocessing data.

        This method performs the complete data loading and preprocessing pipeline:
        1. Loads main feature data (volume or duration)
        2. Loads and normalizes pricing data
        3. Loads and normalizes weather data
        4. Loads site metadata and applies selection strategy
        5. Constructs auxiliary features based on configuration
        6. Normalizes all data for machine learning

        Args:
            feature (str): Feature file to load; 'volume' or 'duration'.
            auxiliary (str): Auxiliary data mode: 'None', 'all', or combination of 'e_price', 's_price', weather keys.
            data_path (str): Directory path containing CSV files.
            max_sites (int): Maximum number of sites to select (default: 300).
            weather_columns (list[str]): Columns to load from weather data (default: ['temp', 'precip', 'visibility']).
            selection_mode (str): Site selection mode: 'top', 'middle', or 'random' (default: 'top').

        Raises:
            ValueError: If feature name or selection_mode is invalid.
        """
        super(EVDataset, self).__init__()
        self.feature = feature
        self.auxiliary = auxiliary
        self.data_path = data_path
        self.sat_embed_root = sat_embed_root
        self.sat_embed_zoom = sat_embed_zoom
        self.sat_embed_dim = sat_embed_dim
        self.sat_embed_strict = sat_embed_strict
        self.poi_topk = poi_topk
        self.poi_min_count = poi_min_count
        self.poi_radius_m = poi_radius_m
        self.poi_radius_weights = poi_radius_weights

        # Load main feature DataFrame based on feature type
        if self.feature == 'volume':
            self.feat = pd.read_csv(f'{self.data_path}volume.csv', header=0, index_col=0)
        elif self.feature == 'duration':
            self.feat = pd.read_csv(f'{self.data_path}duration.csv', header=0, index_col=0)
        else:
            raise ValueError("Unknown feature - must be 'volume' or 'duration'")

        self.feat_columns = list(self.feat.columns)  # preserve column names before numpy conversion

        # Load and normalize price data
        self.e_price = pd.read_csv(f'{self.data_path}e_price.csv', index_col=0, header=0).values  # Electricity price
        self.s_price = pd.read_csv(f'{self.data_path}s_price.csv', index_col=0, header=0).values  # Service price
        self.time = pd.to_datetime(self.feat.index)  # Convert index to datetime

        # Normalize price data to [0, 1] range using MinMaxScaler
        price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.e_price = price_scaler.fit_transform(self.e_price)  # Normalize electricity prices
        self.s_price = price_scaler.fit_transform(self.s_price)  # Normalize service prices

        # Load and normalize weather data
        self.weather = pd.read_csv(f'{self.data_path}weather.csv', header=0, index_col='time')
        self.weather = self.weather[weather_columns]  # Select specified weather columns
        
        # Apply domain-specific normalization for weather features
        if 'temp' in self.weather.columns:
            # Scale temperature from [-5, 40] to [0, 1]
            self.weather['temp'] = (self.weather['temp'] + 5) / 45
        if 'precip' in self.weather.columns:
            # Scale precipitation from [0, 120] to [0, 1]
            self.weather['precip'] = self.weather['precip'] / 120
        if 'visibility' in self.weather.columns:
            # Scale visibility from [0, 50] to [0, 1]
            self.weather['visibility'] = self.weather['visibility'] / 50

        # Load site metadata for selection and feature construction
        sites_info = pd.read_csv(f'{self.data_path}sites.csv', header=0)
        site_col = 'site' if 'site' in sites_info.columns else 'site_id' if 'site_id' in sites_info.columns else None
        if site_col is None:
            raise KeyError("sites.csv must contain either 'site' or 'site_id' column")
        sites_info = sites_info.set_index(site_col)
        sites_info.index = sites_info.index.astype(str)  # Convert index to string type

        # Normalize site latitude and longitude to [0, 1] range
        lat_long = sites_info.loc[self.feat.columns, ['latitude', 'longitude']].values
        lat_norm = (lat_long[:, 0] + 90) / 180  # Normalize latitude: [-90, 90] -> [0, 1]
        lon_norm = (lat_long[:, 1] + 180) / 360  # Normalize longitude: [-180, 180] -> [0, 1]
        self.lat_long_norm = np.stack([lat_norm, lon_norm], axis=1)  # Stack into 2D array
        
        # Create spatial features by repeating coordinates for each time step
        # Only set extra_feat when auxiliary features are requested
        self.extra_feat = None
        if self.auxiliary != 'None':
            self.extra_feat = np.tile(
                self.lat_long_norm[np.newaxis, :, :], (self.feat.shape[0], 1, 1)
            )

        # Build auxiliary features if requested
        if self.auxiliary != 'None':
            # Initialize with zero channel for base features
            self.extra_feat = np.zeros([
                self.feat.shape[0], self.feat.shape[1], 1
            ])
            
            if self.auxiliary == 'all':
                # Add all auxiliary features: e_price, s_price, weather
                self.extra_feat = np.concatenate([
                    self.extra_feat, self.e_price[:, :, np.newaxis]  # Add electricity price
                ], axis=2)
                self.extra_feat = np.concatenate([
                    self.extra_feat, self.s_price[:, :, np.newaxis]  # Add service price
                ], axis=2)
                self.extra_feat = np.concatenate([
                    self.extra_feat,
                    np.repeat(
                        self.weather.values[:, np.newaxis, :],  # Repeat weather for all sites
                        self.feat.shape[1], axis=1
                    )
                ], axis=2)  # Add weather features
            else:
                # Add specified auxiliary features based on '+' separated list
                for add_feat in self.auxiliary.split('+'):
                    if add_feat == 'e_price':
                        # Add electricity price feature
                        self.extra_feat = np.concatenate([
                            self.extra_feat, self.e_price[:, :, np.newaxis]
                        ], axis=2)
                    elif add_feat == 's_price':
                        # Add service price feature
                        self.extra_feat = np.concatenate([
                            self.extra_feat, self.s_price[:, :, np.newaxis]
                        ], axis=2)
                    elif add_feat == 'poi':
                        poi_extra = self._load_poi_extra(site_ids=[str(c) for c in self.feat.columns], sites_info=sites_info)
                        if poi_extra is not None:
                            self.extra_feat = np.concatenate([
                                self.extra_feat, poi_extra
                            ], axis=2)
                    else:
                        # Add weather feature channel
                        self.extra_feat = np.concatenate([
                            self.extra_feat,
                            np.repeat(
                                self.weather[add_feat].values[:, np.newaxis, np.newaxis],
                                self.feat.shape[1], axis=1  # Repeat for all sites
                            )
                        ], axis=2)
            
            # Remove initial zero channel to get final auxiliary features
            self.extra_feat = self.extra_feat[:, :, 1:]

        site_ids = [str(c) for c in self.feat.columns]

        # Convert main feature DataFrame to numpy array for consistency
        self.feat = np.array(self.feat)

        # Optionally inject cached satellite embedding as extra channels.
        sat_extra = self._load_satellite_extra(site_ids)
        if sat_extra is not None:
            if self.extra_feat is None:
                self.extra_feat = sat_extra
            else:
                self.extra_feat = np.concatenate([self.extra_feat, sat_extra], axis=2)

    def load_distance_and_adjacency(
        self,
        adj_method: str = 'k_nearest',
        k: int = 5,
        normalize: bool = True,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """
        Load distance matrix and construct adjacency matrix.
        
        This method loads the distance matrix from the data directory and converts
        it to an adjacency matrix using the specified method. The adjacency matrix
        represents spatial dependencies between nodes (sites) and is used by
        graph neural network models like GWNet.
        
        Args:
            adj_method (str): Method for adjacency construction:
                - 'k_nearest': k-nearest neighbors graph (default)
                - 'threshold': distance threshold-based
                - 'rbf': RBF kernel
                - 'inverse': inverse distance
            k (int): Number of nearest neighbors (for k_nearest method).
            normalize (bool): Whether to normalize adjacency matrix.
            device (str): Device for the adjacency matrix tensor.
            
        Returns:
            torch.Tensor: Adjacency matrix [N, N].
            
        Raises:
            FileNotFoundError: If distance matrix file not found.
        """
        distance_path = os.path.join(self.data_path, 'distance.csv')
        
        if not os.path.exists(distance_path):
            raise FileNotFoundError(
                f"Distance matrix not found at {distance_path}. "
                "Please ensure distance.csv exists in {self.data_path}"
            )
        
        # Load full distance matrix
        try:
            dist_df = pd.read_csv(distance_path, index_col=0)
            dist_df.index = dist_df.index.astype(str)
            dist_df.columns = dist_df.columns.astype(str)
            dist_matrix = dist_df.values
        except Exception as e:
            print(f"Warning: Failed to parse distance.csv with index/columns: {e}")
            dist_matrix = load_distance_matrix(distance_path)
        
        # Ensure distance matrix matches current feature dimensions
        num_sites = self.feat.shape[1]
        current_dist_size = dist_matrix.shape[0]
        
        if current_dist_size != num_sites:
            print(f"⚠️  Warning: Distance matrix size ({current_dist_size}) != feature sites ({num_sites})")
            print(f"  Will use first {min(num_sites, current_dist_size)} sites")
            
            # Use only the sites that exist in both
            use_size = min(num_sites, current_dist_size)
            dist_matrix = dist_matrix[:use_size, :use_size]
            
            # Also trim features if needed
            if self.feat.shape[1] > use_size:
                self.feat = self.feat[:, :use_size]
        
        # Construct adjacency matrix
        adj = construct_adjacency_matrix(
            dist_matrix,
            method=adj_method,
            k=k,
            normalize=normalize
        )
        
        # Store locally
        self.adjacency = adj
        self.distance_matrix = dist_matrix
        self.num_nodes = adj.shape[0]
        
        # Convert to PyTorch tensor
        adj_tensor = to_torch_tensor(adj, device=device)
        
        return adj_tensor
    
    def get_adjacency_matrix(self, device: str = 'cpu') -> torch.Tensor:
        """
        Get adjacency matrix as PyTorch tensor.
        
        Requires calling load_distance_and_adjacency() first.
        
        Args:
            device (str): Device for tensor placement.
            
        Returns:
            torch.Tensor: Adjacency matrix [N, N].
        """
        if not hasattr(self, 'adjacency'):
            raise RuntimeError(
                "Adjacency matrix not loaded. "
                "Call load_distance_and_adjacency() first."
            )
        
        return to_torch_tensor(self.adjacency, device=device)

    def _load_poi_extra(self, site_ids: list[str], sites_info: pd.DataFrame) -> np.ndarray | None:
        """Build static POI features per site and expand to [T, N, C]."""
        poi_path = os.path.join(self.data_path, 'poi.csv')
        if not os.path.exists(poi_path):
            return None

        poi = pd.read_csv(poi_path)
        required = {'type', 'longitude', 'latitude'}
        if not required.issubset(set(poi.columns)):
            return None

        poi = poi.dropna(subset=['type', 'longitude', 'latitude']).copy()
        if poi.empty:
            return None

        type_counts = poi['type'].value_counts()
        keep_types = type_counts[type_counts >= self.poi_min_count].head(self.poi_topk).index.tolist()
        if not keep_types:
            return None

        poi = poi[poi['type'].isin(keep_types)].copy()
        if poi.empty:
            return None

        radius_vals = [float(x.strip()) for x in self.poi_radius_m.split(',') if x.strip()]
        weight_vals = [float(x.strip()) for x in self.poi_radius_weights.split(',') if x.strip()]
        if not radius_vals or not weight_vals:
            raise ValueError('poi_radius_m and poi_radius_weights must be non-empty lists')
        if len(radius_vals) != len(weight_vals):
            raise ValueError('poi_radius_m and poi_radius_weights must have the same length')
        if any(r <= 0 for r in radius_vals):
            raise ValueError('All poi radii must be > 0')
        radius_vals, weight_vals = zip(*sorted(zip(radius_vals, weight_vals), key=lambda x: x[0]))

        type_to_idx = {t: i for i, t in enumerate(keep_types)}

        site_ll = sites_info.loc[site_ids, ['latitude', 'longitude']].values.astype(np.float64)
        site_lat = np.radians(site_ll[:, 0])[np.newaxis, :]
        site_lon = np.radians(site_ll[:, 1])[np.newaxis, :]

        poi_lat = np.radians(poi['latitude'].to_numpy(dtype=np.float64))
        poi_lon = np.radians(poi['longitude'].to_numpy(dtype=np.float64))
        poi_type_idx = np.array([type_to_idx[t] for t in poi['type'].tolist()], dtype=np.int64)

        counts = np.zeros((len(site_ids), len(keep_types) * len(radius_vals)), dtype=np.float64)
        chunk_size = 20000
        earth_radius_m = 6371000.0

        for start in range(0, len(poi_lat), chunk_size):
            end = min(start + chunk_size, len(poi_lat))
            lat_chunk = poi_lat[start:end][:, np.newaxis]
            lon_chunk = poi_lon[start:end][:, np.newaxis]

            dlat = lat_chunk - site_lat
            dlon = lon_chunk - site_lon
            a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_chunk) * np.cos(site_lat) * (np.sin(dlon / 2.0) ** 2)
            dist_m = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 1e-12))) * earth_radius_m

            for ridx, (radius_m, radius_w) in enumerate(zip(radius_vals, weight_vals)):
                if radius_w == 0:
                    continue
                within = dist_m <= radius_m
                if not np.any(within):
                    continue

                # Distance-decayed contribution inside each radius layer.
                decay = np.exp(-dist_m / max(radius_m, 1e-6)) * within.astype(np.float64) * radius_w
                poi_types_chunk = poi_type_idx[start:end]
                for p_idx, t_idx in enumerate(poi_types_chunk):
                    col = ridx * len(keep_types) + int(t_idx)
                    contrib = decay[p_idx]
                    if np.any(contrib):
                        counts[:, col] += contrib

        counts = np.log1p(counts)
        col_min = counts.min(axis=0, keepdims=True)
        col_max = counts.max(axis=0, keepdims=True)
        denom = np.where((col_max - col_min) > 1e-12, (col_max - col_min), 1.0)
        counts = (counts - col_min) / denom
        counts = counts.astype(np.float32)

        return np.tile(counts[np.newaxis, :, :], (self.feat.shape[0], 1, 1))

    def _load_satellite_extra(self, site_ids: list[str]) -> np.ndarray | None:
        if not self.sat_embed_root:
            return None

        city_tag = os.path.basename(os.path.normpath(self.data_path)).replace('_remove_zero', '')
        embed_dir = os.path.join(self.sat_embed_root, city_tag, 'embedding', self.sat_embed_zoom)
        embed_path = os.path.join(embed_dir, f'sat_embed_{self.sat_embed_dim}.npy')
        sites_path = os.path.join(embed_dir, f'sat_embed_{self.sat_embed_dim}_sites.csv')

        if not os.path.exists(embed_path) or not os.path.exists(sites_path):
            if self.sat_embed_strict:
                raise FileNotFoundError(
                    f"Missing satellite embedding files: {embed_path} or {sites_path}"
                )
            return None

        embed = np.load(embed_path)
        site_df = pd.read_csv(sites_path)
        if 'site_id' not in site_df.columns:
            raise KeyError(f"{sites_path} must contain 'site_id' column")

        embed_site_ids = site_df['site_id'].astype(str).tolist()
        if len(embed_site_ids) != len(embed):
            raise ValueError(
                f"Embedding/site mapping size mismatch: {len(embed)} vs {len(embed_site_ids)}"
            )

        emb_map = {sid: vec for sid, vec in zip(embed_site_ids, embed)}
        missing_sites = [sid for sid in site_ids if sid not in emb_map]
        if missing_sites:
            msg = f"Missing {len(missing_sites)} sites in satellite embedding for city {city_tag}"
            if self.sat_embed_strict:
                raise ValueError(msg)

        if embed.ndim != 2:
            raise ValueError(f"Expected 2D embedding array, got shape {embed.shape}")

        aligned_rows = []
        zero_vec = np.zeros((embed.shape[1],), dtype=np.float32)
        for sid in site_ids:
            aligned_rows.append(emb_map.get(sid, zero_vec))

        if not aligned_rows:
            return None

        aligned = np.stack(aligned_rows, axis=0).astype(np.float32)
        # Expand to [T, N, C] so it follows existing extra_feat pipeline.
        return np.tile(aligned[np.newaxis, :, :], (self.feat.shape[0], 1, 1))

    def ev_vlm_collate(self, batch):
        """Collate function for EV-VLM DataLoader (works with num_workers)."""
        feat_list = [s[0] for s in batch]
        label_list = [s[1] for s in batch]
        indices = [s[-1] for s in batch]  # global window-start index (absolute)

        feat = torch.stack(feat_list)
        label = torch.stack(label_list)  # Use CreateDataset's correct per-split labels

        B, N = feat.shape[0], feat.shape[1]

        # Text tokens (shared fixed, time per-sample, static per-site)
        shared_ids = self._SHARED_IDS          # [1, S_shared]
        time_ids = self._TIME_IDS[indices]     # [B, S_time]
        time_mask = self._TIME_MASK[indices]   # [B, S_time]
        static_ids = self._STATIC_IDS          # [N, S_static]
        static_mask = self._STATIC_MASK        # [N, S_static]

        # Build and tokenize stats text per sample-site
        stats_texts = []
        for b_idx in range(B):
            for n_idx in range(N):
                win = feat[b_idx, n_idx, :]
                stats_texts.append(
                    f"• The historical EV charging sequence ranges from {float(win.min()):.2f} "
                    f"to {float(win.max()):.2f}, with a mean value of {float(win.mean()):.2f}.\n"
                )
        stats_enc = self._gpt_tokenizer(stats_texts, padding=True, return_tensors='pt', add_special_tokens=False)
        stats_ids = stats_enc['input_ids'].reshape(B, N, -1)
        stats_mask = stats_enc['attention_mask'].reshape(B, N, -1)

        vision_emb = self._SITE_VISION_EMB  # [N, 196, 768]

        return feat, label, shared_ids, time_ids, time_mask, static_ids, static_mask, stats_ids, stats_mask, vision_emb

    def split_cross_validation(
            self,
            fold: int,
            total_fold: int,
            train_ratio: float,
            valid_ratio: float,
    ) -> None:
        """
        Split data by month for cross-validation.

        This method implements a time-aware cross-validation strategy where
        data is split by months to preserve temporal dependencies. Each fold
        corresponds to a specific month, and within each fold, data is further
        split into training, validation, and test sets.

        Args:
            fold (int): Current fold index (1-based).
            total_fold (int): Total number of folds (should equal number of months).
            train_ratio (float): Fraction of fold data for training (e.g., 0.7).
            valid_ratio (float): Fraction of fold data for validation (e.g., 0.2).

        Raises:
            AssertionError: If time and feature lengths mismatch or train set empty.
        """
        # Validate data consistency
        assert len(self.time) == len(self.feat), "Time and feature lengths must match"
        month_list = sorted(np.unique(self.time.month))  # Get unique months
        assert total_fold == len(month_list), f"Total folds ({total_fold}) must equal number of months ({len(month_list)})"

        # Calculate split points based on month membership
        fold_time = self.time.month.isin(month_list[0:fold]).sum()  # Count timesteps in current fold
        train_end = int(fold_time * train_ratio)  # Training set end index
        valid_start = train_end  # Validation set start index
        valid_end = int(valid_start + fold_time * valid_ratio)  # Validation set end index
        test_start = valid_end  # Test set start index
        test_end = int(fold_time)  # Test set end index

        # Slice numpy arrays for each split
        train_feat = self.feat[:train_end]  # Training features
        valid_feat = self.feat[valid_start:valid_end]  # Validation features
        test_feat = self.feat[test_start:test_end]  # Test features

        # Apply standard scaling to normalize features
        self.scaler = StandardScaler()  # Initialize scaler
        self.train_feat = self.scaler.fit_transform(train_feat)  # Fit and transform training data
        self.valid_feat = self.scaler.transform(valid_feat)  # Transform validation data
        self.test_feat = self.scaler.transform(test_feat)  # Transform test data
        self.scaled_feat = self.scaler.transform(self.feat)  # Keep full continuous history for ASTGCN sampling

        # Initialize auxiliary feature splits
        self.train_extra_feat, self.valid_extra_feat, self.test_extra_feat = None, None, None
        
        # Split auxiliary features if present
        if self.extra_feat is not None:
            self.train_extra_feat = self.extra_feat[:train_end]  # Training auxiliary features
            self.valid_extra_feat = self.extra_feat[valid_start:valid_end]  # Validation auxiliary features
            self.test_extra_feat = self.extra_feat[test_start:test_end]  # Test auxiliary features

        # Store split boundaries for models that need full-series context.
        self.train_start = 0
        self.train_end = train_end
        self.valid_start = valid_start
        self.valid_end = valid_end
        self.test_start = test_start
        self.test_end = test_end

        # Validate that training set is not empty
        assert len(train_feat) > 0, "The training set cannot be empty!"

    def create_loaders(
            self,
            seq_l: int,
            pre_len: int,
            batch_size: int,
            device: torch.device,
            model_name: str = 'rnn',
            tiles_path: str = None,
            city: str = None,
            image_size: int = 384,
            skip_images: bool = False,
            gpt2_model_path: str = None,
            siglip_model_path: str = None,
    ) -> None:
        """
        Create PyTorch DataLoaders for training, validation, and testing.

        This method creates PyTorch DataLoader objects that handle batching,
        shuffling, and device placement for efficient training. Training data
        is shuffled and uses the specified batch size, while validation and
        test data use full-batch loading for consistent evaluation.

        Args:
            seq_l (int): Input sequence length for sliding window sampling.
            pre_len (int): Prediction horizon length.
            batch_size (int): Batch size for training loader.
            device (torch.device): Device for tensor allocation (CPU/GPU).
            model_name (str): Model name used to select ASTGCN-specific sampling.
            gpt2_model_path (str): Local path to GPT-2 model files.
            siglip_model_path (str): Local path to SigLIP model files.
        """
        self.tiles_path = tiles_path
        self.city = city
        if model_name == 'astgcn':
            train_dataset = ASTGCNDataset(
                self.scaled_feat,
                self.train_start,
                self.train_end,
                pre_len,
                device,
                recent_len=ASTGCN_RECENT_LEN,
                daily_len=ASTGCN_DAILY_LEN,
                daily_interval=ASTGCN_DAILY_INTERVAL,
            )
            valid_dataset = ASTGCNDataset(
                self.scaled_feat,
                self.valid_start,
                self.valid_end,
                pre_len,
                device,
                recent_len=ASTGCN_RECENT_LEN,
                daily_len=ASTGCN_DAILY_LEN,
                daily_interval=ASTGCN_DAILY_INTERVAL,
            )
            test_dataset = ASTGCNDataset(
                self.scaled_feat,
                self.test_start,
                self.test_end,
                pre_len,
                device,
                recent_len=ASTGCN_RECENT_LEN,
                daily_len=ASTGCN_DAILY_LEN,
                daily_interval=ASTGCN_DAILY_INTERVAL,
            )
        elif model_name == 'ev_vlm':
            # EV-VLM requires multimodal data loading
            from transformers import GPT2Tokenizer, SiglipProcessor, SiglipVisionModel
            import os
            import torch
            import pandas as pd
            from PIL import Image
            import numpy as np

            # Initialize GPT-2 tokenizer from local path
            gpt2_model_path_local = (
                gpt2_model_path if gpt2_model_path
                else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'weights', 'gpt2_model')
            )
            siglip_model_path_local = (
                siglip_model_path if siglip_model_path
                else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'weights', 'siglip2_model')
            )
            gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path_local)
            gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

            siglip_processor = SiglipProcessor.from_pretrained(siglip_model_path_local)

            # Load site metadata for generating informative static text
            sites_info = pd.read_csv(os.path.join(self.data_path, 'sites.csv'))
            site_col_name = 'site' if 'site' in sites_info.columns else 'site_id'
            sites_info[site_col_name] = sites_info[site_col_name].astype(str)
            sites_info = sites_info.set_index(site_col_name)

            # --- POI filtering by distance (with per-city cache) ---
            # Satellite mosaic at zoom 16 is ~2.9km per side,
            # so "nearby" = within ~1.5km of site center.
            POI_RADIUS_M = 1500
            poi_cache_path = os.path.join(self.data_path, f'site_poi_{POI_RADIUS_M}m.csv')
            site_poi_str_cache = {}  # site_id -> str like "restaurant: 3, cafe: 2"

            # Try loading cached POI results
            if os.path.exists(poi_cache_path):
                poi_cache = pd.read_csv(poi_cache_path)
                for _, row in poi_cache.iterrows():
                    if str(row['site_id']) in self.feat_columns:
                        site_poi_str_cache[str(row['site_id'])] = row['poi_str'] if pd.notna(row['poi_str']) else ""
            else:
                # Compute and cache
                poi_path_local = os.path.join(self.data_path, 'poi.csv')
                if os.path.exists(poi_path_local):
                    poi_df = pd.read_csv(poi_path_local)
                    required = {'type', 'longitude', 'latitude'}
                    if required.issubset(set(poi_df.columns)):
                        poi_df = poi_df.dropna(subset=['type', 'longitude', 'latitude'])
                        poi_lats = np.radians(poi_df['latitude'].to_numpy(dtype=np.float64))
                        poi_lons = np.radians(poi_df['longitude'].to_numpy(dtype=np.float64))
                        for sid in self.feat_columns:
                            if sid not in sites_info.index:
                                site_poi_str_cache[sid] = ""
                                continue
                            lat0 = float(sites_info.loc[sid, 'latitude'])
                            lon0 = float(sites_info.loc[sid, 'longitude'])
                            lat0_r = np.radians(lat0)
                            lon0_r = np.radians(lon0)
                            dlat = poi_lats - lat0_r
                            dlon = poi_lons - lon0_r
                            a = np.sin(dlat / 2) ** 2 + np.cos(lat0_r) * np.cos(poi_lats) * np.sin(dlon / 2) ** 2
                            c = 2 * np.arcsin(np.sqrt(a))
                            dists = 6371000 * c  # meter
                            nearby_mask = dists <= POI_RADIUS_M
                            nearby_types = poi_df.loc[nearby_mask, 'type']
                            if len(nearby_types) > 0:
                                type_counts = nearby_types.value_counts().head(5)
                                site_poi_str_cache[sid] = ', '.join(
                                    f"{t}: {cnt}" for t, cnt in type_counts.items()
                                )
                            else:
                                site_poi_str_cache[sid] = ""
                # Write cache
                cache_rows = [{'site_id': sid, 'poi_str': site_poi_str_cache.get(sid, "")}
                              for sid in self.feat_columns]
                pd.DataFrame(cache_rows).to_csv(poi_cache_path, index=False)

            # ------------------------------------------------------------
            # Pre-cache multimodal data
            #   per-site: SigLIP vision embeddings + GPT-2 static POI text tokens
            #   per-sample: GPT-2 shared/time text tokens
            # ------------------------------------------------------------
            N_sites = len(self.feat_columns)

            # Load SigLIP vision model once for precomputation (will be discarded)
            siglip_vision = SiglipVisionModel.from_pretrained(siglip_model_path_local)
            siglip_vision.eval().to(device)

            SITE_VISION_EMB = []  # [N, 196, 768]

            for site_id in self.feat_columns:
                # --- Satellite image ---
                tile_path = os.path.join(tiles_path, city, 'mosaic', 'z16', f'{site_id}.png')
                if os.path.exists(tile_path):
                    image = Image.open(tile_path).convert('RGB')
                else:
                    tile_path_legacy = os.path.join(tiles_path, city, 'tiles', f'{site_id}.png')
                    if os.path.exists(tile_path_legacy):
                        image = Image.open(tile_path_legacy).convert('RGB')
                    else:
                        image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                image = image.resize((224, 224), Image.BILINEAR)
                img_inputs = siglip_processor(images=image, return_tensors='pt')
                with torch.no_grad():
                    vis_out = siglip_vision(pixel_values=img_inputs['pixel_values'].to(device))
                SITE_VISION_EMB.append(vis_out.last_hidden_state.cpu())  # [1, 196, 768]

            # Free SigLIP vision model
            del siglip_vision

            SITE_VISION_EMB = torch.cat(SITE_VISION_EMB, dim=0)  # [N, 196, 768]

            # --- Static site context text via GPT-2 tokenizer (no city / no lat-lon) ---
            static_texts = [
                f"Nearby: {site_poi_str_cache.get(sid, 'none')}." for sid in self.feat_columns
            ]
            static_enc = gpt_tokenizer(static_texts, padding=True, return_tensors='pt', add_special_tokens=False)
            self._STATIC_IDS = static_enc['input_ids']         # [N, S_static]
            self._STATIC_MASK = static_enc['attention_mask']   # [N, S_static]

            # --- Shared task instruction tokens (fixed, no padding) ---
            shared_prefix_text = (
                "• You are good at charging demand forecasting.\n"
                f"• Your task is forecasting the next {pre_len} steps using past {seq_l} steps.\n"
            )
            shared_enc = gpt_tokenizer(shared_prefix_text, return_tensors='pt', add_special_tokens=False)
            self._SHARED_IDS = shared_enc['input_ids']         # [1, S_shared]

            # Tokenize and cache ALL time context token IDs (global window index)
            time_arr_full = self.time.values if hasattr(self.time, 'values') else self.time
            import pandas as _pd
            if not isinstance(time_arr_full, _pd.DatetimeIndex):
                time_arr_full = _pd.to_datetime(time_arr_full)
            total_samples = len(time_arr_full) - seq_l
            time_texts = []
            for i in range(total_samples):
                forecast_start = i + seq_l
                if forecast_start >= len(time_arr_full):
                    break
                ts = time_arr_full[forecast_start]
                hour = ts.hour
                if 6 <= hour < 12: time_of_day = "morning"
                elif 12 <= hour < 18: time_of_day = "afternoon"
                elif 18 <= hour < 24: time_of_day = "evening"
                else: time_of_day = "night"
                if ts.weekday() < 5: day_type = "weekdays"
                elif ts.weekday() == 5: day_type = "Saturday"
                else: day_type = "Sunday"
                time_texts.append(
                    f"• The forecast start time is {ts.strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"which is in the {time_of_day} on {day_type}.\n"
                )
            time_enc_all = gpt_tokenizer(time_texts, padding=True, return_tensors='pt', add_special_tokens=False)
            self._TIME_IDS = time_enc_all.input_ids    # [total_samples, S_time]
            self._TIME_MASK = time_enc_all.attention_mask

            # Store for collate_fn
            self._device = device
            self._seq_l = seq_l
            self._pre_len = pre_len
            self._gpt_tokenizer = gpt_tokenizer
            self._SITE_VISION_EMB = SITE_VISION_EMB

            # Create base datasets
            base_train = CreateDataset(seq_l, pre_len, self.train_feat, self.train_extra_feat, device)
            base_valid = CreateDataset(seq_l, pre_len, self.valid_feat, self.valid_extra_feat, device)
            base_test = CreateDataset(seq_l, pre_len, self.test_feat, self.test_extra_feat, device)

            # Wrap with EV_VLM_Dataset, passing the global window-start offset per split
            # so time_ids can be looked up by absolute sample index.
            train_dataset = EV_VLM_Dataset(base_train, tiles_path, city, self.feat_columns, {}, {}, start_offset=0)
            valid_dataset = EV_VLM_Dataset(base_valid, tiles_path, city, self.feat_columns, {}, {}, start_offset=self.valid_start)
            test_dataset = EV_VLM_Dataset(base_test, tiles_path, city, self.feat_columns, {}, {}, start_offset=self.test_start)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                collate_fn=self.ev_vlm_collate
            )
            valid_loader = DataLoader(
                valid_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=self.ev_vlm_collate
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=self.ev_vlm_collate
            )
        else:
            # Create training dataset and loader with shuffling
            train_dataset = CreateDataset(
                seq_l, pre_len, self.train_feat, self.train_extra_feat, device
            )
            # Create validation dataset and loader (no shuffling)
            valid_dataset = CreateDataset(
                seq_l, pre_len, self.valid_feat, self.valid_extra_feat, device
            )
            # Create test dataset and loader (no shuffling)
            test_dataset = CreateDataset(
                seq_l, pre_len, self.test_feat, self.test_extra_feat, device
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False, drop_last=True
            )
            valid_loader = DataLoader(
                valid_dataset, batch_size=batch_size, shuffle=False
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader


# ---------------------------------------------------------------------------
# Deprecated -- pre-caching now done inside create_loaders(ev_vlm) above.
# ---------------------------------------------------------------------------
def ev_vlm_collate_fn(
        batch,
        tiles_path: str,
        city: str,
        gpt_tokenizer: any,
        siglip_processor: any,
        site_ids: list,
        seq_l: int,
        pre_len: int,
        device: torch.device,
):
    raise RuntimeError(
        "ev_vlm_collate_fn is no longer used. "
        "Pre-caching is done inside EVDataset.create_loaders()."
    )
