# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/13 20:14
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/13 20:14

"""
Module for loading and processing electric vehicle dataset features in distributed settings.

This module provides functionality for handling distributed electric vehicle
charging data across multiple cities and sites. It supports federated learning
scenarios where data is partitioned across different clients (cities or sites)
and includes cross-validation splits, client partitioning, and DataLoader creation.

Key Features:
    - Multi-city data loading and preprocessing
    - Client-based data partitioning for federated learning
    - Site-level and city-level prediction modes
    - Automatic client selection for training and evaluation
    - Support for auxiliary features (pricing, weather, location)
"""

# Standard library imports
from collections import defaultdict

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Local imports
from api.utils import get_n_feature


class DistributedEVDataset(object):
    """
    Distributed dataset handler for EV features across multiple clients (cities/sites).

    This class provides comprehensive functionality for loading, preprocessing,
    and partitioning electric vehicle charging data across multiple cities and
    sites. It supports both site-level and city-level prediction modes,
    making it suitable for federated learning scenarios.

    The class handles data loading from multiple cities, feature normalization,
    auxiliary feature construction, and client partitioning for distributed
    training and evaluation.

    Attributes:
        feature (str): Type of feature to load ('volume' or 'duration').
        auxiliary (str): Auxiliary feature modes ('None', 'all', or '+'-separated list).
        clients_data (dict): Mapping client_id to its data dict with keys 'feat', 'extra_feat', 'time'.
        city_scalers (dict): Mapping city codes to their max value scalers.
        n_fea (int): Number of feature channels in extra_feat.
        training_clients_data (dict): Clients reserved for training.
        eval_clients_data (dict): Clients reserved for evaluation.
    """

    def __init__(
            self,
            feature: str,
            auxiliary: str,
            data_paths: dict[str, str],
            pred_type: str,
            eval_percentage: float,
            eval_city: str,
            max_sites: int = 300,
            weather_columns: list[str] = ['temp', 'precip', 'visibility'],
            selection_mode: str = 'middle',
    ) -> None:
        """
        Initialize DistributedEVDataset by loading and preprocessing per-city data.

        This method performs the complete data loading and preprocessing pipeline
        for multiple cities. It loads data from each city, applies normalization,
        constructs auxiliary features, and partitions data into client-specific
        datasets for federated learning.

        Args:
            feature (str): Feature file to load; 'volume' or 'duration'.
            auxiliary (str): Auxiliary data mode: 'None', 'all', or combination of features.
            data_paths (dict): Mapping city abbreviations to data directories.
            pred_type (str): Prediction granularity; 'siter 'city'.
            eval_percentage (float): % of clients per city for evaluation (sitede).
            eval_city (str): City code reserved for evaluation (city mode).
            max_siteint): Max sitsitecity (site msite
            weather_columns (list[str]): Weather attributes to include.
            selection_mode (str): Site selection strategy: 'top', 'middle', 'random'.

        Raises:
            ValueError: If feature or pred_type or selection_mode is invalid.
        """
        super(DistributedEVDataset, self).__init__()
        self.feature = feature
        self.auxiliary = auxiliary
        self.clients_data = {}  # Store data for each client
        self.city_scalers = {}  # Store scaling factors for each city

        # Process each city's data independently
        for city_abbr, data_path in data_paths.items():
            # Load and normalize main feature DataFrame based on feature type
            if self.feature == 'volume':
                feat_df = pd.read_csv(f'{data_path}volume.csv', header=0, index_col=0)
            elif self.feature == 'duration':
                feat_df = pd.read_csv(f'{data_path}duration.csv', header=0, index_col=0)
            else:
                raise ValueError("Unknown feature - must be 'volume' or 'duration'")

            # Scale features by city maximum value for normalization
            max_val = feat_df.max().max()  # Get maximum value across all sitend time
            feat_scaled = feat_df / max_val  # Normalize by maximum value
            self.city_scalers[city_abbr] = max_val  # Store scaling factor for later use
            feat_df = feat_scaled.copy()  # Use scaled features

            sites = list(feat_df.columns)  # Get list of sitsite

            # Load and normalize price data
            e_price_all = pd.read_csv(f'{data_path}e_price.csv', index_col=0, header=0).values  # Electricity prices
            s_price_all = pd.read_csv(f'{data_path}s_price.csv', index_col=0, header=0).values  # Service prices
            time_series = pd.to_datetime(feat_df.index)  # Convert index to datetime
            price_scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize price scaler
            e_price_all = price_scaler.fit_transform(e_price_all)  # Normalize electricity prices
            s_price_all = price_scaler.fit_transform(s_price_all)  # Normalize service prices

            # Load and normalize weather data
            weather = pd.read_csv(f'{data_path}weather.csv', header=0, index_col='time')
            weather = weather[weather_columns]  # Select specified weather columns
            
            # Apply domain-specific normalization for weather features
            if 'temp' in weather.columns:
                weather['temp'] = (weather['temp'] + 5) / 45  # Scale temperature: [-5, 40] -> [0, 1]
            if 'precip' in weather.columns:
                weather['precip'] = weather['precip'] / 120  # Scale precipitation: [0, 120] -> [0, 1]
            if 'visibility' in weather.columns:
                weather['visibility'] = weather['visibility'] / 50  # Scale visibility: [0, 50] -> [0, 1]

            # Load sitetadata for selection and feature construction
            sitenfo = pd.read_csv(f'{data_path}sitsite).set_index('site_isite
            sitenfo.index = sitsite.index.astype(str)  # Convert index to string type

            # Site selection if pred_type == 'sitend exceeding max_sitsite
            if pred_type == 'sitend len(sitsite) > max_sites:site
                if selection_mode == 'top':
                    # Select top sitey total_duration (most active)
                    selected_site sitsite.sort_values(by='total_duration', ascending=False).head(
                        max_site
                elif selection_mode == 'middle':
                    # Select middle-range sitey total_duration
                    sorted_site sitsite.sort_values(by='total_duration', ascending=True)
                    start = max((len(sorted_site- max_sitsite2, 0)  # Calculate start index
                    selected_site sorted_sitsite[start:start + max_sites]site
                elif selection_mode == 'random':
                    # Select random sample of site
                    selected_site sitsite.sample(n=max_sites,site_state=42)  # Fixed seed
                else:
                    raise ValueError(f"Unknown selection_mode: {selection_mode}")
                
                # Filter data for selected site
                selected_ids = selected_sitendex.tolist()  # Get selected sitsite
                feat_df = feat_df[selected_ids]  # Filter main features
                
                # Re-load and re-scale price data for selected site
                e_price_df = pd.read_csv(f'{data_path}e_price.csv', index_col=0, header=0)
                s_price_df = pd.read_csv(f'{data_path}s_price.csv', index_col=0, header=0)
                e_price_all = price_scaler.fit_transform(e_price_df[selected_ids])  # Re-scale electricity prices
                s_price_all = price_scaler.fit_transform(s_price_df[selected_ids])  # Re-scale service prices
                sitenfo = selected_sitsitepdate sites site
                sites = selected_ids  # Update sitsite

            # Normalize latitude/longitude and create base extra_feat
            lat_long = sitenfo.loc[feat_df.columns, ['latitude', 'longitude']].values
            lat_norm = (lat_long[:, 0] + 90) / 180  # Normalize latitude: [-90, 90] -> [0, 1]
            lon_norm = (lat_long[:, 1] + 180) / 360  # Normalize longitude: [-180, 180] -> [0, 1]
            lat_long_norm = np.stack([lat_norm, lon_norm], axis=1)  # Stack into 2D array
            
            # Create spatial features by repeating coordinates for each time step
            extra_feat = np.tile(lat_long_norm[np.newaxis, :, :], (feat_df.shape[0], 1, 1))

            # Add auxiliary channels if requested
            if self.auxiliary != 'None':
                # Initialize with zero channel for base features
                extra_feat = np.zeros([feat_df.shape[0], feat_df.shape[1], 1])
                
                if self.auxiliary == 'all':
                    # Add all auxiliary features: e_price, s_price, weather
                    extra_feat = np.concatenate([extra_feat, e_price_all[:, :, np.newaxis]], axis=2)  # Add electricity price
                    extra_feat = np.concatenate([extra_feat, s_price_all[:, :, np.newaxis]], axis=2)  # Add service price
                    extra_feat = np.concatenate([extra_feat,
                                                 np.repeat(weather.values[:, np.newaxis, :], feat_df.shape[1], axis=1)],
                                                axis=2)  # Add weather features
                else:
                    # Add specified auxiliary features based on '+' separated list
                    add_feat_list = self.auxiliary.split('+')
                    for add_feat in add_feat_list:
                        if add_feat == 'e_price':
                            # Add electricity price feature
                            extra_feat = np.concatenate([extra_feat, e_price_all[:, :, np.newaxis]], axis=2)
                        elif add_feat == 's_price':
                            # Add service price feature
                            extra_feat = np.concatenate([extra_feat, s_price_all[:, :, np.newaxis]], axis=2)
                        else:
                            # Add weather feature channel
                            extra_feat = np.concatenate([extra_feat,
                                                         np.repeat(weather[add_feat].values[:, np.newaxis, np.newaxis],
                                                                   feat_df.shape[1], axis=1)], axis=2)
                
                # Remove initial zero channel to get final auxiliary features
                extra_feat = extra_feat[:, :, 1:]

            # Convert to numpy array and compute number of features
            feat_array = np.array(feat_df)  # Convert DataFrame to numpy array
            self.n_fea = get_n_feature(extra_feat)  # Calculate number of feature channels

            # Partition data into clients based on prediction type
            if pred_type == 'site
                # Create individual client for each site
                for idx, site in enumerate(sitsite
                    client_feat = feat_array[:, idx:idx + 1]  # Extract single siteatures
                    client_extra = extra_feat[:, idx:idx + 1, :] if extra_feat is not None else None  # Extract auxiliary features
                    client_id = f"{city_abbr}_{site}"  # Create client ID: city_sitsite
                    self.clients_data[client_id] = {
                        'feat'      : client_feat,      # Site features
                        'extra_feat': client_extra,     # Site auxiliary features
                        'time'      : time_series       # Time series
                    }
            elif pred_type == 'city':
                # Create single client for entire city (aggregated data)
                aggregated_feat = np.sum(feat_array, axis=1, keepdims=True)  # Sum across all site
                aggregated_extra = np.mean(extra_feat, axis=1, keepdims=True) if extra_feat is not None else None  # Mean across site
                self.clients_data[city_abbr] = {
                    'feat'      : aggregated_feat,      # City-level features
                    'extra_feat': aggregated_extra,     # City-level auxiliary features
                    'time'      : time_series           # Time series
                }
            else:
                raise ValueError("Unknown pred_type - must be 'siter 'city'")

        # Perform final client partitioning for training and evaluation
        self.partition_clients(eval_percentage, eval_city, pred_type)

    def partition_clients(
            self,
            eval_percentage: float,
            eval_city: str,
            pred_type: str,
    ) -> None:
        """
        Partition client IDs into training and evaluation sets.

        This method separates clients into training and evaluation sets based on
        the prediction type and evaluation parameters. For sitevel prediction,
        it partitions siteithin each city. For city-level prediction, it
        reserves a specific city for evaluation.

        Args:
            eval_percentage (float): Percentage of clients for evaluation (sitede).
            eval_city (str): City code for evaluation (city mode).
            pred_type (str): 'siter 'city'.

        Raises:
            ValueError: If pred_type is invalid.
        """
        if pred_type == 'site
            # Site-level partitioning: split siteithin each city
            training_clients_data = {}  # Store training clients
            eval_clients_data = {}      # Store evaluation clients
            city_clients = defaultdict(list)  # Group clients by city
            
            # Group client IDs by city
            for client_id in self.clients_data.keys():
                city = client_id.split('_')[0]  # Extract city from client ID
                city_clients[city].append(client_id)  # Add to city group
            
            # Partition clients within each city
            for city, client_ids in city_clients.items():
                client_ids_sorted = sorted(client_ids)  # Sort for deterministic partitioning
                num_clients = len(client_ids_sorted)  # Total number of clients in city
                num_eval = int(num_clients * eval_percentage / 100)  # Calculate evaluation count
                
                # Ensure at least one client for evaluation if percentage > 0
                if num_clients > 0 and num_eval == 0 and eval_percentage > 0:
                    num_eval = 1
                
                # Split clients into evaluation and training sets
                eval_ids = client_ids_sorted[:num_eval]      # First clients for evaluation
                train_ids = client_ids_sorted[num_eval:]     # Remaining clients for training
                
                # Assign clients to respective sets
                for cid in train_ids:
                    training_clients_data[cid] = self.clients_data[cid]  # Add to training set
                for cid in eval_ids:
                    eval_clients_data[cid] = self.clients_data[cid]      # Add to evaluation set
            
            # Store partitioned data
            self.training_clients_data = training_clients_data
            self.eval_clients_data = eval_clients_data
            
        elif pred_type == 'city':
            # City-level partitioning: reserve specific city for evaluation
            training_clients_data = {}  # Store training cities
            eval_clients_data = {}      # Store evaluation city
            
            # Partition cities based on evaluation city
            for city, data in self.clients_data.items():
                if city == eval_city:
                    eval_clients_data[city] = data  # Reserve city for evaluation
                else:
                    training_clients_data[city] = data  # Use for training
            
            # Store partitioned data
            self.training_clients_data = training_clients_data
            self.eval_clients_data = eval_clients_data
            
        else:
            raise ValueError("Unknown pred_type - must be 'site' or 'city'")

    def get_client_ids(self) -> list[str]:
        """
        Get list of all client IDs.

        Returns:
            list[str]: List of all client identifiers.
        """
        return list(self.clients_data.keys())

    def get_client_data(self, client_id: str) -> dict | None:
        """
        Retrieve data for a specific client.

        Args:
            client_id (str): Identifier of the client.

        Returns:
            dict or None: Data dictionary with 'feat', 'extra_feat', 'time', or None if client not found.
        """
        return self.clients_data.get(client_id, None)
