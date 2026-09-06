# -*- coding: utf-8 -*-
"""
Utilities for spatial graph construction from distance matrices.

This module provides functions to:
1. Load distance matrices from CSV files
2. Convert distance matrices to adjacency matrices using various methods
3. Apply normalization strategies (row normalization, symmetric normalization)
4. Support different distance thresholds for graph connectivity
"""

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.spatial.distance import pdist, squareform


def load_distance_matrix(distance_path: str) -> np.ndarray:
    """
    Load distance matrix from CSV file.
    
    Args:
        distance_path (str): Path to distance CSV file.
        
    Returns:
        np.ndarray: Distance matrix [N, N].
    """
    dist_df = pd.read_csv(distance_path, index_col=0)
    dist_matrix = dist_df.values
    
    # Ensure symmetric
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    
    return dist_matrix


def distance_to_adjacency(
    dist_matrix: np.ndarray,
    method: str = 'k_nearest',
    k: int = 5,
    sigma: float = 1.0,
    threshold: float = None
) -> np.ndarray:
    """
    Convert distance matrix to adjacency matrix.
    
    Args:
        dist_matrix (np.ndarray): Distance matrix [N, N].
        method (str): Method for conversion:
            - 'k_nearest': k-nearest neighbors graph
            - 'threshold': threshold-based (distance < threshold)
            - 'rbf': RBF kernel exp(-dist^2 / sigma^2)
            - 'inverse': Inverse distance 1 / (1 + dist)
        k (int): Number of nearest neighbors (for k_nearest method).
        sigma (float): RBF kernel bandwidth.
        threshold (float): Distance threshold (for threshold method).
        
    Returns:
        np.ndarray: Adjacency matrix [N, N].
    """
    N = dist_matrix.shape[0]
    adj = np.zeros((N, N))
    
    if method == 'k_nearest':
        # k-nearest neighbors: each node connects to k nearest nodes
        for i in range(N):
            # Get indices of k nearest neighbors (excluding self)
            k_nearest_idx = np.argsort(dist_matrix[i, :])[:k+1]
            k_nearest_idx = k_nearest_idx[k_nearest_idx != i][:k]
            adj[i, k_nearest_idx] = 1
        
        # Make symmetric (undirected graph)
        adj = (adj + adj.T) / 2
        adj = np.where(adj > 0, 1, 0)
    
    elif method == 'threshold':
        if threshold is None:
            threshold = np.median(dist_matrix[np.nonzero(dist_matrix)])
        
        adj = np.where(dist_matrix < threshold, 1, 0)
        adj = np.where(np.eye(N, dtype=bool), 0, adj)  # Remove self-loops
    
    elif method == 'rbf':
        # RBF kernel: exp(-d^2 / sigma^2)
        adj = np.exp(-(dist_matrix ** 2) / (sigma ** 2))
        adj = np.where(np.eye(N, dtype=bool), 0, adj)  # Remove self-loops
    
    elif method == 'inverse':
        # Inverse distance: 1 / (1 + d)
        with np.errstate(divide='ignore', invalid='ignore'):
            adj = 1.0 / (1.0 + dist_matrix)
        adj = np.where(np.eye(N, dtype=bool), 0, adj)  # Remove self-loops
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return adj


def normalize_adjacency(adj: np.ndarray, method: str = 'row') -> np.ndarray:
    """
    Normalize adjacency matrix.
    
    Args:
        adj (np.ndarray): Adjacency matrix [N, N].
        method (str): Normalization method:
            - 'row': Row normalization (divide by row sum)
            - 'symmetric': Symmetric normalization D^{-1/2} A D^{-1/2}
            - 'no_norm': No normalization
            
    Returns:
        np.ndarray: Normalized adjacency matrix.
    """
    if method == 'row':
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1  # Avoid division by zero
        adj_norm = adj / row_sum
    
    elif method == 'symmetric':
        # D^{-1/2} A D^{-1/2}
        d = adj.sum(axis=1)
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0  # Handle 1/0
        d_inv_sqrt = np.diag(d_inv_sqrt)
        adj_norm = np.dot(np.dot(d_inv_sqrt, adj), d_inv_sqrt)
    
    elif method == 'no_norm':
        adj_norm = adj
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return adj_norm


def construct_adjacency_matrix(
    dist_matrix: np.ndarray,
    method: str = 'k_nearest',
    normalize: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Complete pipeline: distance -> adjacency -> normalize.
    
    Args:
        dist_matrix (np.ndarray): Distance matrix.
        method (str): Distance-to-adjacency method.
        normalize (bool): Whether to apply normalization.
        **kwargs: Additional arguments for distance_to_adjacency.
        
    Returns:
        np.ndarray: Normalized adjacency matrix.
    """
    # Convert distance to adjacency
    adj = distance_to_adjacency(dist_matrix, method=method, **kwargs)
    
    # Normalize
    if normalize:
        adj = normalize_adjacency(adj, method='symmetric')
    
    return adj


def to_torch_tensor(adj: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Convert adjacency matrix to PyTorch tensor.
    
    Args:
        adj (np.ndarray): Adjacency matrix.
        device (str): Device to place tensor on ('cpu' or 'cuda').
        
    Returns:
        torch.Tensor: Adjacency matrix tensor [N, N].
    """
    adj_tensor = torch.from_numpy(adj).float().to(device)
    return adj_tensor


# Example usage
if __name__ == '__main__':
    # Load distance matrix
    dist = load_distance_matrix('data/MEL_remove_zero/distance.csv')
    
    # Convert to adjacency matrix using k-nearest neighbors
    adj = construct_adjacency_matrix(dist, method='k_nearest', k=5)
    
    print(f"Distance matrix shape: {dist.shape}")
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Adjacency matrix sparsity: {1 - np.count_nonzero(adj) / (adj.shape[0] * adj.shape[1]):.2%}")
    
    # Convert to PyTorch tensor
    adj_tensor = to_torch_tensor(adj, device='cpu')
    print(f"Tensor shape: {adj_tensor.shape}")
