# -*- coding: utf-8 -*-
"""
Graph WaveNet for spatial-temporal forecasting.

This implementation is closer to the original Graph WaveNet structure:
- gated dilated temporal convolutions
- residual and skip connections
- adaptive adjacency learning
- graph convolution over fixed and learned supports

The model keeps the existing workspace interface:
- input: [B, N, T] or [B, C, N, T]
- output: [B, N] for pred_len=1, otherwise [B, N, pred_len]
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric adjacency normalization with self-loops."""
    if adj.dim() != 2 or adj.size(0) != adj.size(1):
        raise ValueError("adj must be a square matrix")
    adj = adj.clone()
    adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    degree = adj.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree.clamp(min=1e-12), -0.5)
    degree_inv_sqrt = torch.diag(degree_inv_sqrt)
    return degree_inv_sqrt @ adj @ degree_inv_sqrt


class nconv(nn.Module):
    """Neighborhood convolution from the original Graph WaveNet code path."""

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,bjct->bict", adj, x)


class linear(nn.Module):
    """Channel-wise 1x1 linear projection."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class mixprop(nn.Module):
    """Original-style mix propagation over a single support."""

    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super().__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.alpha = alpha
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: [B, N, C, T]."""
        out = [x]
        h = x
        for _ in range(self.gdep):
            h = self.alpha * x + (1.0 - self.alpha) * self.nconv(h, adj)
            out.append(h)
        out = torch.cat(out, dim=2)  # [B, N, (gdep+1)C, T]
        out = out.permute(0, 2, 1, 3)  # [B, (gdep+1)C, N, T]
        out = self.mlp(out)
        out = F.dropout(out, self.dropout, training=self.training)
        return out.permute(0, 2, 1, 3)


class gcn(nn.Module):
    """Original-style graph convolution that mixes fixed and adaptive supports."""

    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super().__init__()
        self.mixed = mixprop(c_in, c_out, gdep, dropout, alpha)
        self.mlp = linear(c_out * 2, c_out)

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        outs = [self.mixed(x, support) for support in supports]
        if len(outs) == 1:
            return outs[0].permute(0, 2, 1, 3)

        out = torch.cat(outs, dim=2)  # [B, N, S*C, T]
        out = out.permute(0, 2, 1, 3)  # [B, S*C, N, T]
        return self.mlp(out)  # [B, C_out, N, T]


class GWNetBlock(nn.Module):
    """One Graph WaveNet residual block."""

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        gcn_depth: int,
        dropout: float,
        alpha: float,
    ):
        super().__init__()
        self.filter_conv = nn.Conv2d(
            residual_channels, residual_channels, kernel_size=(1, kernel_size), dilation=(1, dilation)
        )
        self.gate_conv = nn.Conv2d(
            residual_channels, residual_channels, kernel_size=(1, kernel_size), dilation=(1, dilation)
        )
        self.residual_conv = nn.Conv2d(residual_channels, residual_channels, kernel_size=(1, 1))
        self.skip_conv = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1))
        self.gconv = gcn(residual_channels, residual_channels, gcn_depth, dropout, alpha)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(residual_channels)
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor]):
        """x: [B, C, N, T]."""
        residual = x
        pad = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad, 0, 0, 0))

        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        x = filter_out * gate_out

        skip = self.skip_conv(x)
        x = self.gconv(x.permute(0, 2, 1, 3), supports)
        x = self.residual_conv(x)
        x = x + residual[..., -x.size(-1):]
        x = self.dropout(x)

        x_ln = x.permute(0, 2, 3, 1)  # [B, N, T, C]
        x_ln = self.norm(x_ln)
        x = x_ln.permute(0, 3, 1, 2)

        return x, skip


class GWNet(nn.Module):
    """Graph WaveNet with residual, skip, and adaptive adjacency blocks."""

    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        pred_len: int,
        c_in: int = 1,
        c_out: int = 1,
        nhid: int = 32,
        layers: int = 4,
        kernel_size: int = 2,
        dropout: float = 0.3,
        gdep: int = 2,
        supports_len: int = 2,
        adj_matrix: Optional[torch.Tensor] = None,
        node_dim: int = 10,
        alpha: float = 0.05,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_out = c_out
        self.nhid = nhid
        self.layers = layers
        self.supports_len = supports_len
        self.alpha = alpha

        self.start_conv = nn.Conv2d(c_in, nhid, kernel_size=(1, 1))
        self.end_conv1 = nn.Conv2d(nhid, nhid, kernel_size=(1, 1))
        self.end_conv2 = nn.Conv2d(nhid, pred_len, kernel_size=(1, 1))

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(
                GWNetBlock(
                    residual_channels=nhid,
                    skip_channels=nhid,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    gcn_depth=gdep,
                    dropout=dropout,
                    alpha=alpha,
                )
            )

        self.skip_proj = nn.Conv2d(nhid, nhid, kernel_size=(1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if adj_matrix is not None:
            adj = adj_matrix.detach().clone().float()
        else:
            adj = torch.eye(num_nodes, dtype=torch.float32)
        adj = _normalize_adj(adj)
        self.register_buffer("fixed_support", adj)

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, node_dim) * 0.1)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, num_nodes) * 0.1)

    def _build_supports(self) -> List[torch.Tensor]:
        supports = [self.fixed_support]
        if self.supports_len > 1:
            adaptive = torch.relu(torch.mm(self.nodevec1, self.nodevec2))
            adaptive = F.softmax(adaptive, dim=1)
            supports.append(adaptive)
        return supports

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, T] or [B, C, N, T]."""
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, N, T]
        elif x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")

        supports = self._build_supports()

        x = self.start_conv(x)
        skip = None

        for block in self.blocks:
            x, block_skip = block(x, supports)
            block_skip = self.skip_proj(block_skip)
            skip = block_skip if skip is None else skip[..., -block_skip.size(-1):] + block_skip

        if skip is None:
            skip = x

        x = self.relu(skip)
        x = self.end_conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.end_conv2(x)

        # [B, pred_len, N, T] -> take the last time step after causal stack
        x = x[:, :, :, -1]
        x = x.permute(0, 2, 1)  # [B, N, pred_len]

        if self.pred_len == 1:
            return x.squeeze(-1)
        return x
