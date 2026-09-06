# -*- coding: utf-8 -*-
"""ASTGCN paper-style implementation with explicit recent/daily branches."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_cheb_polynomials(adj: torch.Tensor, k: int) -> List[torch.Tensor]:
    """Build K-order Chebyshev polynomials from the normalized Laplacian."""
    n = adj.size(0)
    identity = torch.eye(n, device=adj.device, dtype=adj.dtype)
    laplacian = identity - adj
    try:
        lambda_max = torch.linalg.eigvalsh(laplacian).max().real.clamp(min=1e-6)
    except RuntimeError:
        lambda_max = torch.tensor(2.0, device=adj.device, dtype=adj.dtype)

    scaled_laplacian = (2.0 / lambda_max) * laplacian - identity
    cheb = [identity]
    if k == 1:
        return cheb
    cheb.append(scaled_laplacian)
    for _ in range(2, k):
        cheb.append(2 * scaled_laplacian @ cheb[-1] - cheb[-2])
    return cheb


class TemporalAttention(nn.Module):
    """Temporal attention module from ASTGCN."""

    def __init__(self, num_nodes: int, in_channels: int, seq_len: int):
        super().__init__()
        self.u1 = nn.Parameter(torch.empty(num_nodes))
        self.u2 = nn.Parameter(torch.empty(in_channels, num_nodes))
        self.u3 = nn.Parameter(torch.empty(in_channels))
        self.be = nn.Parameter(torch.empty(1, seq_len, seq_len))
        self.ve = nn.Parameter(torch.empty(seq_len, seq_len))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.u1, -0.1, 0.1)
        nn.init.xavier_uniform_(self.u2)
        nn.init.uniform_(self.u3, -0.1, 0.1)
        nn.init.zeros_(self.be)
        nn.init.xavier_uniform_(self.ve)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C, T]
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.u1), self.u2)  # [B, T, N]
        rhs = torch.matmul(self.u3, x)  # [B, N, T]
        product = torch.matmul(lhs, rhs)  # [B, T, T]
        scores = torch.matmul(self.ve, torch.sigmoid(product + self.be))
        return F.softmax(scores, dim=-1)


class SpatialAttention(nn.Module):
    """Spatial attention module from ASTGCN."""

    def __init__(self, num_nodes: int, in_channels: int, seq_len: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(seq_len))
        self.w2 = nn.Parameter(torch.empty(in_channels, seq_len))
        self.w3 = nn.Parameter(torch.empty(in_channels))
        self.bs = nn.Parameter(torch.empty(1, num_nodes, num_nodes))
        self.vs = nn.Parameter(torch.empty(num_nodes, num_nodes))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.w1, -0.1, 0.1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.uniform_(self.w3, -0.1, 0.1)
        nn.init.zeros_(self.bs)
        nn.init.xavier_uniform_(self.vs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C, T]
        lhs = torch.matmul(torch.matmul(x, self.w1), self.w2)  # [B, N, C]
        rhs = torch.matmul(self.w3, x).transpose(-1, -2)  # [B, T, N]
        product = torch.matmul(lhs, rhs)  # [B, N, N]
        scores = torch.matmul(self.vs, torch.sigmoid(product + self.bs))
        return F.softmax(scores, dim=-1)


class ChebConvWithSpatialAttention(nn.Module):
    """Chebyshev graph convolution guided by spatial attention."""

    def __init__(self, k_order: int, in_channels: int, out_channels: int):
        super().__init__()
        self.k_order = k_order
        self.theta = nn.ParameterList(
            [nn.Parameter(torch.empty(in_channels, out_channels)) for _ in range(k_order)]
        )
        for param in self.theta:
            nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor, s_att: torch.Tensor, cheb_polys: List[torch.Tensor]) -> torch.Tensor:
        # x: [B, N, C, T], s_att: [B, N, N]
        batch_size, num_nodes, _, time_steps = x.shape
        outputs = []
        out_channels = self.theta[0].size(1)

        for time_step in range(time_steps):
            graph_signal = x[:, :, :, time_step]  # [B, N, C]
            out = torch.zeros(batch_size, num_nodes, out_channels, device=x.device, dtype=x.dtype)
            for k in range(self.k_order):
                cheb_k = cheb_polys[k]  # [N, N]
                cheb_k_att = cheb_k.unsqueeze(0) * s_att  # [B, N, N]
                rhs = torch.matmul(cheb_k_att.transpose(1, 2), graph_signal)  # [B, N, C]
                out = out + torch.matmul(rhs, self.theta[k])
            outputs.append(out.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))  # [B, N, C_out, T]


class ASTGCNBlock(nn.Module):
    """One ASTGCN block: temporal attention -> spatial attention -> ChebConv -> time conv."""

    def __init__(
            self,
            num_nodes: int,
            in_channels: int,
            k_order: int,
            nb_chev_filter: int,
            nb_time_filter: int,
            seq_len: int,
    ) -> None:
        super().__init__()
        self.t_att = TemporalAttention(num_nodes, in_channels, seq_len)
        self.s_att = SpatialAttention(num_nodes, in_channels, seq_len)
        self.cheb_conv = ChebConvWithSpatialAttention(k_order, in_channels, nb_chev_filter)

        self.time_conv = nn.Conv2d(
            in_channels=nb_chev_filter,
            out_channels=nb_time_filter,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
        )
        self.residual_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=nb_time_filter,
            kernel_size=(1, 1),
            stride=(1, 1),
        )
        self.layer_norm = nn.LayerNorm(nb_time_filter)

    def forward(self, x: torch.Tensor, cheb_polys: List[torch.Tensor]) -> torch.Tensor:
        # x: [B, N, C, T]
        batch_size, num_nodes, channels, time_steps = x.shape

        time_attention = self.t_att(x)  # [B, T, T]
        x_time = torch.matmul(x.reshape(batch_size, -1, time_steps), time_attention).reshape(
            batch_size, num_nodes, channels, time_steps
        )

        spatial_attention = self.s_att(x_time)  # [B, N, N]
        x_graph = self.cheb_conv(x_time, spatial_attention, cheb_polys)  # [B, N, Cg, T]

        x_time_conv = self.time_conv(x_graph.permute(0, 2, 1, 3))
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        output = F.relu(x_time_conv + x_residual)

        output = output.permute(0, 2, 3, 1)
        output = self.layer_norm(output)
        return output.permute(0, 1, 3, 2)


class ASTGCNBranch(nn.Module):
    """One independent ASTGCN branch for a single temporal view."""

    def __init__(
            self,
            num_nodes: int,
            seq_len: int,
            pred_len: int,
            in_channels: int,
            k_order: int,
            nb_chev_filter: int,
            nb_time_filter: int,
            num_blocks: int,
    ) -> None:
        super().__init__()

        blocks = [
            ASTGCNBlock(
                num_nodes=num_nodes,
                in_channels=in_channels,
                k_order=k_order,
                nb_chev_filter=nb_chev_filter,
                nb_time_filter=nb_time_filter,
                seq_len=seq_len,
            )
        ]

        for _ in range(1, num_blocks):
            blocks.append(
                ASTGCNBlock(
                    num_nodes=num_nodes,
                    in_channels=nb_time_filter,
                    k_order=k_order,
                    nb_chev_filter=nb_chev_filter,
                    nb_time_filter=nb_time_filter,
                    seq_len=seq_len,
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.final_conv = nn.Conv2d(
            in_channels=seq_len,
            out_channels=pred_len,
            kernel_size=(1, nb_time_filter),
        )

    def forward(self, x: torch.Tensor, cheb_polys: List[torch.Tensor]) -> torch.Tensor:
        # x: [B, N, C, T]
        for block in self.blocks:
            x = block(x, cheb_polys)

        x = x.permute(0, 3, 1, 2)  # [B, T, N, C]
        x = self.final_conv(x).squeeze(-1)  # [B, pred_len, N]
        return x.permute(0, 2, 1)  # [B, N, pred_len]


class ASTGCN(nn.Module):
    """Paper-style ASTGCN with independent recent and daily branches."""

    def __init__(
            self,
            num_nodes: int,
            seq_len: int,
            pred_len: int,
            c_in: int = 1,
            c_out: int = 1,
            nhid: int = 64,
            layers: int = 3,
            num_heads: int = 8,
            dropout: float = 0.0,
            k_order: int = 3,
                recent_len: int = 24,
                daily_len: int = 24,
                daily_interval: int = 24,
            device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = device or torch.device('cpu')
        self.adj_matrix: Optional[torch.Tensor] = None
        self.k_order = k_order

        self.recent_len = recent_len
        self.daily_len = daily_len
        self.daily_interval = daily_interval

        if seq_len != recent_len + daily_len:
            raise ValueError(
                "ASTGCN expects seq_len to equal recent_len + daily_len for the two-branch sampler"
            )

        # Keep compatibility with the existing factory signature.
        _ = num_heads
        _ = dropout
        _ = c_out

        self.recent_branch = ASTGCNBranch(
            num_nodes=num_nodes,
            seq_len=recent_len,
            pred_len=pred_len,
            in_channels=c_in,
            k_order=k_order,
            nb_chev_filter=nhid,
            nb_time_filter=nhid,
            num_blocks=layers,
        )
        self.daily_branch = ASTGCNBranch(
            num_nodes=num_nodes,
            seq_len=daily_len,
            pred_len=pred_len,
            in_channels=c_in,
            k_order=k_order,
            nb_chev_filter=nhid,
            nb_time_filter=nhid,
            num_blocks=layers,
        )

        self.w_recent = nn.Parameter(torch.ones(1, num_nodes, pred_len))
        self.w_daily = nn.Parameter(torch.ones(1, num_nodes, pred_len))

    def _unpack_inputs(
            self,
            x: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, dict):
            if {'recent', 'daily'} - set(x.keys()):
                raise ValueError("ASTGCN dict input must contain recent and daily")
            return x['recent'], x['daily']
        if isinstance(x, (tuple, list)):
            if len(x) != 2:
                raise ValueError("ASTGCN tuple/list input must contain 2 tensors: (recent, daily)")
            return x[0], x[1]
        raise TypeError("ASTGCN expects a dict or tuple/list with recent and daily inputs")

    def forward(
            self,
            x: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]],
            adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        recent, daily = self._unpack_inputs(x)

        for branch_input in (recent, daily):
            if branch_input.dim() != 3:
                raise ValueError("Each ASTGCN branch input must be 3D: [B, N, T]")

        if adj is None:
            if self.adj_matrix is None:
                raise ValueError("adj is required for ASTGCN forward when self.adj_matrix is not set")
            adj = self.adj_matrix
        adj = adj.to(recent.device)

        cheb_polys = _build_cheb_polynomials(adj, k=self.k_order)

        recent = recent.unsqueeze(2)  # [B, N, 1, Tr]
        daily = daily.unsqueeze(2)    # [B, N, 1, Td]

        y_recent = self.recent_branch(recent, cheb_polys)
        y_daily = self.daily_branch(daily, cheb_polys)

        weights = torch.softmax(
            torch.stack([self.w_recent, self.w_daily], dim=0),
            dim=0,
        )
        y = weights[0] * y_recent + weights[1] * y_daily

        if self.pred_len == 1:
            return y.squeeze(-1)
        return y
