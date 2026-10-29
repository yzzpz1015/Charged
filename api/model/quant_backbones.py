# -*- coding: utf-8 -*-
"""#7–#10：点预测主干 + **同一个**分位头。

所有模型接口一致::

    y_norm, (loc, scale) = model(x)      # x [B,T,N] raw volume
    pred = model.denorm(y_norm, loc, scale)   # [B,N,pre_len,Q] raw

* :class:`iTransformerQuant`  —— 变元 token，显式跨站注意力（隐式图）
* :class:`STGCNQuant`         —— 静态图卷积；``graph=False`` 时邻接退化为 I，
                                 与带图版**参数量完全相同**（#8 vs #9 的正式对照）
* :class:`GWNetQuant`         —— 自适应图（Graph WaveNet），固定支撑 + 学出来的支撑

主干选择只影响 hidden 的来路；分位头、RevIN、损失、窗口全部共用。
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from api.model.gwnet import GWNet
from api.model.quant_head import CausalRevIN, QuantileHead


# ------------------------------------------------------------------ iTransformer
class iTransformerQuant(nn.Module):
    """iTransformer：每条站序列 → 一个 token，Transformer 在站维做注意力。"""

    def __init__(self, num_nodes: int, seq_len: int, pre_len: int, n_quant: int,
                 d_model: int = 64, nhead: int = 4, n_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.revin = CausalRevIN()
        self.embed = nn.Linear(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = QuantileHead(d_model, pre_len, n_quant, dropout)

    def forward(self, x: torch.Tensor):
        xn, loc, scale = self.revin(x)                 # [B,T,N]
        h = self.embed(xn.transpose(1, 2))             # [B,N,d]
        h = self.norm(self.encoder(h))
        return self.head(h), (loc, scale)

    @staticmethod
    def denorm(y_norm, loc, scale):
        return CausalRevIN.denorm(y_norm, loc, scale)


# ------------------------------------------------------------------ STGCN
class _TemporalGLU(nn.Module):
    """GLU 沿**时间轴**卷积；输入布局 ``[B,C,T,N]``，卷积核必须放在高维 T。"""

    def __init__(self, c_in: int, c_out: int, k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(k, 1),
                              padding=(k // 2, 0))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                              # [B,C,T,N]
        a, b = self.conv(x).chunk(2, dim=1)
        return self.drop(a * torch.sigmoid(b))


class _GraphConv(nn.Module):
    """A @ (X W)，A 为 [N,N] 归一化邻接（含自环）。"""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.w = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x, adj):                         # x [B,C,T,N]
        x = self.w(x)
        return torch.einsum("ij,bctj->bcti", adj, x)


class _STBlock(nn.Module):
    def __init__(self, c: int, dropout: float = 0.1):
        super().__init__()
        self.t1 = _TemporalGLU(c, c, dropout=dropout)
        self.g = _GraphConv(c, c)
        self.t2 = _TemporalGLU(c, c, dropout=dropout)
        self.norm = nn.LayerNorm(c)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        h = self.t1(x)
        h = self.g(h, adj)
        h = self.t2(h)
        h = self.drop(h) + x
        return self.norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    deg = adj.sum(1).clamp_min(1e-12).pow(-0.5)
    return deg[:, None] * adj * deg[None, :]


class STGCNQuant(nn.Module):
    """STGCN：时间 GLU 卷积 + 静态图卷积。``graph=False`` 时 A=I。"""

    def __init__(self, num_nodes: int, seq_len: int, pre_len: int, n_quant: int,
                 d_model: int = 64, n_layers: int = 2, dropout: float = 0.1,
                 adj_matrix: Optional[torch.Tensor] = None, graph: bool = True):
        super().__init__()
        self.revin = CausalRevIN()
        self.inp = nn.Conv2d(1, d_model, kernel_size=(1, 1))
        self.blocks = nn.ModuleList([_STBlock(d_model, dropout) for _ in range(n_layers)])
        self.head = QuantileHead(d_model, pre_len, n_quant, dropout)

        if not graph or adj_matrix is None:
            adj = torch.eye(num_nodes, dtype=torch.float32)
        else:
            adj = adj_matrix.detach().float()
        self.register_buffer("adj", normalize_adj(adj))

    def forward(self, x: torch.Tensor):
        xn, loc, scale = self.revin(x)
        h = self.inp(xn.unsqueeze(1))                  # [B,C,T,N]
        for blk in self.blocks:
            h = blk(h, self.adj)
        h = h[:, :, -1, :].transpose(1, 2)             # [B,N,C]（最后一个时间步）
        return self.head(h), (loc, scale)

    @staticmethod
    def denorm(y_norm, loc, scale):
        return CausalRevIN.denorm(y_norm, loc, scale)


# ------------------------------------------------------------------ Graph WaveNet
class GWNetQuant(nn.Module):
    """Graph WaveNet hidden + 共用分位头。

    邻接**必须在构造时**传入：``GWNet`` 只在 ``__init__`` 里注册
    ``fixed_support`` 缓冲区，事后赋值 ``model.adj_matrix`` 不生效。
    """

    def __init__(self, num_nodes: int, seq_len: int, pre_len: int, n_quant: int,
                 d_model: int = 64, n_layers: int = 3, dropout: float = 0.1,
                 adj_matrix: Optional[torch.Tensor] = None):
        super().__init__()
        self.revin = CausalRevIN()
        self.backbone = GWNet(
            num_nodes=num_nodes, seq_len=seq_len, pred_len=pre_len,
            c_in=1, c_out=1, nhid=d_model, layers=n_layers,
            kernel_size=2, dropout=dropout, gdep=2, supports_len=2,
            adj_matrix=adj_matrix)
        self.head = QuantileHead(d_model, pre_len, n_quant, dropout)

    def forward(self, x: torch.Tensor):
        xn, loc, scale = self.revin(x)
        h = self.backbone(xn.transpose(1, 2), return_hidden=True)   # [B,N,nhid]
        return self.head(h), (loc, scale)

    @staticmethod
    def denorm(y_norm, loc, scale):
        return CausalRevIN.denorm(y_norm, loc, scale)


# ------------------------------------------------------------------ 图工具
def knn_adjacency(distance: torch.Tensor, k: int = 8,
                  self_loop: bool = False) -> torch.Tensor:
    """距离矩阵 [N,N] → kNN 邻接，权重 ``exp(-d / median(d>0))``。

    kNN 是**有向**的（A 选 B 不代表 B 选 A）；这里保留原始方向，再做行归一化前的
    对称化（max(A, Aᵀ)），避免单向边造成信息只进不出。

    默认**不加**自环：STGCN/GWNet 的归一化函数会自己加一次 I。需要自环且下游
    不再加 I 的调用方（DiffSTG、OpenCity 官方图）显式传 ``self_loop=True``。
    """
    n = distance.size(0)
    d = distance.clone().float()
    d.fill_diagonal_(float("inf"))
    med = d[d < float("inf")].median().clamp_min(1e-6)
    vals, idx = torch.topk(-d, k=k, dim=1)             # 最近的 k 个
    adj = torch.zeros(n, n, dtype=torch.float32)
    adj.scatter_(1, idx, torch.exp(vals / med))
    adj = torch.maximum(adj, adj.t())
    if self_loop:
        adj.fill_diagonal_(1.0)
    return adj
