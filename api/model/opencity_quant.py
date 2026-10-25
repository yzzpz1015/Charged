# -*- coding: utf-8 -*-
"""OpenCity-Plus 适配：72→72 + 逐站分位头。

官方仓库 ``HKUDS/OpenCity``（Apache-2.0）公开 ``OpenCity-plus.pth``。官方模型是
288→288 点预测，末层写死 ``Linear(24*skip_dim, 288)``；本协议是 72→72（6 个
patch），所以只替换末层。主干（patch embedding / 时空注意力 / GCN / Laplacian PE）
权重全部原样加载。

官方 ``Run.py`` 的 ``eval`` 模式就是**冻结主干、只训 ``predictor.linear``**，这里
沿用该口径：

* ``--ft head``：冻结主干，只训分位头；
* ``--ft full``：主干 + 分位头一起微调（低学习率）。

分位头直接吃展平后的逐站表示 ``[B,N,6*skip_dim]``，``τ=0.5`` 即点预测，因此点
预测与概率预测共用同一个头，不额外训练点预测头。

时间特征沿用官方 ``time_add`` 的编码：一天中的分钟 ``(hour+1)*60 ∈ [60,1440]``，
星期 ``ISO 1..7``。输入通道 = ``[flow, minute_of_day, weekday]``；OpenCity 的
``TCH/TCP`` 只取后两维，与训练时的布局一致。
"""
import importlib.util
import os

import numpy as np
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OC_PATH = os.path.join(BASE, "external", "OpenCity", "model", "OpenCity", "OpenCity.py")


def _load_reference():
    spec = importlib.util.spec_from_file_location("_opencity_ref", _OC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OpenCity


OpenCity = _load_reference()


class OpenCityQuant(OpenCity):
    """官方 OpenCity + 72 步输出 + 99 分位头。"""

    def __init__(self, args, dataset_use, device, dim_in=1, n_quant=99,
                 hidden=256, dropout=0.1):
        super().__init__(args, dataset_use, device, dim_in)
        self.n_quant = int(n_quant)
        self.n_patch = int(args.input_window) // 12
        feat = self.n_patch * self.skip_dim
        # 官方头是 24*skip_dim -> 288；72 输入用不到，整头替换（不保留随机点预测头）。
        self.linear = nn.Identity()
        self.qhead = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Linear(feat, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.output_window * self.n_quant),
        )

    def forward(self, input, lbls, select_dataset):
        """input/lbls [B,72,N,3]；返回 raw 空间分位 ``[B,N,72,Q]``。"""
        bs, time_steps, num_nodes, _ = input.size()

        # —— 与官方 forward 完全一致的时空上下文编码 ——
        TCH = input[..., self.output_dim:].long()
        TCP = lbls[..., self.output_dim:].long()
        th, tp = self.patch_embedding_time(torch.cat([TCH, TCP], dim=-1))
        spa = self.spatial_embedding(
            self.lap_mx_dict[select_dataset].to(self.device)
        ).repeat(bs, th.shape[1], 1, 1)
        th = th + spa
        tp = tp + spa

        # 逐窗逐站 instance norm（只用历史；官方原实现）
        x_in = input[..., :self.output_dim]
        means = x_in.mean(1, keepdim=True).detach()
        x_in = x_in - means
        stdev = torch.sqrt(torch.var(x_in, dim=1, keepdim=True, unbiased=False)
                           + 1e-5).detach()
        x_in = x_in / stdev

        enc = self.patch_embedding_flow(x_in)                 # [B,P,N,D]
        adj = self.adj_mx_dict[select_dataset].to(self.device)
        for block in self.encoder_blocks:
            enc = block(enc, enc, enc, th, tp, adj,
                        self.geo_mask_dict[select_dataset].to(self.device),
                        self.sem_mask)

        skip = enc.permute(0, 2, 3, 1).contiguous()           # [B,N,D,P]
        skip = skip.flatten(start_dim=-2)                     # [B,N,D*P]
        q = self.qhead(skip).view(bs, num_nodes, time_steps, self.n_quant)
        # 反归一化：stdev/means [B,1,N,1] -> [B,N,1,1]
        q = q * stdev.squeeze(1).unsqueeze(-1) + means.squeeze(1).unsqueeze(-1)
        return q


# ------------------------------------------------------------------ 图
def laplacian_pe(adj: np.ndarray, k: int = 8) -> np.ndarray:
    """逐字复刻官方 ``lib/predifineGraph.cal_lape``（含 eig，保持符号约定）。"""
    adj = np.asarray(adj, dtype=np.float64)
    d = np.array(adj.sum(1))
    isolated = int(np.sum(np.where(d, 0, 1)))
    dinv = np.power(d, -0.5).flatten()
    dinv[np.isinf(dinv)] = 0.0
    L = np.eye(adj.shape[0]) - (adj * dinv[:, None]) * dinv[None, :]
    w, v = np.linalg.eig(L)
    idx = w.argsort()
    v = np.real(v[:, idx])
    return v[:, isolated + 1: k + isolated + 1].astype(np.float32)


def build_graph(dist: np.ndarray, k: int = 8):
    """距离矩阵 → (A, A_mx, lap_pe)，与官方 ``args.py`` 同公式。"""
    from api.model.quant_backbones import knn_adjacency
    A = knn_adjacency(torch.from_numpy(np.asarray(dist, np.float32)),
                      k=k).numpy().astype(np.float64)
    d = A.sum(1)
    sinv = np.zeros_like(d)
    nz = d > 0
    sinv[nz] = d[nz] ** -0.5
    A_mx = np.eye(A.shape[0]) + (A * sinv[:, None]) * sinv[None, :]
    return A.astype(np.float32), A_mx.astype(np.float32), laplacian_pe(A)


def load_pretrained(model: nn.Module, path: str):
    """加载 OpenCity-Plus；只允许末层（已被替换）缺失。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck = {k.replace("module.predictor.", "", 1): v for k, v in ck.items()}
    ck = {k: v for k, v in ck.items() if not k.startswith("linear.")}
    missing, unexpected = model.load_state_dict(ck, strict=False)
    bad = [k for k in missing if not k.startswith("qhead.")]
    if bad or unexpected:
        raise AssertionError(f"OpenCity 权重异常：missing={bad[:10]} "
                             f"unexpected={unexpected[:10]}")
    return len(ck)
