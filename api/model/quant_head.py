# -*- coding: utf-8 -*-
"""共用分位头 + 因果 RevIN（#7–#10 四个主干完全共享）。

设计约束（见 Baseline补充清单 §六）：

* **分位头完全相同**：LayerNorm → Linear(d,d) → GELU → Linear(d, pre_len*Q)，
  99 个分位，只受主干 hidden 维 d 影响；
* **RevIN 只用 72h 历史**（因果）：逐窗、逐站算 loc/scale，禁止看未来；
* 输出始终是 raw 单位的 ``[B,N,pre_len,Q]``，与 ``PatchTSTQuant`` / ``KairosL1``
  的既有约定一致（head 输出 normalized，再由 ``denorm`` 还原）。

scale 用 ``std + eps`` 并在训练/推理同一条路径上取；不做跨城/跨站全局统计，
避免把测试期信息带进来。
"""
import torch
import torch.nn as nn


class CausalRevIN(nn.Module):
    """逐站、逐窗的 revIN；统计量只来自输入历史 ``x [B,T,N]``。"""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def stats(self, x: torch.Tensor):
        loc = x.mean(dim=1, keepdim=True)                 # [B,1,N]
        scale = x.std(dim=1, unbiased=False, keepdim=True).clamp_min(self.eps)
        return loc, scale

    def forward(self, x: torch.Tensor):
        loc, scale = self.stats(x)
        return (x - loc) / scale, loc, scale

    @staticmethod
    def denorm(y_norm: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """y_norm [B,N,L,Q] 或 [B,N,L]；loc/scale [B,1,N]。"""
        if y_norm.dim() == 4:
            return y_norm * scale.permute(0, 2, 1).unsqueeze(-1) + loc.permute(0, 2, 1).unsqueeze(-1)
        return y_norm * scale.permute(0, 2, 1) + loc.permute(0, 2, 1)


class QuantileHead(nn.Module):
    """与 ``PatchTSTQuant.head`` / ``KairosL1`` 同构的逐站分位头。"""

    def __init__(self, d_model: int, pre_len: int, n_quant: int, dropout: float = 0.1):
        super().__init__()
        self.pre_len = pre_len
        self.n_quant = n_quant
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pre_len * n_quant),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h [B,N,d] -> [B,N,pre_len,Q]。"""
        B, N, _ = h.shape
        return self.net(h).view(B, N, self.pre_len, self.n_quant)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, qs: torch.Tensor,
                 weight: torch.Tensor = None) -> torch.Tensor:
    """raw 空间 pinball（2× 即 CRPS 的经验估计）。

    pred [B,N,L,Q]、target [B,N,L]、weight [B,N] bool（clean mask）。
    """
    err = target.unsqueeze(-1) - pred                       # [B,N,L,Q]
    rho = torch.maximum(qs * err, (qs - 1.0) * err)
    if weight is not None:
        w = weight.to(rho.dtype).unsqueeze(-1).unsqueeze(-1)
        denom = w.expand_as(rho).sum().clamp_min(1.0)
        return (rho * w).sum() / denom
    return rho.mean()
