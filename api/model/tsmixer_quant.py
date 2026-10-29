# -*- coding: utf-8 -*-
"""#22 TSMixer：全 MLP 的时间/特征混合主干 + 共用分位头。

按 Chen et al. (2023) 的 TSMixer 结构重写：

* **time-mixing**：对每条站序列沿时间轴做 MLP（T → d_ff → T），所有站共享；
* **feature-mixing**：在每个时间步沿站轴做 MLP（N → d_ff → N），所有时刻共享；
* 两者都是 pre-norm + residual，堆 ``n_layers`` 层；
* 之后用 ``Linear(T, d_model)`` 取逐站表示，接与 #7–#13 完全相同的
  ``QuantileHead``（99 分位、72 步），损失为 raw 空间 pinball。

与原始 TSMixer 的差异（表注写明）：原版用线性 head 直接输出 H 步点值，这里
换成共享的分位头以保持概率评测口径一致；特征混合作用在站轴（N 为变量维），
因此**没有距离图先验**，与 iTransformer 一样属于“隐式跨站混合”，但 TSMixer
的混合是稠密 MLP、不含注意力。
"""
import torch
import torch.nn as nn

from api.model.quant_head import CausalRevIN, QuantileHead


class _TSMixerBlock(nn.Module):
    """pre-norm residual 的 time-mixing + feature-mixing。"""

    def __init__(self, seq_len: int, num_nodes: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm_t = nn.LayerNorm(seq_len)
        self.mix_t = nn.Sequential(
            nn.Linear(seq_len, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, seq_len), nn.Dropout(dropout),
        )
        self.norm_n = nn.LayerNorm(num_nodes)
        self.mix_n = nn.Sequential(
            nn.Linear(num_nodes, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, num_nodes), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [B,T,N]
        h = x.permute(0, 2, 1)                               # [B,N,T]
        h = h + self.mix_t(self.norm_t(h))
        h = h.permute(0, 2, 1)                               # [B,T,N]
        h = h + self.mix_n(self.norm_n(h))
        return h


class TSMixerQuant(nn.Module):
    def __init__(self, num_nodes: int, seq_len: int, pre_len: int, n_quant: int,
                 d_model: int = 64, n_layers: int = 2, d_ff: int = 128,
                 dropout: float = 0.1, **_):
        super().__init__()
        self.revin = CausalRevIN()
        self.blocks = nn.ModuleList([
            _TSMixerBlock(seq_len, num_nodes, d_ff, dropout)
            for _ in range(n_layers)])
        self.embed = nn.Linear(seq_len, d_model)
        self.head = QuantileHead(d_model, pre_len, n_quant, dropout)

    def forward(self, x: torch.Tensor):
        B, T, N = x.shape
        xn, loc, scale = self.revin(x)                       # [B,T,N]
        h = xn
        for blk in self.blocks:
            h = blk(h)
        h = self.embed(h.transpose(1, 2))                    # [B,N,d]
        return self.head(h), (loc, scale)

    @staticmethod
    def denorm(y_norm, loc, scale):
        return CausalRevIN.denorm(y_norm, loc, scale)
