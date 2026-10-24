# -*- coding: utf-8 -*-
"""#13 MQ-CNN：逐站因果空洞 CNN + 直接多 horizon 分位输出。

GluonTS 0.14 的 torch 侧没有 MQ-CNN（只有 MXNet 版），这里按原论文/官方
MXNet 实现的结构重写：

* 输入只有历史 72h（逐站单变量，batch 内置站维）；
* 一层 stem conv + 若干**因果**空洞残差卷积（dilation 1,2,4,8,16）；
* 取最后一个时间步的表示，接与 #7–#10 完全相同的 ``QuantileHead``，
  一次输出 72×99，训练用 raw 空间 pinball（2× 即 CRPS）。

与原版的差异（表注写明）：没有 static categorical embedding（站点身份），
与 PatchTST 基线保持同输入；损失、分位头、RevIN、窗口与其余 baseline 一致。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from api.model.quant_head import CausalRevIN, QuantileHead


class _CausalResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = dilation * 2                     # kernel=3 → (k-1)*d
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B,C,T]
        h = F.pad(x, (self.pad, 0))
        h = self.conv(h)
        h = F.gelu(h)
        h = self.drop(h)
        return x + h


class MQCNNQuant(nn.Module):
    def __init__(self, num_nodes: int, seq_len: int, pre_len: int, n_quant: int,
                 d_model: int = 64, n_layers: int = 5, dropout: float = 0.1,
                 **_):
        super().__init__()
        del num_nodes, n_layers                     # 结构与站数无关
        self.revin = CausalRevIN()
        self.stem = nn.Conv1d(1, d_model, 3, padding=1)
        self.blocks = nn.ModuleList(
            [_CausalResBlock(d_model, 2 ** i, dropout) for i in range(5)])
        self.head = QuantileHead(d_model, pre_len, n_quant, dropout)

    def forward(self, x: torch.Tensor):
        B, T, N = x.shape
        xn, loc, scale = self.revin(x)               # [B,T,N]
        h = xn.permute(0, 2, 1).reshape(B * N, 1, T)  # [B*N,1,T]
        h = F.gelu(self.stem(h))
        for blk in self.blocks:
            h = blk(h)
        h = h[:, :, -1].view(B, N, -1)                # [B,N,d]
        return self.head(h), (loc, scale)

    @staticmethod
    def denorm(y_norm, loc, scale):
        return CausalRevIN.denorm(y_norm, loc, scale)
