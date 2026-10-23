# -*- coding: utf-8 -*-
"""PatchTST + quantile head (ablation for 做法③).

Reuses HuggingFace ``PatchTSTForPrediction`` as the backbone (RevIN + patch
transformer) and replaces its point head with a per-site quantile head.  The
interface mirrors :class:`api.model.kairos_l1.KairosL1` so the existing
``train_kairos_l1.py`` loop / pinball loss can be reused unchanged:

    y_norm, (loc, scale) = model(x)          # x: [B, T] univariate
    loss = model.quantile_loss(y_norm, y, loc, scale)
    pred = model.denorm(y_norm, loc, scale)  # [B, pre_len, Q]
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForPrediction


class PatchTSTQuant(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pre_len: int,
        quantiles: List[float],
        patch_len: int = 6,
        stride: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pre_len = pre_len
        self.quantiles = [float(q) for q in quantiles]
        Q = len(self.quantiles)

        config = PatchTSTConfig(
            num_input_channels=1,
            context_length=seq_len,
            prediction_length=pre_len,
            patch_length=patch_len,
            patch_stride=stride,
            d_model=d_model,
            num_attention_heads=nhead,
            num_hidden_layers=n_layers,
            ffn_dim=d_ff,
            attention_dropout=dropout,
            ff_dropout=dropout,
            head_dropout=dropout,
            positional_dropout=dropout,
            do_mask_input=False,
        )
        self.backbone = PatchTSTForPrediction(config)
        # Unused point head; keep module construction cheap and explicit.
        self.backbone.head = None

        # Same head shape as KairosL1: LayerNorm -> Linear -> GELU -> Linear.
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pre_len * Q),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """x: [B, T] raw volume -> (y_norm [B, pre_len, Q], (loc [B,1], scale [B,1]))."""
        if x.dim() == 2:
            x = x.unsqueeze(1)                       # [B, 1, T]
        if x.shape[1] != 1:
            raise ValueError('PatchTSTQuant currently supports univariate per-site input only')

        out = self.backbone.model(past_values=x.permute(0, 2, 1), return_dict=True)
        h = out.last_hidden_state.mean(dim=2)        # [B, 1, d_model] (pooling_type='mean')
        y_norm = self.head(h).view(-1, self.pre_len, len(self.quantiles))
        loc = out.loc.squeeze(-1)                    # [B, 1]
        scale = out.scale.squeeze(-1)                # [B, 1]
        return y_norm, (loc, scale)

    def quantile_loss(self, y_norm, y, loc, scale) -> torch.Tensor:
        """Pinball loss in raw space; same convention as KairosL1."""
        q = torch.tensor(self.quantiles, device=y.device, dtype=y.dtype)
        pred = y_norm * scale.unsqueeze(-1) + loc.unsqueeze(-1)
        err = y[:, :, None] - pred
        rho = torch.maximum(q * err, (q - 1.0) * err)
        return rho.mean()

    def denorm(self, y_norm, loc, scale):
        if y_norm.dim() == 3:
            return y_norm * scale.unsqueeze(-1) + loc.unsqueeze(-1)
        return y_norm * scale + loc
