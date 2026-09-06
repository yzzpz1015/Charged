# -*- coding: utf-8 -*-
"""Kairos L2: frozen trunk + pure-time-series anchor + external-modality injection.

Builds on L1-plain (frozen Kairos encoder/decoder + point head). Each external
modality is a *static per-site feature* injected into the decoder output with a
scalar gate initialised to 0:

    h_i    = h_base + gate_i * proj_i(aux_i)
    y_i    = head_i(h_i)

The anchor (no injection) and each modality have their own head, so we can read
off per-path R2 and per-path gate value: gate > 0 means the modality contributes
increment beyond what the time series already encodes (POI/satellite are NOT in
the history, so a positive gate is the hoped-for signal).

Modalities (stage 1):
  - POI           : density vector [C_poi] (log1p + minmax, from poi.csv)
  - sat semantic  : DINOv2 CLS [768] (L2-normalised by the loader)
  - sat raw (later): patch tokens via cross-attention
"""
import torch
import torch.nn as nn

from typing import Dict, Optional, Tuple


def _make_head(d_model: int, pre_len: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, pre_len),
    )


class KairosL2(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pre_len: int,
        poi_dim: int = 96,
        sat_dim: int = 768,
        use_poi: bool = True,
        use_sat: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_model = backbone.config.d_model
        self.pre_len = pre_len
        self.use_poi = use_poi
        self.use_sat = use_sat

        # freeze trunk
        for p in backbone.parameters():
            p.requires_grad = False

        # pure time-series anchor head
        self.head = _make_head(self.d_model, pre_len)

        # POI path
        self.poi_proj = None
        self.poi_gate = None
        self.poi_head = None
        if use_poi:
            self.poi_proj = nn.Linear(poi_dim, self.d_model)
            self.poi_gate = nn.Parameter(torch.zeros(1))
            self.poi_head = _make_head(self.d_model, pre_len)

        # satellite semantic path
        self.sat_proj = None
        self.sat_gate = None
        self.sat_head = None
        if use_sat:
            self.sat_proj = nn.Linear(sat_dim, self.d_model)
            self.sat_gate = nn.Parameter(torch.zeros(1))
            self.sat_head = _make_head(self.d_model, pre_len)

    def forward(
        self,
        x: torch.Tensor,
        poi_feat: Optional[torch.Tensor] = None,
        sat_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """x: [B, T] -> ({name: y_norm [B, pre_len]}, loc_scale)."""
        hidden, loc_scale, input_embeds, attn_mask, _, s_features = self.backbone.encode(x)
        seq_out, _ = self.backbone.decode(input_embeds, attn_mask, hidden, s_features=s_features)
        h_base = seq_out[:, 0, :]  # [B, d_model] first decoder segment

        outs: Dict[str, torch.Tensor] = {'anchor': self.head(h_base)}

        if self.use_poi and poi_feat is not None:
            h_poi = h_base + self.poi_gate * self.poi_proj(poi_feat)
            outs['poi'] = self.poi_head(h_poi)

        if self.use_sat and sat_feat is not None:
            h_sat = h_base + self.sat_gate * self.sat_proj(sat_feat)
            outs['sat'] = self.sat_head(h_sat)

        return outs, loc_scale
