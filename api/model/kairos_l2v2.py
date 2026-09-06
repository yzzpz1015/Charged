# -*- coding: utf-8 -*-
"""Kairos L2 v2: three-stream, layer-by-layer cross-attention injection + fusion.

Corrected architecture (per user): the three modalities run as three parallel
decoder *streams* through the FROZEN Kairos decoder; POI and satellite are
injected into EVERY decoder layer via additional cross-attention (gate-scaled).
Each stream differs only in which external modality is active:

    stream_ts : decoder with native time-series cross-attn only           -> h_ts
    stream_poi: decoder + POI  injection at every layer                   -> h_poi
    stream_sat: decoder + satellite injection at every layer              -> h_sat

  injection at layer l:  h = h + gate_m[l] * CrossAttn_m(h, aux_m)

gates are per-layer scalars init 0 (start = pure time series; train to read how
much the model uses each modality, diagnostic only). Each stream has its own
point head -> y_ts / y_poi / y_sat. y_ts never sees external info, so
y_poi - y_ts / y_sat - y_ts is a CLEAN within-model net-contribution reading
(no gate toggling at eval, no cross-model L1 needed for the per-modality claim).

A pluggable fusion module combines the streams into the final forecast y_collab
(the deployed model; headline comparison vs separately-trained L1).

Fusion variants (--fusion):
  static : y = sum w_i y_i, w = softmax(learnable logits)            [baseline]
  gate   : y = sum w_i(x) y_i, w = softmax(MLP(concat hiddens))      [confidence-style, recommended]
  mlp    : y = MLP(concat hiddens)                                   [hidden-level fusion]
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional, Tuple


def _make_head(d_model: int, pre_len: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, pre_len),
    )


class InjectionAttn(nn.Module):
    """Lightweight cross-attention for external-modality injection.

    Q = decoder hidden state, K = V = aux tokens (per-layer, per-modality).
    Pre-LN, single attention layer. Multi-token aux makes this a real
    cross-attention (query decides which aux tokens to read).
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.ln = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, h: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """h: [B, S, d], aux: [B, K, d] -> [B, S, d]."""
        B, S, _ = h.shape
        K = aux.shape[1]
        x = self.ln(h)
        q = self.q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(aux).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(aux).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o(out)


# ---------------- fusion variants ----------------

class FusionStatic(nn.Module):
    """Learnable equal-init weights over stream predictions (late fusion baseline)."""

    def __init__(self, n_streams: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_streams))

    def forward(self, names: List[str], preds: Dict[str, torch.Tensor],
                hiddens: Dict[str, torch.Tensor]) -> torch.Tensor:
        w = torch.softmax(self.logits, dim=0)
        return sum(w[i] * preds[n] for i, n in enumerate(names))


class FusionGate(nn.Module):
    """Per-sample dynamic gating (confidence-style): weights from stream hiddens."""

    def __init__(self, d_model: int, n_streams: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model * n_streams),
            nn.Linear(d_model * n_streams, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_streams),
        )

    def forward(self, names: List[str], preds: Dict[str, torch.Tensor],
                hiddens: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = torch.cat([hiddens[n] for n in names], dim=-1)
        w = torch.softmax(self.net(h), dim=-1)  # [B, n_streams]
        return sum(w[:, i:i + 1] * preds[n] for i, n in enumerate(names))


class FusionMLP(nn.Module):
    """Hidden-level fusion: MLP over concatenated stream hiddens."""

    def __init__(self, d_model: int, n_streams: int, pre_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model * n_streams),
            nn.Linear(d_model * n_streams, d_model),
            nn.GELU(),
            nn.Linear(d_model, pre_len),
        )

    def forward(self, names: List[str], preds: Dict[str, torch.Tensor],
                hiddens: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = torch.cat([hiddens[n] for n in names], dim=-1)
        return self.net(h)


FUSION_BUILDERS = {
    'static': lambda d, n, pl: FusionStatic(n),
    'gate': lambda d, n, pl: FusionGate(d, n),
    'mlp': lambda d, n, pl: FusionMLP(d, n, pl),
}


class KairosL2(nn.Module):
    """Three-stream layer-wise-injection L2 model. Frozen Kairos trunk.

    Inputs (same interface as stage-1):
        x         : [B, T] time-series history
        poi_feat  : [B, C_poi] per-sample POI density (static per site)
        sat_feat  : [B, C_sat] per-sample satellite semantic (static per site)

    Returns (outs, loc_scale); outs keys = stream names + 'collab', each [B, pre_len].
    """

    def __init__(
        self,
        backbone: nn.Module,
        pre_len: int,
        poi_dim: int = 96,
        sat_dim: int = 768,
        use_poi: bool = True,
        use_sat: bool = True,
        fusion: str = 'gate',
        n_poi_tokens: int = 4,
        n_sat_tokens: int = 4,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_model = backbone.config.d_model
        self.pre_len = pre_len
        self.use_poi = use_poi
        self.use_sat = use_sat
        self.fusion_name = fusion

        # freeze the Kairos trunk (encoder + decoder)
        for p in backbone.parameters():
            p.requires_grad = False

        n_layers = len(backbone.decoder.block)

        # static external vector -> a few tokens (real cross-attention keys)
        self.poi_proj = nn.Linear(poi_dim, n_poi_tokens * self.d_model) if use_poi else None
        self.sat_proj = nn.Linear(sat_dim, n_sat_tokens * self.d_model) if use_sat else None
        self.n_tokens = {'poi': n_poi_tokens, 'sat': n_sat_tokens}

        # per-layer injection cross-attn + per-layer scalar gates (init 0)
        self.inj_ca = nn.ModuleDict()
        self.gates = nn.ParameterDict()
        for m in ('poi', 'sat'):
            if (m == 'poi' and use_poi) or (m == 'sat' and use_sat):
                self.inj_ca[m] = nn.ModuleList([InjectionAttn(self.d_model) for _ in range(n_layers)])
                self.gates[m] = nn.Parameter(torch.zeros(n_layers))

        # streams: ts always; poi/sat only if enabled
        self.streams = ['ts']
        if use_poi:
            self.streams.append('poi')
        if use_sat:
            self.streams.append('sat')

        # per-stream point heads (auxiliary; give the clean per-modality reading)
        self.heads = nn.ModuleDict({s: _make_head(self.d_model, pre_len) for s in self.streams})

        # pluggable fusion
        self.fusion = FUSION_BUILDERS[fusion](self.d_model, len(self.streams), pre_len)

        # hook state, set per stream decode
        self._aux_tokens: Dict[str, torch.Tensor] = {}
        self._active: Dict[str, bool] = {}
        for l in range(n_layers):
            self.backbone.decoder.block[l].register_forward_hook(self._make_injection_hook(l))

    def _make_injection_hook(self, l: int):
        def hook(module, args, output):
            h = output[0]
            dtype = h.dtype
            for m, active in self._active.items():
                if active:
                    aux = self._aux_tokens[m]  # [B, K, d]
                    inj = self.gates[m][l] * self.inj_ca[m][l](h, aux)
                    h = h + inj.to(dtype)
            return (h,) + output[1:]
        return hook

    def _stream_decode(self, input_embeds, attn_mask, hidden, s_features) -> torch.Tensor:
        seq_out, _ = self.backbone.decode(input_embeds, attn_mask, hidden, s_features=s_features)
        return seq_out[:, 0, :]  # [B, d] first decoder segment

    def forward(
        self,
        x: torch.Tensor,
        poi_feat: Optional[torch.Tensor] = None,
        sat_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        hidden, loc_scale, input_embeds, attn_mask, _, s_features = self.backbone.encode(x)

        # aux tokens (same for all streams)
        self._aux_tokens = {}
        if self.use_poi and poi_feat is not None:
            self._aux_tokens['poi'] = self.poi_proj(poi_feat).view(-1, self.n_tokens['poi'], self.d_model)
        if self.use_sat and sat_feat is not None:
            self._aux_tokens['sat'] = self.sat_proj(sat_feat).view(-1, self.n_tokens['sat'], self.d_model)

        hiddens: Dict[str, torch.Tensor] = {}

        # stream_ts: no external injection
        self._active = {'poi': False, 'sat': False}
        hiddens['ts'] = self._stream_decode(input_embeds, attn_mask, hidden, s_features)

        if self.use_poi and 'poi' in self._aux_tokens:
            self._active = {'poi': True, 'sat': False}
            hiddens['poi'] = self._stream_decode(input_embeds, attn_mask, hidden, s_features)

        if self.use_sat and 'sat' in self._aux_tokens:
            self._active = {'poi': False, 'sat': True}
            hiddens['sat'] = self._stream_decode(input_embeds, attn_mask, hidden, s_features)

        self._active = {}

        preds = {s: self.heads[s](hiddens[s]) for s in hiddens}
        collab = self.fusion(self.streams, preds, hiddens)
        outs = dict(preds)
        outs['collab'] = collab
        return outs, loc_scale
