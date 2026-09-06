# -*- coding: utf-8 -*-
"""Kairos L2 v3: three EXTERNAL streams (POI / satellite image / satellite description)
layer-by-layer cross-attention injected into the frozen decoder + collaboration.

Corrected path composition (per user): the three paths are the three external
modalities — there is NO pure-time-series stream. The frozen Kairos backbone
keeps its native time-series cross-attention (the decoder's eyes, always on);
each external path is injected into EVERY decoder layer via a gate-scaled
cross-attention and forms its own stream with its own point head:

    stream_poi : decoder + POI  injection every layer          -> h_poi  -> y_poi
    stream_img : decoder + satellite-image injection every layer-> h_img  -> y_img
    stream_desc: decoder + satellite-description injection      -> h_desc -> y_desc

  injection at layer l:  h = h + gate_m[l] * CrossAttn_m(h, aux_m)

The three stream hiddens are combined by a collaboration module (--fusion) into
y_collab, the deployed forecast. Baselines:
  * headline   : collab R^2 vs separately-trained L1 (pure TS, no external path)
  * per-modal  : y_poi / y_img / y_desc vs each other (same-model, clean: each
                 stream head's own loss never touches the other modalities)
  * reference  : 'gate' late fusion, to show the fancy collabs add value

Collaboration schemes:
  gate  : softmax(MLP(concat hiddens)) weights over stream preds   [reference]
  urban : UrbanAgent-style confidence iteration over 3 interaction edges
  align : align img<->desc hiddens (mutual cross-attn), then combine with POI
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_head(d_model: int, pre_len: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, pre_len),
    )


class InjectionAttn(nn.Module):
    """Lightweight cross-attention (pre-LN single head). Q = query hidden state,
    K = V = aux tokens. Reused both for layer injection and collaboration edges."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1,
                 zero_init_o: bool = False):
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
        if zero_init_o:
            # LoRA-style: output zero-init -> forward starts at identity for any gate.
            nn.init.zeros_(self.o.weight)
            nn.init.zeros_(self.o.bias)
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


# ---------------- collaboration modules ----------------

class CollabGate(nn.Module):
    """Reference late fusion: per-sample dynamic weights from stream hiddens."""

    def __init__(self, d_model: int, n_streams: int, pre_len: int = None):
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
        w = torch.softmax(self.net(h), dim=-1)
        return sum(w[:, i:i + 1] * preds[n] for i, n in enumerate(names))


class CollabUrban(nn.Module):
    """UrbanAgent-style confidence iteration over the three interaction edges.

    Start from the mean of the three stream hiddens (each already encodes TS via
    the frozen backbone's native cross-attn). For R rounds, each modality emits a
    confidence and a cross-attn edge into the central state:

        c <- c + sigma(conf_i(c)) * CrossAttn(c, h_i)      i in {poi, img, desc}
    """

    def __init__(self, d_model: int, n_streams: int, pre_len: int, rounds: int = 3):
        super().__init__()
        self.rounds = rounds
        self.conf = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(n_streams)])
        self.edges = nn.ModuleList([InjectionAttn(d_model) for _ in range(n_streams)])
        self.head = _make_head(d_model, pre_len)

    def forward(self, names: List[str], preds: Dict[str, torch.Tensor],
                hiddens: Dict[str, torch.Tensor]) -> torch.Tensor:
        hs = [hiddens[n] for n in names]
        c = torch.stack(hs).mean(dim=0)                      # [B, d]
        for _ in range(self.rounds):
            for i, h in enumerate(hs):
                conf = torch.sigmoid(self.conf[i](c))        # [B, 1]
                edge = self.edges[i](c.unsqueeze(1), h.unsqueeze(1))[:, 0]  # [B, d]
                c = c + conf * edge
        return self.head(c)


class CollabAlign(nn.Module):
    """Align satellite image & description streams BY TRAINING, then trust-weight vs POI.

    NOT architectural fusion (no mutual cross-attn at inference). The two
    satellite-view streams (img, desc) are pulled together by an auxiliary
    alignment loss computed in the trainer on the stream hiddens:

        L_align = 1 - cos(h_img, h_desc)     (or MSE, --align_loss)

    Its gradient flows back through the frozen decoder's forward hooks into the
    two streams' layer-injection cross-attns (inj_ca['img'] / inj_ca['desc']),
    shaping them to extract mutually consistent satellite evidence. At forward
    time the aligned satellite representation is just the mean; a learned
    per-sample confidence gate combines it with POI (independent source).
    """

    def __init__(self, d_model: int, n_streams: int, pre_len: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.head = _make_head(d_model, pre_len)

    def forward(self, names: List[str], preds: Dict[str, torch.Tensor],
                hiddens: Dict[str, torch.Tensor]) -> torch.Tensor:
        h_img = hiddens['img']
        h_desc = hiddens['desc']
        h_poi = hiddens['poi']
        h_sat = 0.5 * (h_img + h_desc)                    # 训练后对齐的卫星一致表示
        g = self.gate(torch.cat([h_sat, h_poi], dim=-1))  # [B, 1] trust sat vs poi
        c = g * h_sat + (1 - g) * h_poi
        return self.head(c)


FUSION_BUILDERS = {
    'gate': CollabGate,
    'urban': CollabUrban,
    'align': CollabAlign,
}


class KairosL2V3(nn.Module):
    """Three-external-stream layer-wise-injection L2 model. Frozen Kairos trunk.

    Inputs:
        x        : [B, T] time-series history
        poi_feat : [B, C_poi]  per-site POI density (static)
        img_feat : [B, C_img]  per-site satellite image embedding (static)
        desc_ids : [B, S_desc] GPT-2 token ids of the satellite description

    Returns (outs, loc_scale); outs keys = stream names + 'collab', each [B, pre_len].
    """

    def __init__(
        self,
        backbone: nn.Module,
        gpt2_wte: nn.Embedding,
        pre_len: int,
        poi_dim: int = 96,
        img_dim: int = 768,
        n_poi_tokens: int = 4,
        n_img_tokens: int = 4,
        use_poi: bool = True,
        use_img: bool = True,
        use_desc: bool = True,
        fusion: str = 'urban',
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.gpt2_wte = gpt2_wte
        self.d_model = backbone.config.d_model
        self.pre_len = pre_len
        self.use_poi = use_poi
        self.use_img = use_img
        self.use_desc = use_desc

        for p in backbone.parameters():
            p.requires_grad = False
        gpt2_wte.weight.requires_grad = False

        n_layers = len(backbone.decoder.block)

        # external feature -> tokens (real cross-attention keys)
        self.poi_proj = nn.Linear(poi_dim, n_poi_tokens * self.d_model) if use_poi else None
        self.img_proj = nn.Linear(img_dim, n_img_tokens * self.d_model) if use_img else None
        # description: gpt2 text tokens -> d_model (one shared proj over token dim)
        self.desc_embed = nn.Linear(gpt2_wte.weight.shape[1], self.d_model) if use_desc else None
        self.n_tokens = {'poi': n_poi_tokens, 'img': n_img_tokens}

        # per-layer injection cross-attn + per-layer scalar gates.
        # gate_init=1 + O zero-init (LoRA-style): forward starts at the frozen
        # backbone exactly, yet injection params get FULL gradient (not killed
        # by gate=0), so the injection can actually learn. gate=0 legacy keeps
        # the injection params at zero gradient until the gate drifts off 0.
        self.inj_ca = nn.ModuleDict()
        self.gates = nn.ParameterDict()
        for m, on in (('poi', use_poi), ('img', use_img), ('desc', use_desc)):
            if on:
                self.inj_ca[m] = nn.ModuleList([
                    InjectionAttn(self.d_model, zero_init_o=(gate_init == 1.0))
                    for _ in range(n_layers)])
                self.gates[m] = nn.Parameter(torch.full((n_layers,), float(gate_init)))

        self.streams = [m for m, on in (('poi', use_poi), ('img', use_img), ('desc', use_desc)) if on]
        self.heads = nn.ModuleDict({s: _make_head(self.d_model, pre_len) for s in self.streams})

        self.fusion = FUSION_BUILDERS[fusion](self.d_model, len(self.streams), pre_len)

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
                    aux = self._aux_tokens[m]
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
        img_feat: Optional[torch.Tensor] = None,
        desc_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        hidden, loc_scale, input_embeds, attn_mask, _, s_features = self.backbone.encode(x)

        self._aux_tokens = {}
        if self.use_poi and poi_feat is not None:
            self._aux_tokens['poi'] = self.poi_proj(poi_feat).view(
                -1, self.n_tokens['poi'], self.d_model)
        if self.use_img and img_feat is not None:
            self._aux_tokens['img'] = self.img_proj(img_feat).view(
                -1, self.n_tokens['img'], self.d_model)
        if self.use_desc and desc_ids is not None:
            emb = self.gpt2_wte(desc_ids)                      # [B, S_desc, 768]
            self._aux_tokens['desc'] = self.desc_embed(emb)    # [B, S_desc, d]

        hiddens: Dict[str, torch.Tensor] = {}
        for s in self.streams:
            self._active = {m: (m == s) for m in self.streams}
            hiddens[s] = self._stream_decode(input_embeds, attn_mask, hidden, s_features)
        self._active = {}

        preds = {s: self.heads[s](hiddens[s]) for s in self.streams}
        collab = self.fusion(self.streams, preds, hiddens)
        outs = dict(preds)
        outs['collab'] = collab
        return outs, loc_scale, hiddens
