"""
EV-VLM: Electric Vehicle Vision-Language Model for Charging Demand Forecasting

Architecture (PKVC-optimized):
    1. Temporal Patch Embedding → [B,N,L,d]
    2. PrototypeSemanticLayer → Semantic-aligned patches [B,N,L,d]
    3. DynamicTextInjector: GPT-2 with prefix KV cache + chunked stats-only forward
       - prefix_hidden: [B,82,768], prefix_pkvc: [B,12,2,12,82,64]
       - Only ~26 stats tokens pass through GPT-2 per site-chunk
       - Output: Q [B,N,L,d] (for satellite), h_cond [B,N,L,d] (fused)
    4. StaticVisionTextEncoder: SigLIP vision+text as unified KV memory [B,N,M+S_s,d]
    5. CrossModalGateFusion: gated fusion [h_cond; C] → G · C
    6. RefinementBlock: TransformerEncoderLayer
    7. ForecastHead: MLP → [B,N,pred_len]
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import PatchTSTConfig, PatchTSTModel


class PatchTSTBackbone(nn.Module):
    """PatchTST encoder wrapper: RevIN + Patching + Transformer Encoder.

    Uses HuggingFace PatchTSTModel to get encoder features [B, N, L, d_model]
    plus loc/scale for denormalization.

    Config: patch_len=6, stride=3, 3 layers, pre-norm BatchNorm, d_model=128.
    """

    def __init__(self, seq_len: int, num_vars: int, d_model: int = 128,
                 n_layers: int = 3, nhead: int = 8, d_ff: int = 512,
                 patch_len: int = 6, stride: int = 3, dropout: float = 0.2):
        super().__init__()
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.d_model = d_model

        config = PatchTSTConfig(
            num_input_channels=num_vars,
            context_length=seq_len,
            prediction_length=1,  # unused — we only use the encoder
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
            norm_type='batchnorm',
            pre_norm=True,
        )
        self.patchtst = PatchTSTModel(config)

    def forward(self, feat: torch.Tensor):
        """feat: [B, N, T] → h: [B, N, L, d_model], loc: [B, 1, N], scale: [B, 1, N]"""
        # HF expects past_values [B, T, N]
        out = self.patchtst(past_values=feat.permute(0, 2, 1))
        # out.last_hidden_state: [B, N, L, d_model]
        return out.last_hidden_state, out.loc, out.scale


class PrototypeSemanticLayer(nn.Module):
    """Prototype-based MHA aligned to LLM semantic space."""

    """CALF-style: project time-series patches onto PCA principal directions of GPT-2 wte.

    Offline: PCA(gpt2.wte) → principal_emb [P, 768] saved as .pt.
    Online: principal = Linear(768→d)(principal_emb), then MHA(Q=x_patch, K=principal, V=principal).
    Residual: h_ts = x_patch + alpha * h_sem, with alpha a learnable scalar (default 0.1).

    buffer:
        principal_emb: [P, 768] — PCA components, frozen.
    trainable:
        principal_proj: Linear(768 → d_model)
        alpha: scalar
        q_norm, kv_norm: LayerNorm
        mha: MultiheadAttention
    """

    def __init__(self, d_model: int, num_heads: int, num_prototypes: int = 500,
                 principal_emb: torch.Tensor = None):
        super().__init__()
        self.d_model = d_model

        if principal_emb is not None:
            self.register_buffer('principal_emb', principal_emb)
        else:
            # Fallback: random orthogonal init of same shape
            self.register_buffer('principal_emb', torch.randn(num_prototypes, 768))

        self.principal_proj = nn.Linear(768, d_model)
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_patch: torch.Tensor) -> torch.Tensor:
        """x_patch: [B, N, L, d] → h_ts: [B, N, L, d]

        h_ts = x_patch + alpha * MHA(q_norm(x_patch), kv_norm(principal), kv_norm(principal))
        """
        B, N, L, d = x_patch.shape
        B_total = B * N

        principal = self.principal_proj(self.principal_emb)               # [P, d]
        principal = principal.unsqueeze(0).expand(B_total, -1, -1)        # [B*N, P, d]

        x_flat = x_patch.view(B_total, L, d)                              # [B*N, L, d]

        h_sem, _ = self.mha(
            query=self.q_norm(x_flat),
            key=self.kv_norm(principal),
            value=self.kv_norm(principal),
        )
        h_ts = x_patch + self.alpha * h_sem.view(B, N, L, d)
        return h_ts


class DynamicTextInjector(nn.Module):
    """Dynamic text injection with PKVC (prefix key-value cache)."""

    def __init__(self, d_model: int, num_heads: int, gpt2_path: str = None):
        super().__init__()
        self.d_model = d_model
        self.gpt_dim = 768

        from transformers import GPT2Model, GPT2Tokenizer
        model_name = gpt2_path if gpt2_path else 'gpt2'
        self.text_encoder = GPT2Model.from_pretrained(model_name)
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.text_proj = nn.Linear(self.gpt_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        h_ts: torch.Tensor,
        prefix_hidden: torch.Tensor,
        prefix_pkvc: torch.Tensor,
        stats_ids: torch.Tensor,
        stats_mask: torch.Tensor
    ) -> tuple:
        """
        Args:
            h_ts: [B, N, L, d]
            prefix_hidden: [B, S_pfx, 768]
            prefix_pkvc: [B, 12, 2, 12, S_pfx, 64]
            stats_ids: [B, N, S_stats] (~26 tokens)
            stats_mask: [B, N, S_stats]
        Returns:
            Q: [B, N, L, d] - Query for satellite cross-attn
            h_cond: [B, N, L, d] - Fused representation
        """
        B, N, L, d = h_ts.shape
        B_total = B * N
        S_pfx = prefix_hidden.shape[1]

        from transformers.cache_utils import DynamicCache
        chunk_size = 8  # Memory-efficient chunking

        all_stats_out = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            C = end - start

            # Expand PKVC from [B, 12, 2, 12, S_pfx, 64] to [B*C, 12, 2, 12, S_pfx, 64]
            # PKVC is per-batch-item (shared across all N sites), repeat for each site in chunk
            pkvc_chunk = prefix_pkvc.unsqueeze(1).expand(-1, C, -1, -1, -1, -1, -1)
            pkvc_flat = pkvc_chunk.reshape(B * C, 12, 2, 12, S_pfx, prefix_pkvc.shape[-1])

            past_kv = DynamicCache()
            for l in range(12):
                past_kv.update(pkvc_flat[:, l, 0], pkvc_flat[:, l, 1], l)

            # Stats for this chunk
            stats_ids_chunk = stats_ids[:, start:end].reshape(B * C, -1)
            stats_mask_chunk = stats_mask[:, start:end].reshape(B * C, -1)
            full_attn = torch.cat([
                torch.ones(B * C, S_pfx, device=h_ts.device, dtype=stats_mask_chunk.dtype),
                stats_mask_chunk
            ], dim=1)

            with torch.no_grad():
                out = self.text_encoder(
                    input_ids=stats_ids_chunk,
                    attention_mask=full_attn,
                    past_key_values=past_kv,
                    use_cache=False,
                    return_dict=True
                ).last_hidden_state
            all_stats_out.append(out)

        stats_out = torch.cat(all_stats_out, dim=0)  # [B*N, S_stats, d]

        # Concat prefix hidden + stats
        prefix_hidden_exp = prefix_hidden.unsqueeze(1).expand(-1, N, -1, -1).reshape(B_total, -1, self.gpt_dim)
        dyn_emb = torch.cat([prefix_hidden_exp, stats_out], dim=1)  # [B*N, 108, gpt_dim]
        dyn_emb = self.text_proj(dyn_emb)  # [B*N, 108, d]

        # Cross-attention: h_ts queries text
        h_ts_flat = h_ts.reshape(B_total, L, d)
        h_dyn, _ = self.cross_attn(
            query=self.norm_q(h_ts_flat),
            key=self.norm_kv(dyn_emb),
            value=self.norm_kv(dyn_emb)
        )

        h_cond = h_ts_flat + h_dyn
        Q = self.query_proj(h_cond)

        return Q.view(B, N, L, d), h_cond.view(B, N, L, d)


class StaticVisionTextEncoder(nn.Module):
    """SigLIP vision + text encoder as unified KV memory."""

    def __init__(self, d_model: int, model_name: str = "google/siglip2-base-patch16-224"):
        super().__init__()
        self.d_model = d_model

        from transformers import SiglipVisionModel, SiglipTextModel
        self.vision_tower = SiglipVisionModel.from_pretrained(model_name)
        self.text_tower = SiglipTextModel.from_pretrained(model_name)

        for p in self.vision_tower.parameters():
            p.requires_grad = False
        for p in self.text_tower.parameters():
            p.requires_grad = False

        self.vision_dim = self.vision_tower.config.hidden_size
        self.text_dim = self.text_tower.config.hidden_size

        self.vision_proj = nn.Linear(self.vision_dim, d_model)
        self.text_proj = nn.Linear(self.text_dim, d_model)

    def forward(
        self,
        vision_emb: torch.Tensor,   # [N, 196, 768]
        static_emb: torch.Tensor,   # [N, 64, 768]
        static_mask: torch.Tensor   # [B, N, 64]
    ) -> tuple:
        """
        Args:
            vision_emb: precomputed [N, 196, 768]
            static_emb: precomputed [N, 64, 768]
            static_mask: [B, N, 64]
        Returns:
            K, V, kv_mask: per-site expanded tensors
        """
        B, N = static_mask.shape[0], static_mask.shape[1]
        B_total = B * N

        h_img_raw = vision_emb.unsqueeze(0).expand(B, -1, -1, -1).reshape(B_total, -1, self.vision_dim)
        h_static_raw = static_emb.unsqueeze(0).expand(B, -1, -1, -1).reshape(B_total, -1, self.text_dim)

        h_img = self.vision_proj(h_img_raw)
        h_static = self.text_proj(h_static_raw)

        K = torch.cat([h_img, h_static], dim=1)
        V = torch.cat([h_img, h_static], dim=1)

        img_mask = torch.ones(B_total, h_img.size(1), device=static_mask.device, dtype=static_mask.dtype)
        kv_mask = torch.cat([img_mask, static_mask.reshape(B_total, -1)], dim=1)

        return K, V, kv_mask


class CrossModalGateFusion(nn.Module):
    """Gated fusion of satellite context into temporal representations."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model

        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.gate_proj = nn.Linear(d_model * 2, d_model)

        # Initialize gate to start near 0 (biased closed)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -4.0)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        kv_mask: torch.Tensor,
        h_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            Q: [B, N, L, d] - From dynamic injector
            K, V: [B, N, M+S_s, d] - From static encoder
            kv_mask: [B, N, M+S_s]
            h_cond: [B, N, L, d] - Fused with dynamic text
        Returns:
            H_fuse: [B, N, L, d] - Fused representation
        """
        B, N, L, d = Q.shape
        B_total = B * N

        Q_flat = Q.reshape(B_total, L, d)
        K_flat = K.reshape(B_total, -1, d)
        V_flat = V.reshape(B_total, -1, d)
        kv_mask_flat = kv_mask.reshape(B_total, -1)

        kv_padding_mask = (kv_mask_flat == 0)
        C, _ = self.cross_attn(
            query=self.norm_q(Q_flat),
            key=self.norm_kv(K_flat),
            value=self.norm_kv(V_flat),
            key_padding_mask=kv_padding_mask
        )

        G = torch.sigmoid(self.gate_proj(torch.cat([h_cond.reshape(B_total, L, d), C], dim=-1)))
        H_fuse = h_cond.reshape(B_total, L, d) + G * C

        return H_fuse.view(B, N, L, d)


class ForecastHead(nn.Module):
    """Flatten all patches → LayerNorm → Dropout → Linear → predict horizon.

    L patches × d dims → LN → flatten → Dropout → Linear(pre_len).
    LayerNorm stabilizes distribution after multi-modal residual injections.
    """

    def __init__(self, d_model: int, num_patches: int, pred_len: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model * num_patches, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, L, d] → [B, N, pred_len]"""
        x = self.norm(x)
        return self.linear(self.dropout(x.reshape(x.shape[0], x.shape[1], -1)))


class EV_VLM(nn.Module):
    """Full EV-VLM multimodal model with PatchTST backbone + three-stage training."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        feat_dim: int = 1,
        d_model: int = 128,
        num_heads: int = 8,
        num_prototypes: int = 64,
        dropout: float = 0.2,
        ablate_prototype: bool = False,
        ablate_dynamic: bool = False,
        siglip_model_name: str = "google/siglip2-base-patch16-224",
        gpt2_model_path: str = None,
        siglip_model_path: str = None,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.num_prototypes = num_prototypes
        self.ablate_prototype = ablate_prototype
        self.ablate_dynamic = ablate_dynamic

        # 0. PatchTST backbone: RevIN + Patching + 3-layer pre-norm BatchNorm Transformer
        self.backbone = PatchTSTBackbone(seq_len, num_nodes, d_model, dropout=dropout)

        # 1. Prototype layer: CALF-style PCA (500 principal directions)
        _principal_emb = None
        if not ablate_prototype:
            try:
                pca_path = os.path.join(os.path.dirname(__file__), 'weights', 'pca_principal_emb.pt')
                if os.path.exists(pca_path):
                    _principal_emb = torch.load(pca_path, map_location='cpu', weights_only=True)
            except Exception:
                pass
        self.prototype_layer = PrototypeSemanticLayer(d_model, num_heads, num_prototypes, _principal_emb)

        # 2. Dynamic text injector (GPT-2 PKVC)
        self.dynamic_injector = DynamicTextInjector(d_model, num_heads, gpt2_model_path) if not ablate_dynamic else None

        # 3. Static vision-text encoder (SigLIP)
        siglip_name = siglip_model_path if siglip_model_path else siglip_model_name
        self.static_vision_encoder = StaticVisionTextEncoder(d_model, siglip_name)

        # 4. Cross-modal gate fusion
        self.cross_gate_fusion = CrossModalGateFusion(d_model, num_heads)

        # 5. Forecast head: LN → flatten → Dropout → Linear
        self.forecast_head = ForecastHead(d_model, self.backbone.num_patches, pred_len, dropout)

    def forward(
        self,
        feat: torch.Tensor,
        prefix_hidden: Optional[torch.Tensor] = None,
        prefix_pkvc: Optional[torch.Tensor] = None,
        stats_ids: Optional[torch.Tensor] = None,
        stats_mask: Optional[torch.Tensor] = None,
        vision_emb: Optional[torch.Tensor] = None,
        static_emb: Optional[torch.Tensor] = None,
        static_mask: Optional[torch.Tensor] = None,
        stage: int = 3,
    ) -> torch.Tensor:
        """
        Args:
            feat: [B, N, T]
            stage: 1=dynamic only, 2=satellite branch, 3=end-to-end
        Returns:
            [B, N, pred_len]
        """
        # 1. PatchTST backbone → [B,N,L,d] + loc/scale for denorm
        h, loc, scale = self.backbone(feat)

        # 2. Prototype layer
        if self.ablate_prototype:
            h_ts = h
        else:
            h_ts = self.prototype_layer(h)

        # 3. Dynamic text injection
        if self.ablate_dynamic or prefix_hidden is None:
            Q, h_cond = h_ts, h_ts
        else:
            Q, h_cond = self.dynamic_injector(h_ts, prefix_hidden, prefix_pkvc, stats_ids, stats_mask)

        # 4. Stage-dependent satellite branch
        if stage == 1 or vision_emb is None:
            H_ref = h_cond
        else:
            K, V, kv_mask = self.static_vision_encoder(vision_emb, static_emb, static_mask)
            H_fuse = self.cross_gate_fusion(Q, K, V, kv_mask, h_cond)
            H_ref = H_fuse

        # 5. Forecast head → prediction in normalized space
        y_norm = self.forecast_head(H_ref)  # [B, N, pred_len]

        # 6. RevIN denorm (using PatchTST's loc/scale)
        # loc/scale: [B, 1, N] → [B, N, pred_len]
        y_hat = y_norm * scale.permute(0, 2, 1) + loc.permute(0, 2, 1)

        if self.pred_len == 1:
            return y_hat.squeeze(-1)
        return y_hat

    def set_train_stage(self, stage: int) -> None:
        """Set training stage and freeze/unfreeze components."""
        for param in self.parameters():
            param.requires_grad = True

        # Freeze GPT-2 inside dynamic_injector
        if self.dynamic_injector is not None:
            for param in self.dynamic_injector.text_encoder.parameters():
                param.requires_grad = False

        # Freeze SigLIP towers
        for param in self.static_vision_encoder.vision_tower.parameters():
            param.requires_grad = False
        for param in self.static_vision_encoder.text_tower.parameters():
            param.requires_grad = False

        if stage == 1:
            self.static_vision_encoder.requires_grad_(False)
            self.cross_gate_fusion.requires_grad_(False)
        elif stage == 2:
            if self.dynamic_injector is not None:
                self.dynamic_injector.requires_grad_(False)
        elif stage == 3:
            # End-to-end stage: keep the small trainable modules unfrozen.
            pass
        else:
            raise ValueError(f"Unsupported EV-VLM training stage: {stage}")
