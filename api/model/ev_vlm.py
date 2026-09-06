"""
EV-VLM: Electric Vehicle Vision-Language Model for Charging Demand Forecasting.

Architecture (LLM-as-backbone, Q-former satellite compression):
    1. feat [B,N,T] -> RevIN (instance norm + per-site affine) -> PatchEmbed -> [B,N,L,768]
    2. unified text prompt (shared | time | stats | static POI) token ids -> gpt2.wte -> [B,N,S_text,768]
    3. satellite vision_emb [N,196,768] -> QFormer (learnable queries, cross-attn) -> [B,N,Q,768]
    4. forecast_token (learnable) -> [B,N,1,768]
    5. inputs_embeds = [text | sat | temporal | forecast] -> GPT-2 (frozen + LoRA on c_attn)
    6. h_forecast = last_hidden_state[:, -1, :] -> ForecastHead -> y_norm -> RevIN denorm -> [B,N,pred_len]

All dimensions are unified at 768 (GPT-2 n_embd). GPT-2 base weights are frozen
(requires_grad=False, NOT wrapped in torch.no_grad so gradients still flow to the
trainable front-ends); only LoRA + PatchEmbed + QFormer + ForecastHead + RevIN +
forecast_token are trainable.
"""

import math
import os
import torch
import torch.nn as nn
from typing import Optional


class RevIN(nn.Module):
    """Reversible instance normalization with per-site learnable affine."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor):
        """x: [B, N, T] -> x_norm: [B, N, T], loc: [B, N, 1], scale: [B, N, 1]"""
        mean = x.mean(dim=-1, keepdim=True)                                   # [B, N, 1]
        std = x.std(dim=-1, keepdim=True, unbiased=False) + self.eps          # [B, N, 1]
        x_norm = (x - mean) / std
        if self.affine:
            a = self.affine_weight.view(1, -1, 1)
            b = self.affine_bias.view(1, -1, 1)
            x_norm = x_norm * a + b
        return x_norm, mean, std


class PatchEmbed(nn.Module):
    """Patchify the (already normalized) series and project each patch to d_model."""

    def __init__(self, patch_len: int = 6, stride: int = 3, d_model: int = 768):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, T] -> [B, N, L, d_model]"""
        # unfold over the last dim -> [B, N, L, patch_len]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return self.proj(patches)


class QFormerBlock(nn.Module):
    """Single BLIP-2 style block: self-attn over queries, then cross-attn to vision."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, vision_emb: torch.Tensor) -> torch.Tensor:
        """q: [N, Q, d], vision_emb: [N, M, d] -> [N, Q, d]"""
        q = self.norm1(q + self.dropout(self.self_attn(q, q, q)[0]))
        q = self.norm2(q + self.dropout(self.cross_attn(q, vision_emb, vision_emb)[0]))
        q = self.norm3(q + self.dropout(self.ffn(q)))
        return q


class QFormer(nn.Module):
    """Compress per-site vision tokens [N, M, d] into Q learnable queries [N, Q, d]."""

    def __init__(self, num_queries: int = 32, d_model: int = 768,
                 num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [QFormerBlock(d_model, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, vision_emb: torch.Tensor) -> torch.Tensor:
        """vision_emb: [N, M, d] -> [N, Q, d] (per-site, no padding mask needed)."""
        N = vision_emb.shape[0]
        q = self.query_tokens.unsqueeze(0).expand(N, -1, -1)  # [N, Q, d]
        for layer in self.layers:
            q = layer(q, vision_emb)
        return q


class LoRAConv1D(nn.Module):
    """Low-rank side path added in parallel to a frozen GPT-2 Conv1D (c_attn).

    GPT-2's Conv1D stores weight as [in, out] and computes `x @ W + b`, so the
    equivalent low-rank factor is `(x @ A) @ B` with A: [in, r], B: [r, out].
    """

    def __init__(self, conv1d: nn.Module, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.conv1d = conv1d
        in_features, out_features = conv1d.weight.shape
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.conv1d(x)                      # x @ W + b (W frozen)
        lora = (x @ self.A) @ self.B * self.scaling
        return base + lora


class ForecastHead(nn.Module):
    """Readout from the <FORECAST> token's final hidden state."""

    def __init__(self, d_model: int = 768, pred_len: int = 6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, pred_len)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B*N, d] -> [B*N, pred_len]"""
        return self.linear(self.norm(h))


class EV_VLM(nn.Module):
    """GPT-2 unified backbone with Q-former satellite compression and LoRA."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        gpt2_model_path: str = None,
        patch_len: int = 6,
        patch_stride: int = 3,
        qformer_queries: int = 32,
        qformer_layers: int = 2,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        llm_chunk_size: int = 64,
        dropout: float = 0.1,
        ablate_qformer: bool = False,
        ablate_dyn: bool = False,
        ablate_text: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.llm_chunk_size = llm_chunk_size
        self.ablate_qformer = ablate_qformer
        self.ablate_dyn = ablate_dyn
        self.ablate_text = ablate_text

        # --- GPT-2 backbone (frozen base + LoRA on c_attn) ---
        from transformers import GPT2Model
        model_name = gpt2_model_path if gpt2_model_path else 'gpt2'
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.d_model = self.gpt2.config.n_embd  # 768

        # Freeze base weights first, then inject LoRA (LoRA params stay trainable).
        for p in self.gpt2.parameters():
            p.requires_grad = False
        for block in self.gpt2.h:
            block.attn.c_attn = LoRAConv1D(block.attn.c_attn, rank=lora_rank, alpha=lora_alpha)

        # Gradient checkpointing bounds activation memory (recompute during backward),
        # which is what lets B*N ~ 2000 site-forecasts per batch fit on one GPU.
        self.gpt2.gradient_checkpointing_enable()

        # --- Trainable front-ends ---
        self.revin = RevIN(num_nodes)
        self.patch_embed = PatchEmbed(patch_len, patch_stride, self.d_model)
        self.qformer = QFormer(qformer_queries, self.d_model, num_layers=qformer_layers, dropout=dropout)
        self.forecast_head = ForecastHead(self.d_model, pred_len)
        self.forecast_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

    def forward(
        self,
        feat: torch.Tensor,               # [B, N, T]
        shared_ids: torch.Tensor,         # [1, S_shared]
        time_ids: torch.Tensor,           # [B, S_time]
        time_mask: torch.Tensor,          # [B, S_time]
        static_ids: torch.Tensor,         # [N, S_static]
        static_mask: torch.Tensor,        # [N, S_static]
        stats_ids: torch.Tensor,          # [B, N, S_stats]
        stats_mask: torch.Tensor,         # [B, N, S_stats]
        vision_emb: torch.Tensor,         # [N, 196, 768]
    ) -> torch.Tensor:
        B, N, T = feat.shape

        # 1. RevIN + PatchEmbed -> temporal [B, N, L, d]
        x_norm, loc, scale = self.revin(feat)
        temporal = self.patch_embed(x_norm)                              # [B, N, L, d]
        L = temporal.shape[2]

        # 2. Unified text prompt embedding (order: shared -> time -> stats -> static)
        shared_b = shared_ids.unsqueeze(1).expand(B, N, -1)              # [B, N, S_shared]
        time_b = time_ids.unsqueeze(1).expand(B, N, -1)                  # [B, N, S_time]
        static_b = static_ids.unsqueeze(0).expand(B, N, -1)              # [B, N, S_static]
        text_ids = torch.cat([shared_b, time_b, stats_ids, static_b], dim=-1)  # [B, N, S_text]
        text = self.gpt2.wte(text_ids)                                   # [B, N, S_text, d]

        # 3. QFormer satellite compression -> [B, N, Q, d]
        sat = self.qformer(vision_emb)                                   # [N, Q, d]
        sat = sat.unsqueeze(0).expand(B, N, -1, -1)                      # [B, N, Q, d]
        Q = sat.shape[2]

        # 4. forecast token -> [B, N, 1, d]
        fc = self.forecast_token.expand(B, N, -1, -1)                    # [B, N, 1, d]

        # 5. Concatenate sequence and flatten batch/site
        inputs_embeds = torch.cat([text, sat, temporal, fc], dim=2)      # [B, N, S, d]
        S = inputs_embeds.shape[2]
        inputs_embeds = inputs_embeds.reshape(B * N, S, self.d_model)

        # 6. Build attention mask (HF semantics: 1 = keep, 0 = mask out)
        def _ones(shape):
            return torch.ones(shape, device=feat.device, dtype=inputs_embeds.dtype)

        S_shared = shared_ids.shape[1]
        S_time = time_ids.shape[1]
        S_stats = stats_ids.shape[1]
        S_static = static_ids.shape[1]

        mask_shared = _ones((B, N, S_shared))
        mask_time = time_mask.float().unsqueeze(1).expand(B, N, -1)
        mask_stats = stats_mask.float()
        mask_static = static_mask.float().unsqueeze(0).expand(B, N, -1)
        mask_sat = _ones((B, N, Q))
        mask_temporal = _ones((B, N, L))
        mask_forecast = _ones((B, N, 1))

        # Ablations: mask positions out (keep sequence length fixed)
        if self.ablate_text:
            mask_shared = torch.zeros_like(mask_shared)
            mask_time = torch.zeros_like(mask_time)
            mask_stats = torch.zeros_like(mask_stats)
            mask_static = torch.zeros_like(mask_static)
        elif self.ablate_dyn:
            mask_time = torch.zeros_like(mask_time)
            mask_stats = torch.zeros_like(mask_stats)
        if self.ablate_qformer:
            mask_sat = torch.zeros_like(mask_sat)

        attention_mask = torch.cat(
            [mask_shared, mask_time, mask_stats, mask_static,
             mask_sat, mask_temporal, mask_forecast], dim=-1
        )                                                                 # [B, N, S]
        attention_mask = attention_mask.reshape(B * N, S)

        # 7. GPT-2 forward, chunked over B*N to bound memory.
        #    Sites are independent inside GPT-2, so chunking is exact and only
        #    splits the batch dimension. No torch.no_grad: gradients flow back
        #    to the front-ends through each chunk.
        h_forecast_list = []
        for start in range(0, B * N, self.llm_chunk_size):
            end = min(start + self.llm_chunk_size, B * N)
            out = self.gpt2(
                inputs_embeds=inputs_embeds[start:end],
                attention_mask=attention_mask[start:end],
                use_cache=False,
            )
            h_forecast_list.append(out.last_hidden_state[:, -1, :])       # [c, d]
        h_forecast = torch.cat(h_forecast_list, dim=0)                    # [B*N, d]

        # 8. <FORECAST> readout token -> forecast head -> RevIN denorm
        y_norm = self.forecast_head(h_forecast).view(B, N, self.pred_len) # [B, N, pred_len]

        if self.revin.affine:
            a = self.revin.affine_weight.view(1, -1, 1)
            b = self.revin.affine_bias.view(1, -1, 1)
            y_norm = (y_norm - b) / a
        y_hat = y_norm * scale + loc                                      # [B, N, pred_len]

        if self.pred_len == 1:
            return y_hat.squeeze(-1)
        return y_hat

    def set_train_stage(self, stage: int) -> None:
        """Single-stage: freezing is done once in __init__; kept for API compatibility."""
        return
