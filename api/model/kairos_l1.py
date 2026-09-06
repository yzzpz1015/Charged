# -*- coding: utf-8 -*-
"""Kairos L1 experiments: frozen Kairos backbone + trainable prediction head.

Three variants, switched by flags (the first three rungs of the L0/L1/L2 ladder):

  - **L1-plain** : frozen Kairos backbone + point prediction head (no LoRA,
    no semantics). Isolates "how much a tuned head alone beats zero-shot".
  - **L1-lora**  : L1-plain + LoRA (rank r) on every T5Attention q/k/v/o.
    Orthogonal variable: does a small adapter on the frozen trunk help?
  - **L1b**      : L1-plain + LLM time-series semantics. A frozen distilgpt2
    encodes a per-sample statistical description of the history, mean-pooled,
    projected to d_model, and gated (scalar gate, init 0) into the decoder
    output before the head. Isolates "does LLM semantics add anything".

Kairos is a univariate per-quantile forecaster. We keep its encoder + decoder
frozen and swap its quantile head for a single point head, so the three L1
variants are directly comparable to the zero-shot median baseline.

Forward returns ``(y_norm, loc_scale)`` where ``y_norm`` lives in the instance-
normalized space (same space Kairos trains its quantile loss in). The caller
denormalizes with ``y_norm * scale + loc`` for raw-space metrics.
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank side branch wrapped around a frozen nn.Linear.

        out = W @ x + b + (x @ A^T @ B^T) * (alpha / r)

    A is kaiming-init, B is zero-init so the branch is identity at t=0.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        out_features, in_features = linear.weight.shape
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # freeze the wrapped weights (safe regardless of injection order)
        for p in linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base + delta * self.scaling


def inject_lora(backbone: nn.Module, rank: int = 8, alpha: int = 16) -> List[LoRALinear]:
    """Wrap every T5Attention q/k/v/o Linear in the backbone with LoRALinear."""
    injected: List[LoRALinear] = []
    for module in backbone.modules():
        if module.__class__.__name__ == 'T5Attention':
            for proj_name in ('q', 'k', 'v', 'o'):
                linear = getattr(module, proj_name)
                if isinstance(linear, LoRALinear):
                    continue
                lora_linear = LoRALinear(linear, rank=rank, alpha=alpha)
                setattr(module, proj_name, lora_linear)
                injected.append(lora_linear)
    return injected


class KairosL1(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pre_len: int,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        use_llm_semantics: bool = False,
        text_model_path: Optional[str] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.pre_len = pre_len
        self.d_model = backbone.config.d_model
        self.use_lora = use_lora
        self.use_llm_semantics = use_llm_semantics

        # 1. freeze the whole trunk first (so later LoRA params stay trainable)
        for p in backbone.parameters():
            p.requires_grad = False

        # 2. optional LoRA on attention projections
        self.lora_modules = inject_lora(backbone, lora_rank, lora_alpha) if use_lora else []

        # 3. point prediction head (replaces Kairos's quantile head)
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, pre_len),
        )

        # 4. optional LLM semantics injection (L1b)
        self.text_tokenizer = None
        self.text_encoder = None
        self.text_proj = None
        self.gate = None
        if use_llm_semantics:
            from transformers import GPT2Model, GPT2Tokenizer
            model_name = text_model_path or 'distilgpt2'
            self.text_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            if self.text_tokenizer.pad_token is None:
                self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
            self.text_encoder = GPT2Model.from_pretrained(model_name)
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            text_dim = self.text_encoder.config.n_embd
            self.text_proj = nn.Linear(text_dim, self.d_model)
            self.gate = nn.Parameter(torch.zeros(1))  # scalar gate, init 0 => L1b == L1-plain at t=0

    def forward(
        self,
        x: torch.Tensor,
        text_ids: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """x: [B, T] raw volume -> (y_norm [B, pre_len], loc_scale (loc, scale) [B, 1])."""
        hidden, loc_scale, input_embeds, attn_mask, _, s_features = self.backbone.encode(x)
        seq_out, _ = self.backbone.decode(input_embeds, attn_mask, hidden, s_features=s_features)
        h = seq_out[:, 0, :]  # [B, d_model] first decoder segment (covers the first 64 steps)

        if self.use_llm_semantics:
            text_hidden = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask
            ).last_hidden_state  # [B, S, text_dim]
            mask = text_mask.unsqueeze(-1).float()  # [B, S, 1]
            pooled = (text_hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # [B, text_dim]
            s_text = self.text_proj(pooled)  # [B, d_model]
            h = h + self.gate * torch.tanh(s_text)

        y_norm = self.head(h)  # [B, pre_len]
        return y_norm, loc_scale
