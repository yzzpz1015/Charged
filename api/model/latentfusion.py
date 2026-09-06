# -*- coding: utf-8 -*-
"""
LatentFusion: FreTS backbone with latent satellite modulation.

This module implements a multimodal forecasting model that keeps the original
FreTS temporal backbone and injects satellite information through a latent
space modulation branch.

Structure:
    1. FreTS temporal backbone extracts latent temporal features
    2. Satellite encoder projects auxiliary features into a latent space
    3. Modulation head generates FiLM-style gamma/beta parameters
    4. Temporal latent is modulated before the FreTS prediction head

Compared with the previous standalone latent-space fusion design, this version
preserves the frequency-domain inductive bias of FreTS and only uses satellite
features as a conditioning signal.
"""

import torch
import torch.nn as nn

from api.model.modules import FreTS


class SatelliteLatentModulator(nn.Module):
    """Encode satellite embeddings into modulation parameters."""

    def __init__(self, sat_dim: int, latent_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super(SatelliteLatentModulator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(sat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.gamma_head = nn.Linear(latent_dim, latent_dim)
        self.beta_head = nn.Linear(latent_dim, latent_dim)
        self.gate_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, sat_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return gamma, beta and gate tensors with shape [B, N, latent_dim]."""
        latent = self.encoder(sat_feat)
        gamma = self.gamma_head(latent)
        beta = self.beta_head(latent)
        gate = self.gate_head(latent)
        return gamma, beta, gate


class LatentFusion(FreTS):
    """FreTS backbone with latent satellite modulation."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        sat_dim: int,
        embed_size: int = 128,
        hidden_size: int = 256,
        sparsity_threshold: float = 0.01,
        scale: float = 0.02,
        sat_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        if sat_dim <= 0:
            raise ValueError("sat_dim must be > 0")

        # Use FreTS as the temporal backbone.
        super(LatentFusion, self).__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            n_fea=1,
            embed_size=embed_size,
            hidden_size=hidden_size,
            sparsity_threshold=sparsity_threshold,
            scale=scale,
        )

        self.sat_dim = sat_dim
        self.chunk_size = 512

        # Satellite branch produces latent modulation parameters.
        self.sat_modulator = SatelliteLatentModulator(
            sat_dim=sat_dim,
            latent_dim=self.embed_size,
            hidden_dim=sat_hidden_dim,
            dropout=dropout,
        )

        # Residual strength for satellite conditioning.
        self.modulation_strength = nn.Parameter(torch.tensor(1.0))

    def _extract_sat_embedding(self, extra_feat: torch.Tensor) -> torch.Tensor:
        """Support [B, N, C] and [B, N, S, C] auxiliary tensors."""
        if extra_feat.dim() == 4:
            sat = extra_feat[:, :, -1, :]
        elif extra_feat.dim() == 3:
            sat = extra_feat
        else:
            raise ValueError(
                "extra_feat for LatentFusion must be [B, N, C] or [B, N, S, C]"
            )

        if sat.shape[-1] < self.sat_dim:
            raise ValueError(
                f"Expected satellite embedding dim >= {self.sat_dim}, got {sat.shape[-1]}"
            )

        return sat[..., -self.sat_dim:]

    def _encode_temporal_latent(self, x_chunk: torch.Tensor) -> torch.Tensor:
        """Run the FreTS temporal backbone up to the latent tensor before the head."""
        batch_size, time_len, channel_size = x_chunk.shape
        x_chunk = self.tokenEmb(x_chunk)
        bias = x_chunk
        x_chunk = self.MLP_channel(x_chunk, batch_size, channel_size, time_len)
        x_chunk = self.MLP_temporal(x_chunk, batch_size, channel_size, time_len)
        x_chunk = x_chunk + bias
        return x_chunk

    def forward(self, feat: torch.Tensor, extra_feat: torch.Tensor = None) -> torch.Tensor:
        if extra_feat is None:
            raise ValueError("LatentFusion requires satellite embeddings via extra_feat")

        x = feat.unsqueeze(-1)

        batch_ori, num_nodes, seq_len, channel = x.shape
        x_enc = x.view(batch_ori * num_nodes, seq_len, channel)

        sat_embed = self._extract_sat_embedding(extra_feat)
        sat_embed_flat = sat_embed.view(batch_ori * num_nodes, -1)

        outputs = []
        current_chunk_size = self.chunk_size
        start_idx = 0

        while start_idx < x_enc.shape[0]:
            try:
                x_chunk = x_enc[start_idx : start_idx + current_chunk_size]
                sat_chunk = sat_embed_flat[start_idx : start_idx + current_chunk_size]

                batch_size, time_len, channel_size = x_chunk.shape
                x_latent = self._encode_temporal_latent(x_chunk)

                gamma, beta, gate = self.sat_modulator(sat_chunk)
                gamma = gamma.view(batch_size, 1, self.embed_size).unsqueeze(2)
                beta = beta.view(batch_size, 1, self.embed_size).unsqueeze(2)
                gate = gate.view(batch_size, 1, self.embed_size).unsqueeze(2)

                # Modulate the FreTS latent tensor before the prediction head.
                x_latent = x_latent * (1.0 + self.modulation_strength * gate * gamma) + gate * beta

                x_out = self.fc(x_latent.reshape(batch_size, channel_size, -1)).permute(0, 2, 1)
                x_out = self.final_linear(x_out)
                x_out = x_out.squeeze(-1)
                outputs.append(x_out)
                start_idx += current_chunk_size
            except RuntimeError as e:
                if "CUDA" in str(e):
                    current_chunk_size = max(1, current_chunk_size // 2)
                    torch.cuda.empty_cache()
                else:
                    raise e

        out = torch.cat(outputs, dim=0)
        out = out.view(batch_ori, num_nodes)
        return out
