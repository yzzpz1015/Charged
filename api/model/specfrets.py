# -*- coding: utf-8 -*-
"""
SpecFreTS: Spectral Frequency-domain Fusion for Multimodal Time Series Forecasting

This module implements SpecFreTS, which performs spectral-aligned fusion of temporal 
and satellite data in the frequency domain. Unlike SatFreTS (which fuses after frequency 
processing), SpecFreTS integrates satellite information directly within the frequency domain.

Key Features:
    1. Temporal to frequency: Convert time series to complex-valued spectrum via FFT
    2. Satellite projection: Project satellite embeddings to frequency coefficients
    3. Spectral fusion: Blend temporal and satellite spectra in frequency space
    4. Frequency MLP: Apply learnable mixing in frequency domain
    5. Inverse transform: Convert fused spectrum back to time domain
    
This approach enables deeper multimodal interaction at the spectral level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecFreTS(nn.Module):
    """
    Spectral Frequency-domain Fusion (SpecFreTS) for multimodal time series forecasting.
    
    This model performs spectral-aligned fusion of temporal and satellite data in the 
    frequency domain. The key difference from SatFreTS:
    - SatFreTS: Temporal (time→freq→time) + Satellite → Gate Fusion (time domain)
    - SpecFreTS: Temporal (time→freq) + Satellite (projected→freq) → Fuse (freq domain) → Gate Fusion (freq)
    
    The fusion happens in the complex-valued frequency space, allowing:
    - Direct spectral component interaction (magnitude + phase)
    - Satellite-modulated frequency responses
    - Learnable frequency-domain weights for each modality
    
    Args:
        seq_len (int): Input sequence length.
        pred_len (int): Prediction horizon length.
        sat_dim (int): Satellite embedding dimension.
        embed_size (int): Frequency-domain embedding dimension (default: 128).
        hidden_size (int): MLP hidden dimension (default: 256).
        sparsity_threshold (float): Soft shrinkage threshold (default: 0.01).
        scale (float): Weight initialization scale (default: 0.02).
        fusion_type (str): Fusion strategy - 'add', 'mul', 'gate' (default: 'gate').
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        sat_dim: int,
        embed_size: int = 128,
        hidden_size: int = 256,
        sparsity_threshold: float = 0.01,
        scale: float = 0.02,
        fusion_type: str = 'gate',
    ):
        super(SpecFreTS, self).__init__()
        
        if sat_dim <= 0:
            raise ValueError("sat_dim must be > 0 for SpecFreTS")
        
        self.seq_length = seq_len
        self.pre_length = pred_len
        self.sat_dim = sat_dim
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.scale = scale
        self.fusion_type = fusion_type
        
        # Chunk size for memory-efficient processing
        self.chunk_size = 512
        
        # ============ Temporal Path Parameters ============
        # Learnable token embeddings for time domain → frequency domain transition
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        
        # Channel dimension frequency processing (FFT on channels)
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        
        # Temporal dimension frequency processing (FFT on time)
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        
        # ============ Satellite Projection to Frequency Domain ============
        # Projects satellite embeddings to complex frequency coefficients
        # These create modulation signals in frequency space
        self.sat_freq_real_proj = nn.Sequential(
            nn.Linear(sat_dim, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )
        self.sat_freq_imag_proj = nn.Sequential(
            nn.Linear(sat_dim, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )
        
        # ============ Spectral Fusion Gate ============
        # Learns how to blend temporal and satellite spectra
        # Input: concatenated magnitude of temporal and satellite spectra
        self.spec_fusion_gate = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_size, embed_size),
            nn.Sigmoid(),
        )
        
        # ============ Output Layers ============
        # Flatten and collapse back to time domain predictions
        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.pre_length)
        )

    def tokenEmb(self, x):
        """
        Apply learnable token embeddings.
        
        Args:
            x: [B, T, C] - batch, time, channel
            
        Returns:
            [B, C, T, E] - embedded with learnable embeddings
        """
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = x.unsqueeze(3)  # [B, C, T, 1]
        y = self.embeddings  # [1, E]
        return x * y  # [B, C, T, E]

    def _extract_sat_embedding(self, extra_feat: torch.Tensor) -> torch.Tensor:
        """
        Extract satellite embedding from auxiliary features.
        
        Args:
            extra_feat: [B, N, C] or [B, N, S, C]
            
        Returns:
            [B, N, sat_dim] satellite embeddings
        """
        if extra_feat.dim() == 4:
            sat = extra_feat[:, :, -1, :]  # Use latest timestep
        elif extra_feat.dim() == 3:
            sat = extra_feat
        else:
            raise ValueError("extra_feat must be [B, N, C] or [B, N, S, C]")
        
        if sat.shape[-1] < self.sat_dim:
            raise ValueError(
                f"Expected satellite embedding dim >= {self.sat_dim}, got {sat.shape[-1]}"
            )
        
        return sat[..., -self.sat_dim:]  # Use trailing channels

    def _project_sat_to_spectrum(self, sat_embed: torch.Tensor) -> tuple:
        """
        Project satellite embeddings to frequency-domain representation.
        
        Args:
            sat_embed: [B*N, sat_dim] - flattened satellite embeddings
            
        Returns:
            sat_real: [B*N, E] - real part of satellite spectrum
            sat_imag: [B*N, E] - imaginary part of satellite spectrum
        """
        sat_real = self.sat_freq_real_proj(sat_embed)  # [B*N, E]
        sat_imag = self.sat_freq_imag_proj(sat_embed)  # [B*N, E]
        return sat_real, sat_imag

    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        """
        Frequency-domain MLP with complex operations.
        
        Applies learnable transormation in frequency space:
        z = relu(real_part) + i*relu(imag_part)
        where real_part = x.real @ r - x.imag @ i + rb
              imag_part = x.imag @ r + x.real @ i + ib
        
        Args:
            B: Batch size
            nd: Number of dimensions
            dimension: Feature dimension (e.g., seq_len or n_features)
            x: Complex tensor [B, nd, dimension//2+1, E]
            r, i, rb, ib: Real and imaginary weights/biases
            
        Returns:
            Complex tensor [B, nd, dimension//2+1, E]
        """
        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - 
            torch.einsum('bijd,dd->bijd', x.imag, i) + 
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + 
            torch.einsum('bijd,dd->bijd', x.real, i) + 
            ib
        )

        # Apply soft shrinkage for sparsity
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def MLP_temporal(self, x, B, N, L):
        """Apply frequency-domain MLP along temporal dimension."""
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    def MLP_channel(self, x, B, N, L):
        """Apply frequency-domain MLP along channel dimension."""
        x = x.permute(0, 2, 1, 3)  # [B, T, N, E]
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=N, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)  # [B, N, T, E]
        return x

    def _fuse_in_spectrum(self, temporal_spec: torch.Tensor, sat_real: torch.Tensor, 
                          sat_imag: torch.Tensor) -> torch.Tensor:
        """
        Fuse temporal and satellite spectra in frequency domain.
        
        Args:
            temporal_spec: [B_chunk, 1, freq_bins, E] - temporal complex spectrum
            sat_real: [B_chunk, E] - satellite real coefficients
            sat_imag: [B_chunk, E] - satellite imaginary coefficients
            
        Returns:
            fused_spec: [B_chunk, 1, freq_bins, E] - fused complex spectrum
        """
        freq_bins = temporal_spec.shape[2]
        
        # Construct satellite spectrum: [B_chunk, 1, 1, E]
        sat_real_exp = sat_real.unsqueeze(1).unsqueeze(1)  # [B_chunk, 1, 1, E]
        sat_imag_exp = sat_imag.unsqueeze(1).unsqueeze(1)
        sat_spec_single = torch.complex(sat_real_exp, sat_imag_exp)
        
        # Expand satellite spectrum to all frequency bins: [B_chunk, 1, freq_bins, E]
        sat_spec = sat_spec_single.expand(-1, -1, freq_bins, -1)
        
        # Compute magnitudes for gating decision
        temp_mag = torch.abs(temporal_spec)  # [B_chunk, 1, freq_bins, E]
        sat_mag = torch.abs(sat_spec)
        
        # Apply spectral fusion gate at each frequency bin
        if self.fusion_type == 'gate':
            # Reshape for gate computation: [B_chunk * freq_bins, 2*E]
            temp_mag_flat = temp_mag.reshape(-1, self.embed_size)
            sat_mag_flat = sat_mag.reshape(-1, self.embed_size)
            gate_input = torch.cat([temp_mag_flat, sat_mag_flat], dim=-1)
            
            gate = self.spec_fusion_gate(gate_input)  # [B_chunk*freq_bins, E]
            gate = gate.reshape(temp_mag.shape)  # [B_chunk, 1, freq_bins, E]
            gate_complex = torch.complex(gate, torch.zeros_like(gate))
            
            # Gated blend: α*sat + (1-α)*temporal
            fused = gate_complex * sat_spec + (1.0 - gate_complex) * temporal_spec
        elif self.fusion_type == 'add':
            fused = temporal_spec + sat_spec
        elif self.fusion_type == 'mul':
            fused = temporal_spec * sat_spec
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        
        return fused

    def forward(self, feat, extra_feat=None):
        """
        Forward pass with spectral fusion.
        
        Args:
            feat: [B, N, T] - temporal features
            extra_feat: [B, N, C] or [B, N, S, C] - satellite embeddings
            
        Returns:
            [B, N] - predictions
        """
        if extra_feat is None:
            raise ValueError("SpecFreTS requires satellite embeddings via extra_feat")
        
        # Prepare inputs
        x = feat.unsqueeze(-1)  # [B, N, T, 1]
        B_ori, num_nodes, seq_len, _ = x.shape
        x_enc = x.view(B_ori * num_nodes, seq_len, 1)
        
        # Extract satellite embeddings
        sat_embed = self._extract_sat_embedding(extra_feat)  # [B, N, sat_dim]
        sat_embed_flat = sat_embed.view(B_ori * num_nodes, -1)  # [B*N, sat_dim]
        
        # Project satellite to frequency domain
        sat_real_flat, sat_imag_flat = self._project_sat_to_spectrum(sat_embed_flat)
        
        outputs = []
        current_chunk_size = self.chunk_size
        start_idx = 0
        
        while start_idx < x_enc.shape[0]:
            try:
                x_chunk = x_enc[start_idx: start_idx + current_chunk_size]
                B_chunk = x_chunk.shape[0]
                
                # Get satellite for this chunk
                sat_end_idx = min(start_idx + B_chunk, B_ori * num_nodes)
                sat_real_chunk = sat_real_flat[start_idx:sat_end_idx]
                sat_imag_chunk = sat_imag_flat[start_idx:sat_end_idx]
                
                # ========== Temporal Path ==========
                # Token embedding
                x_chunk = self.tokenEmb(x_chunk)  # [B_chunk, 1, T, E]
                bias = x_chunk.clone()
                
                # Transform to frequency domain
                x_chunk = x_chunk.permute(0, 2, 1, 3)  # [B_chunk, T, 1, E]
                temporal_freq = torch.fft.rfft(x_chunk, dim=1, norm='ortho')  # [B_chunk, freq, 1, E]
                temporal_freq = temporal_freq.permute(0, 2, 1, 3)  # [B_chunk, 1, freq, E]
                
                # ========== Spectral Fusion ==========
                fused_freq = self._fuse_in_spectrum(temporal_freq, sat_real_chunk, sat_imag_chunk)
                
                # ========== Inverse Transform + MLPs ==========
                fused_freq = fused_freq.permute(0, 2, 1, 3)  # [B_chunk, freq, 1, E]
                fused_time = torch.fft.irfft(fused_freq, n=self.seq_length, dim=1, norm='ortho')
                fused_time = fused_time.permute(0, 2, 1, 3)  # [B_chunk, 1, T, E]
                
                # Apply MLPs in time domain after fusion
                fused_time = fused_time + bias  # Residual
                
                # Flatten and project to prediction: [B_chunk, 1, seq*E] -> [B_chunk, 1, pred_len]
                x_out = self.fc(fused_time.reshape(B_chunk, 1, -1))
                
                # For single-step prediction: [B_chunk, 1, pred_len] -> [B_chunk]
                x_out = x_out.squeeze(1)  # Remove middle dimension: [B_chunk, pred_len]
                if x_out.shape[-1] > 1:
                    x_out = x_out[:, -1]  # Take last prediction if multi-step
                
                outputs.append(x_out)
                start_idx += current_chunk_size
                
            except RuntimeError as e:
                if "CUDA" in str(e):
                    current_chunk_size = max(1, current_chunk_size // 2)
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        out = torch.cat(outputs, dim=0)
        out = out.view(B_ori, num_nodes)
        return out
