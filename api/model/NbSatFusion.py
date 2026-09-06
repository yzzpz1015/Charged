import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFrequencyEncoder(nn.Module):
    def __init__(self, seq_len, input_channels, latent_dim, embed_size=128, hidden_size=256, sparsity_threshold=0.01, scale=0.02, dropout=0.1):
        super().__init__()
        self.seq_length, self.input_channels, self.latent_dim = seq_len, max(1, input_channels), latent_dim
        self.embeddings = nn.Parameter(torch.randn(1, embed_size))
        self.r1 = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.i1 = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.rb1, self.ib1 = nn.Parameter(scale * torch.randn(embed_size)), nn.Parameter(scale * torch.randn(embed_size))
        self.r2 = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.i2 = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.rb2, self.ib2 = nn.Parameter(scale * torch.randn(embed_size)), nn.Parameter(scale * torch.randn(embed_size))
        self.sparsity_threshold = sparsity_threshold
        self.proj = nn.Sequential(
            nn.Linear(seq_len * embed_size * self.input_channels, hidden_size),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, latent_dim), nn.LayerNorm(latent_dim)
        )

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(-1)
        B, N, S, C = x.shape
        x = x.view(B*N, S, C).permute(0, 2, 1).unsqueeze(3) * self.embeddings
        bias = x
        x = x.permute(0, 2, 1, 3)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        re = F.relu(torch.einsum('bijd,dd->bijd', x.real, self.r1) - torch.einsum('bijd,dd->bijd', x.imag, self.i1) + self.rb1)
        im = F.relu(torch.einsum('bijd,dd->bijd', x.imag, self.r1) + torch.einsum('bijd,dd->bijd', x.real, self.i1) + self.ib1)
        x = torch.fft.irfft(torch.view_as_complex(F.softshrink(torch.stack([re, im], -1), self.sparsity_threshold)), n=C, dim=2, norm='ortho').permute(0, 2, 1, 3)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        re = F.relu(torch.einsum('bijd,dd->bijd', x.real, self.r2) - torch.einsum('bijd,dd->bijd', x.imag, self.i2) + self.rb2)
        im = F.relu(torch.einsum('bijd,dd->bijd', x.imag, self.r2) + torch.einsum('bijd,dd->bijd', x.real, self.i2) + self.ib2)
        x = torch.fft.irfft(torch.view_as_complex(F.softshrink(torch.stack([re, im], -1), self.sparsity_threshold)), n=S, dim=2, norm='ortho')
        x = (x + bias).reshape(B*N, -1)
        return self.proj(x).view(B, N, -1)


class SpatialEncoder(nn.Module):
    def __init__(self, image_size=384, patch_size=16, embed_dim=128, depth=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(nn.Linear(embed_dim, embed_dim*4), nn.GELU(), nn.Linear(embed_dim*4, embed_dim))
            }) for _ in range(depth)
        ])
        self.dropout = nn.Dropout(dropout)

    def _coords_to_idx(self, coords, grid_size):
        x = (coords[..., 0] * grid_size).long().clamp(0, grid_size - 1)
        y = (coords[..., 1] * grid_size).long().clamp(0, grid_size - 1)
        return y * grid_size + x

    def forward(self, image, target_coords=None, neighbor_coords=None):
        if image.dim() == 5:
            B, N = image.shape[:2]
            image = image.flatten(0, 1)
            if target_coords is not None: target_coords = target_coords.flatten(0, 1)
            if neighbor_coords is not None: neighbor_coords = neighbor_coords.flatten(0, 1)
        else:
            B, N = image.shape[0], 1

        tokens = self.patch_embed(image).flatten(2).transpose(1, 2)
        grid_size = int(tokens.shape[1] ** 0.5)
        tokens = self.dropout(tokens + self.pos_embed[:, :tokens.shape[1]])

        for blk in self.blocks:
            tokens = tokens + blk['attn'](blk['norm1'](tokens), blk['norm1'](tokens), blk['norm1'](tokens))[0]
            tokens = tokens + blk['mlp'](blk['norm2'](tokens))

        t_idx = self._coords_to_idx(target_coords, grid_size) if target_coords is not None else torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
        batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
        z_t = tokens[batch_idx, t_idx.view(-1)]

        z_k = None
        if neighbor_coords is not None:
            n_idx = self._coords_to_idx(neighbor_coords, grid_size)
            n_idx = n_idx.clamp(0, tokens.shape[1] - 1)
            z_k = tokens.gather(1, n_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))

        z_t = z_t.view(B, N, -1)
        if z_k is not None:
            z_k = z_k.view(B, N, -1, tokens.shape[-1])

        return z_t, z_k


class SpatialRelationModulator(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, hidden_dim=128):
        super().__init__()
        self.relation_mlp = nn.Sequential(
            nn.Linear(spatial_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, temporal_dim)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim), nn.Sigmoid()
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(temporal_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, z_t, z_k, h_k, h_t):
        B, N, K, D_s = z_k.shape
        D_t = h_k.shape[-1]

        r_k = self.relation_mlp(torch.cat([z_t.unsqueeze(2).expand_as(z_k), z_k], dim=-1))
        gate = self.gate_mlp(r_k)
        h_modulated = h_k * gate

        h_t_exp = h_t.unsqueeze(2).expand(-1, -1, K, -1)
        score = self.score_mlp(torch.cat([h_modulated, h_t_exp], dim=-1))
        alpha = torch.softmax(score, dim=2)

        return (alpha * h_modulated).sum(dim=2)


class NbSatFusion(nn.Module):
    def __init__(self, seq_len, pred_len, n_fea, temporal_latent_dim=128, temporal_hidden_dim=256,
                 image_size=384, patch_size=16, image_embed_dim=128, vit_depth=4, vit_heads=4,
                 fusion_dim=256, dropout=0.1):
        super().__init__()
        self.target_encoder = TemporalFrequencyEncoder(seq_len, 1, temporal_latent_dim, hidden_size=temporal_hidden_dim, dropout=dropout)
        self.neighbor_encoder = TemporalFrequencyEncoder(seq_len, 1, temporal_latent_dim, hidden_size=temporal_hidden_dim, dropout=dropout)
        self.spatial_encoder = SpatialEncoder(image_size, patch_size, image_embed_dim, vit_depth, vit_heads, dropout)
        self.relation_modulator = SpatialRelationModulator(image_embed_dim, temporal_latent_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(temporal_latent_dim * 2 + image_embed_dim, fusion_dim),
            nn.GELU(), nn.Linear(fusion_dim, pred_len)
        )

    def forward(self, feat, neighbor_feat=None, neighbor_mask=None, neighbor_coords=None, target_coords=None, image=None, **kwargs):
        if image is None:
            raise ValueError("NbSatFusion requires satellite image input")

        B, N, S = feat.shape
        h_t = self.target_encoder(feat)

        z_t, z_k = self.spatial_encoder(image, target_coords, neighbor_coords)

        if neighbor_feat is not None and neighbor_feat.numel() > 0 and z_k is not None:
            K = neighbor_feat.shape[2]
            h_k = self.neighbor_encoder(
                neighbor_feat.reshape(B * N * K, S, 1).unsqueeze(1)
            ).view(B, N, K, -1)
            modulated_neighbor = self.relation_modulator(z_t, z_k, h_k, h_t)
        else:
            modulated_neighbor = torch.zeros_like(h_t)

        fused = torch.cat([h_t, modulated_neighbor, z_t], dim=-1)
        return self.fusion_mlp(fused).squeeze(-1)
