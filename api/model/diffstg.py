# -*- coding: utf-8 -*-
"""#17 DiffSTG：时空图扩散（图卷积 + 时间卷积的条件 DDPM）。

与原论文（Wen et al., 2023）的关系与差异：

* **保留**：整块未来 [N,72] 联合扩散；去噪网络在每个 block 里做「时间卷积 +
  站间图卷积」；以历史 72h 为条件；DDPM 线性 beta、ε-参数化。
* **差异（表注必须写）**：原版用 UGnet（U-Net + 图卷积），这里是轻量
  「时间 conv + 稀疏 kNN 图聚合」堆叠；采样用 50 步 DDIM（η=1，补齐跳步
  随机性）而非 1000 步 ancestral；邻接与 #8/#10 相同（distance.csv kNN k=8）。

实现要点：
* 图聚合用 **sparse** 邻接（SZH 843 站，稠密 N² 在采样 batch 下不可承受）；
* 图算子只沿站轴聚合，时间轴用膨胀卷积；两者都带残差与时间 FiLM；
* 归一化：CausalRevIN（只用历史 72h），与其它 baseline 一致。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from api.model.quant_head import CausalRevIN
from api.model.timegrad import _sinu


class _STBlock(nn.Module):
    def __init__(self, ch: int, temb: int, dilation: int, dropout: float):
        super().__init__()
        self.dil = dilation
        self.norm = nn.GroupNorm(8, ch)
        self.tconv = nn.Conv2d(ch, ch, (1, 3), padding=(0, dilation),
                               dilation=(1, dilation))
        self.gconv = nn.Conv2d(ch, ch, 1)
        self.emb = nn.Linear(temb, 2 * ch)
        # 历史条件逐 block FiLM：[B,ch,N,1] -> [B,2ch,N,1]。
        # 只在输入层加一次条件会被后续 residual/conv 冲掉，实测 ε 学不动。
        self.cemb = nn.Conv2d(ch, 2 * ch, 1)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.emb.weight)
        nn.init.zeros_(self.emb.bias)
        nn.init.zeros_(self.cemb.weight)
        nn.init.zeros_(self.cemb.bias)

    def forward(self, x, adj_sp, te, cf=None):
        B, C, N, L = x.shape
        h = self.tconv(F.gelu(self.norm(x)))
        # 站间图聚合：[N, B*C*L] 左乘稀疏 A
        g = h.permute(2, 0, 1, 3).reshape(N, -1)
        g = torch.sparse.mm(adj_sp, g).view(N, B, C, L).permute(1, 2, 0, 3)
        h = h + self.gconv(g)
        s, b = self.emb(te).chunk(2, -1)                # [B,ch]
        s = s[:, :, None, None]
        b = b[:, :, None, None]
        if cf is not None:
            sc, bc = self.cemb(cf).chunk(2, 1)           # [B,ch,N,1]
            s = s + sc
            b = b + bc
        h = F.gelu(h * (1.0 + s) + b)
        return x + self.drop(h)


class DiffSTGModel(nn.Module):
    def __init__(self, num_nodes: int, seq_len: int = 72, pre_len: int = 72,
                 d_model: int = 64, n_layers: int = 4, temb: int = 128,
                 dropout: float = 0.1, diffusion_steps: int = 1000,
                 beta0: float = 1e-4, beta1: float = 0.02,
                 adj_matrix: torch.Tensor = None, **_):
        super().__init__()
        if adj_matrix is None:
            raise ValueError("DiffSTG 需要邻接矩阵")
        self.pre_len = pre_len
        self.diffusion_steps = diffusion_steps
        self.revin = CausalRevIN()
        self.temb_dim = temb

        # 行/列对称归一化后的稀疏邻接
        a = adj_matrix.float()
        deg = a.sum(1).clamp_min(1e-6)
        a = a / deg[:, None]
        a = 0.5 * (a + a.t())
        a_sp = a.to_sparse().coalesce()
        # 稀疏张量做 buffer 在 torch.save/load 时不可靠（实测 CUDA illegal access），
        # 改存 indices/values，前向时重建。
        self.register_buffer("adj_idx", a_sp.indices())
        self.register_buffer("adj_val", a_sp.values())
        self._adj_shape = tuple(a_sp.shape)

        self.temb_mlp = nn.Sequential(nn.Linear(temb, temb), nn.GELU())
        # 历史条件用 GRU 读整段并取末状态。早期版本用 cond_enc(...).mean(dim=3)
        # 把历史压成站级均值；RevIN 后该均值≈0，模型拿不到「当前处在周期的哪个
        # 位置」，会退化成预测窗口均值（JHB 实测 E[q]/target=0.67），故改为 GRU。
        self.hist_enc = nn.GRU(1, d_model, batch_first=True)
        self.inp = nn.Conv2d(1, d_model, 3, padding=1)
        self.blocks = nn.ModuleList([
            _STBlock(d_model, temb, 2 ** (i % 4), dropout)
            for i in range(n_layers)])
        self.out_norm = nn.GroupNorm(8, d_model)
        self.out = nn.Conv2d(d_model, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        betas = torch.linspace(beta0, beta1, diffusion_steps)
        ac = torch.cumprod(1.0 - betas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", ac)

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """x_t [B,1,N,L]，cond [B,1,N,T] -> ε̂ [B,1,N,L]。"""
        te = self.temb_mlp(_sinu(t, self.temb_dim))
        adj_sp = torch.sparse_coo_tensor(self.adj_idx, self.adj_val,
                                         self._adj_shape).coalesce()
        B, _, N, T = cond.shape
        hs = cond.permute(0, 2, 3, 1).reshape(B * N, T, 1)
        _, hn = self.hist_enc(hs)
        c = hn[-1].view(B, N, -1).permute(0, 2, 1).unsqueeze(-1)  # [B,d,N,1]
        h = self.inp(x_t) + c
        for blk in self.blocks:
            h = blk(h, adj_sp, te, c)
        return self.out(F.gelu(self.out_norm(h)))

    def p_loss(self, x0: torch.Tensor, cond: torch.Tensor,
               weight: torch.Tensor = None, generator=None) -> torch.Tensor:
        """x0 [B,1,N,L] 归一化未来；weight [B] 0/1。"""
        B = x0.shape[0]
        t = torch.randint(0, self.diffusion_steps, (B,), device=x0.device,
                          generator=generator)
        eps = torch.randn(x0.shape, device=x0.device, generator=generator)
        a = self.alphas_cumprod[t][:, None, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        err = (self.denoise(xt, t, cond) - eps) ** 2
        if weight is None:
            return err.mean()
        w = weight.to(err.dtype)[:, None, :, None]      # [B,1,N,1]
        return (err * w).sum() / (w * err[0].numel() / err.shape[2]).sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_samples: int = 1,
               ddim_steps: int = 50, eta: float = 1.0,
               sample_chunk: int = 20, generator=None) -> torch.Tensor:
        """cond [B,1,N,T] -> [S,B,1,N,L]（归一化空间）。"""
        B, _, N, L = cond.shape[0], None, cond.shape[2], self.pre_len
        out = []
        ts = torch.linspace(0, self.diffusion_steps - 1, ddim_steps,
                            device=cond.device).long().unique_consecutive()
        ts = ts.flip(0)
        for s0 in range(0, n_samples, sample_chunk):
            s1 = min(s0 + sample_chunk, n_samples)
            cs = s1 - s0
            cond_r = cond.repeat_interleave(cs, 0)
            x = torch.randn(cs * B, 1, N, L, device=cond.device,
                            generator=generator)
            for i, t in enumerate(ts):
                tt = t.expand(cs * B)
                eps = self.denoise(x, tt, cond_r)
                a_t = self.alphas_cumprod[t]
                x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-10, 10)
                if i + 1 >= len(ts):
                    x = x0
                else:
                    a_p = self.alphas_cumprod[ts[i + 1]]
                    sig2 = (eta ** 2) * ((1 - a_p) / (1 - a_t)) \
                        * (1 - a_t / a_p).clamp_min(0)
                    x = (a_p.sqrt() * x0
                         + (1 - a_p - sig2).clamp_min(0).sqrt() * eps
                         + sig2.sqrt() * torch.randn(
                             x.shape, device=x.device, generator=generator))
            out.append(x.view(cs, B, 1, N, L))
        return torch.cat(out, 0)

    @staticmethod
    def denorm(y_norm, loc, scale):
        """y_norm [S,B,1,N,L]；loc/scale [B,1,N] -> [S,B,N,L]。"""
        sc = scale.unsqueeze(0).unsqueeze(-1)     # [1,B,1,N,1]
        lo = loc.unsqueeze(0).unsqueeze(-1)
        return (y_norm * sc + lo).squeeze(2)
