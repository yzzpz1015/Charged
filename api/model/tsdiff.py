# -*- coding: utf-8 -*-
"""#16 TSDiff：非自回归条件扩散 + observation self-guidance。

与原论文（Kollovieh et al., 2023）的关系与差异：

* **保留**：整段序列（历史 72 + 未来 72 = 144）联合建模；去噪网络**无 RNN**
  （dilated conv + 时间 FiLM）；采样时用已知历史做 **self-guidance**
  （原版是软梯度引导，这里是硬投影，见下）；DDPM 线性 beta、ε-参数化。
* **差异（表注必须写）**：原版 self-guidance 用观测一致性损失的梯度做软引导，
  这里在每一步 DDIM 的 x0 估计上直接把历史段替换为真值（hard guidance）；
  采样用 50 步 DDIM 而非 1000 步 ancestral；不额外做 unconditional 预训练。

归一化：与其它 baseline 相同的 CausalRevIN（只用 72h 历史）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from api.model.quant_head import CausalRevIN
from api.model.timegrad import _FiLMBlock, _sinu


class TSDiffModel(nn.Module):
    def __init__(self, num_nodes: int, seq_len: int = 72, pre_len: int = 72,
                 d_model: int = 64, n_layers: int = 5, dropout: float = 0.1,
                 diffusion_steps: int = 1000, beta0: float = 1e-4,
                 beta1: float = 0.02, **_):
        super().__init__()
        del num_nodes
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.total = seq_len + pre_len
        self.diffusion_steps = diffusion_steps
        self.revin = CausalRevIN()

        self.temb = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.inp = nn.Conv1d(1, d_model, 3, padding=1)
        self.blocks = nn.ModuleList([
            _FiLMBlock(d_model, d_model, min(2 ** i, 16), dropout)
            for i in range(n_layers)])
        self.out_norm = nn.GroupNorm(4, d_model)
        self.out = nn.Conv1d(d_model, 1, 3, padding=1)

        betas = torch.linspace(beta0, beta1, diffusion_steps)
        ac = torch.cumprod(1.0 - betas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", ac)

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        c = self.temb(_sinu(t, self.temb[0].in_features))
        h = self.inp(x_t.unsqueeze(1))
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(F.gelu(self.out_norm(h))).squeeze(1)

    def p_loss(self, x0: torch.Tensor, weight: torch.Tensor = None,
               generator=None) -> torch.Tensor:
        """x0 [R,144] 归一化整段；weight [R] 0/1（clean mask）。"""
        R, L = x0.shape
        t = torch.randint(0, self.diffusion_steps, (R,), device=x0.device,
                          generator=generator)
        eps = torch.randn(x0.shape, device=x0.device, generator=generator)
        a = self.alphas_cumprod[t][:, None]
        xt = a.sqrt() * x0 + (1.0 - a).sqrt() * eps
        err = (self.denoise(xt, t) - eps) ** 2
        if weight is None:
            return err.mean()
        w = weight[:, None].to(err.dtype)
        return (err * w).sum() / w.sum().clamp_min(1.0) / L

    @torch.no_grad()
    def sample(self, hist_norm: torch.Tensor, n_samples: int = 1,
               ddim_steps: int = 50, generator=None,
               row_chunk: int = 32768, eta: float = 1.0) -> torch.Tensor:
        """hist_norm [R,72] -> 未来 [S,R,72]（归一化空间）。"""
        R = hist_norm.shape[0]
        hist_rep = hist_norm.repeat_interleave(n_samples, 0)
        x = torch.randn(hist_rep.shape[0], self.total,
                        device=hist_norm.device, generator=generator)
        ts = torch.linspace(0, self.diffusion_steps - 1, ddim_steps,
                            device=hist_norm.device).long().unique_consecutive()
        ts = ts.flip(0)
        for i, t in enumerate(ts):
            eps = torch.empty_like(x)
            for r0 in range(0, x.shape[0], row_chunk):
                r1 = min(r0 + row_chunk, x.shape[0])
                eps[r0:r1] = self.denoise(x[r0:r1],
                                          t.expand(r1 - r0))
            a_t = self.alphas_cumprod[t]
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-10.0, 10.0)
            # observation self-guidance：已知历史硬投影回真值
            x0[:, :self.seq_len] = hist_rep
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
        fut = x[:, self.seq_len:]
        return fut.view(R, n_samples, self.pre_len).permute(1, 0, 2).contiguous()

    @staticmethod
    def denorm(y_norm, loc, scale):
        return y_norm * scale[None] + loc[None]
