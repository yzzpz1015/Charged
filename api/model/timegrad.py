# -*- coding: utf-8 -*-
"""#15 TimeGrad：GRU 编码历史 + 条件 DDPM 生成未来 72 步。

与原论文（Rasul et al., 2021）的关系与差异：

* **保留**：RNN（这里用单层 GRU）读 72h 历史得到隐状态 h；扩散模型以 h 和
  扩散步 t 为条件（FiLM）在归一化空间去噪；DDPM 线性 beta、ε-参数化、
  raw 空间反归一化。
* **差异（表注必须写）**：原版在 horizon 上**逐步自回归**（第 j 步以第 j-1 步
  采样值为 RNN 输入，每步独立扩散）；这里为满足统一协议「一次前向出 72 步」
  且控制采样成本，改为**对 72 维未来向量做联合扩散**（非自回归），RNN 只
  编码历史。采样用 50 步 DDIM（原版 1000 步 ancestral），同样在表注写明。

归一化：CausalRevIN，只用本窗 72h 历史（与其它 baseline 完全一致）。
噪声预测在归一化空间训练，采样后反归一化并 clamp 到 ≥0（volume 非负）。
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from api.model.quant_head import CausalRevIN


def _sinu(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32)
                     * (torch.log(torch.tensor(10000.0, device=t.device)) / (half - 1)))
    ang = t.float()[:, None] * freq[None, :]
    return torch.cat([ang.sin(), ang.cos()], -1)


class _FiLMBlock(nn.Module):
    """因果空洞残差块：GroupNorm → conv → FiLM(cond) → GELU → conv。"""

    def __init__(self, ch: int, cond_dim: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = 2 * dilation                # kernel=3 因果：(k-1)*d
        self.norm = nn.GroupNorm(4, ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, 3, dilation=dilation)
        self.film = nn.Linear(cond_dim, 2 * ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.pad(self.norm(x), (self.pad, 0)))
        s, b = self.film(cond).chunk(2, -1)
        h = F.gelu(h * (1.0 + s[:, :, None]) + b[:, :, None])
        h = self.drop(h)
        h = self.conv2(F.pad(h, (self.pad, 0)))
        return x + h


class TimeGradModel(nn.Module):
    """逐站单变量；station 维拍进 batch。

    参数
    ----
    num_nodes : 仅作接口兼容（结构本身与站数无关）
    """

    def __init__(self, num_nodes: int, seq_len: int = 72, pre_len: int = 72,
                 hidden: int = 64, d_model: int = 64, n_layers: int = 3,
                 dropout: float = 0.1, diffusion_steps: int = 1000,
                 beta0: float = 1e-4, beta1: float = 0.02, **_):
        super().__init__()
        del num_nodes
        self.pre_len = pre_len
        self.diffusion_steps = diffusion_steps
        self.revin = CausalRevIN()

        # ---- 历史编码：单层 GRU
        self.gru = nn.GRU(1, hidden, num_layers=1, batch_first=True)
        self.h2c = nn.Sequential(nn.Linear(hidden, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))
        self.temb = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())

        # ---- 去噪网络：因果 1D conv + FiLM
        self.inp = nn.Conv1d(1, d_model, 3, padding=1)
        self.blocks = nn.ModuleList([
            _FiLMBlock(d_model, d_model, 2 ** i, dropout)
            for i in range(n_layers)])
        self.out_norm = nn.GroupNorm(4, d_model)
        self.out = nn.Conv1d(d_model, 1, 3, padding=1)

        # ---- DDPM 常数
        betas = torch.linspace(beta0, beta1, diffusion_steps)
        ac = torch.cumprod(1.0 - betas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", ac)

    # ------------------------------------------------------------ 前向
    def encode(self, x_hist: torch.Tensor) -> torch.Tensor:
        """[R,T] 归一化历史 -> [R,hidden]。"""
        _, h = self.gru(x_hist.unsqueeze(-1))
        return h[-1]

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """x_t [R,L]；t [R] long；cond [R,d] -> ε̂ [R,L]。"""
        c = cond + self.temb(_sinu(t, cond.shape[-1]))
        h = self.inp(x_t.unsqueeze(1))                   # [R,d,L]
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(F.gelu(self.out_norm(h))).squeeze(1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 eps: torch.Tensor) -> torch.Tensor:
        a = self.alphas_cumprod[t][:, None]
        return a.sqrt() * x0 + (1.0 - a).sqrt() * eps

    def p_loss(self, x0: torch.Tensor, cond: torch.Tensor,
               weight: torch.Tensor = None, generator=None) -> torch.Tensor:
        """x0 [R,L] 归一化未来；cond [R,d]；weight [R] 0/1。"""
        R, L = x0.shape
        t = torch.randint(0, self.diffusion_steps, (R,), device=x0.device,
                          generator=generator)
        eps = torch.randn(x0.shape, device=x0.device, generator=generator)
        xt = self.q_sample(x0, t, eps)
        pred = self.denoise(xt, t, cond)
        err = (pred - eps) ** 2
        if weight is None:
            return err.mean()
        w = weight[:, None].to(err.dtype)
        return (err * w).sum() / w.sum().clamp_min(1.0) / L

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_samples: int = 1,
               ddim_steps: int = 50, generator=None,
               row_chunk: int = 65536, eta: float = 1.0) -> torch.Tensor:
        """条件采样。cond [R,d] -> [n_samples,R,L]（归一化空间）。

        用 DDIM（η=0）加速；每个条件独立重复 n_samples 份噪声。
        """
        R, L = cond.shape[0], self.pre_len
        cond_rep = cond.repeat_interleave(n_samples, 0)          # [R*S,d]
        x = torch.randn(cond_rep.shape[0], L, device=cond.device,
                        generator=generator)
        ts = torch.linspace(0, self.diffusion_steps - 1, ddim_steps,
                            device=cond.device).long().unique_consecutive()
        ts = ts.flip(0)
        for i, t in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else torch.tensor(
                -1, device=cond.device)
            eps = torch.empty_like(x)
            for r0 in range(0, x.shape[0], row_chunk):
                r1 = min(r0 + row_chunk, x.shape[0])
                tt = t.expand(r1 - r0)
                eps[r0:r1] = self.denoise(x[r0:r1], tt, cond_rep[r0:r1])
            a_t = self.alphas_cumprod[t]
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            x0 = x0.clamp(-10.0, 10.0)                # 归一化空间防爆
            if t_prev < 0:
                x = x0
            else:
                a_p = self.alphas_cumprod[t_prev]
                # DDIM η：η=0 是确定性映射（会系统性欠离散），η=1 把跳过的
                # DDPM 随机性补回来（与 api/model/csdi.py 的结论一致）。
                sig2 = (eta ** 2) * ((1 - a_p) / (1 - a_t)) \
                    * (1 - a_t / a_p).clamp_min(0)
                x = (a_p.sqrt() * x0
                     + (1 - a_p - sig2).clamp_min(0).sqrt() * eps
                     + sig2.sqrt() * torch.randn(
                         x.shape, device=x.device, generator=generator))
        # [R,S,L] -> [S,R,L]
        return x.view(R, n_samples, L).permute(1, 0, 2).contiguous()

    # ------------------------------------------------------------ 接口
    @staticmethod
    def denorm(y_norm: torch.Tensor, loc: torch.Tensor,
               scale: torch.Tensor) -> torch.Tensor:
        """y_norm [S,R,L]；loc/scale [R,1] -> [S,R,L]。"""
        return y_norm * scale[None] + loc[None]


@torch.no_grad()
def encode_and_cond(model: TimeGradModel, x_hist: torch.Tensor,
                    loc: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x_hist [R,T] raw -> cond [R,d]，并回传 loc/scale [R,1]。"""
    xn = (x_hist - loc) / scale
    return model.encode(xn), loc, scale
