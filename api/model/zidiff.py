# -*- coding: utf-8 -*-
"""吸收态（零膨胀）整块扩散：把「0」做成扩散过程里的一个**吸收态**。

为什么要另起一个模型，而不是继续调做法④
----------------------------------------
现有做法④（``api/model/csdi.py``）是**两段式**的：

    主干点预测 → on/off 分类器给出**硬掩码** → 扩散器只在硬掩码判"有电"的格子上生成

这个结构有一个不可逆的单向性：**掩码在采样开始前抽一次，就冻死了**。判成 0 的格子
在整条采样链里恒等于 0（``csdi_pass`` 里那句 ``ysim *= m``），再没有机会翻回来。

而我们自己量到的事实是：

* on/off 分类器的验证 BCE = **0.396**，而"永远输出常数"的基准是 **0.410**。
  差 0.014 —— **这个信号几乎不可预测**（api/model/csdi.py:hist_feats 的说明）。
* 于是真零格上仍有大量格子被判成"有电"（真零格上 P(有车) 中位 0.74），
  反向地，真为正的格子也会被钉成 0。

一句话：**做法④ 把一个数据只支持"软着做"的判断，硬做了一遍，而且做错不可逆。**

本模型改的就是这一点：**取消独立分类器**，整个块由一个网络联合建模。网络每一步同时
吐出

    (a) 值通道的 v-预测   ``v̂``
    (b) 掩码通道的 g₀ logits —— 这一格**最终**是不是正的

采样时掩码按 D3PM 的**吸收态反向后验**逐步重抽：一个格子在前几步被判成 0，只要后面的
证据不支持，它**可以再活过来**。硬判断变成沿采样链反复修正的软过程。

前向过程（两通道，各自独立）
----------------------------
值通道（与做法④ 相同）：``x_t = √ᾱ_t·zz + √(1−ᾱ_t)·ε``，其中 ``zz = z·g₀``
    （非正格上本来就恒为 0，所以值通道不需要额外机制）。
掩码通道（吸收态）：``g_t = g₀ ⊙ Bern(ᾱ_t)`` —— 原本为正的格子到第 t 步以 ``1−ᾱ_t``
    的概率被"吸收"（变为 0），**一旦吸收不再复活**。t→T 时 ᾱ→0，整块被吸收。
于是 ``g_t = 1 ⟹ g₀ = 1``；``g_t = 0`` 时 g₀ 未知（本来就是 0，或后来被吸收）。

反向后验（D3PM absorbing）
--------------------------
已知 g_t 与网络给的 ĝ₀：

* ``g_t = 1``  →  ``g_{t−1} = 1``（吸收不可逆，反向同理）
* ``g_t = 0``  →  ``g_{t−1} = 1`` 的概率 ``= [ᾱ_{t−1}(1−β_t)/(1−ᾱ_t)] · P(g₀=1)``

  推导：``P(g_{t−1}=1 | g_t=0, g₀=1) = P(g_{t−1}=1, g_t=0)/P(g_t=0)
  = ᾱ_{t−1}(1−β_t)/(1−ᾱ_t)``，这正是"到 t−1 步其实还活着"的概率；再乘上网络认为
  "这格最终为正"的后验。这就是"复活"概率，只在前向还留有存活可能时非零。

一个必须说清的边界
------------------
值通道与掩码通道是**互相成就**的：末段几步 x_t ≈ zz，掩码头能"看见"值本身，
所以"这格是不是 0"最终仍受值通道的生成质量约束。这不是 bug —— 联合建模本来就该如此
—— 但它意味着：**本模型不会自动把门卫做对，它只是把"硬决定"换成了"边生成边决定"。**
若值通道学得不好，掩码会跟着错，只是错法不同（会来回摆而不是一锤定音）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from api.model.csdi import BlockTrunk, LOSS_CLIP


class ZiScoreNet(nn.Module):
    """联合去噪网络：**一个主干，两个头**。

    输入 ``(x_t, g_t, cond)`` → 输出 ``(v̂, ĝ₀ logits)``。

    与做法④ 的"两网并联"不同：掩码和数值共享同一份中间表示，所以"这格该不该是 0"
    与"这格是多少"是在同一个上下文里、同一次前向里决定的，不会互相打架。

    两个输出头都零初始化：初始时输出恒为 0（v̂=0、logits=0），是纯残差块，比一开始
    就往特征里灌随机输出稳得多。
    """

    PARAM = "zidiff"          # 存进 ckpt，防止拿做法④ 的权重当本模型用

    def __init__(self, cond_ch: int, base: int = 64, temb: int = 128, **kw):
        super().__init__()
        # 通道 = x_t(1) + g_t(1) + 条件(cond_ch)
        self.trunk = BlockTrunk(2 + cond_ch, base=base, temb=temb, **kw)
        self.out_v = nn.Conv2d(base, 1, 3, padding=1)
        self.out_g = nn.Conv2d(base, 1, 3, padding=1)
        nn.init.zeros_(self.out_v.weight)
        nn.init.zeros_(self.out_v.bias)
        nn.init.zeros_(self.out_g.weight)
        nn.init.zeros_(self.out_g.bias)

    def forward(self, x_t, g_t, t, cond):
        h = self.trunk(torch.cat([x_t, g_t, cond], dim=1), t)
        return self.out_v(h), self.out_g(h)


class ZiDiffusion(nn.Module):
    """DDPM 调度（与做法④ **逐字相同**，好让对比只隔离出"掩码怎么定"这一件事）。

    steps/beta 必须配套（Σβ≈10 才让最高噪声端接近纯噪声），理由见
    ``api/model/csdi.py`` 的 GaussianDiffusion 类文档 —— 那个坑踩过一次，不重走。
    """

    def __init__(self, steps: int = 1000, beta0: float = 1e-4, beta1: float = 0.02):
        super().__init__()
        self.steps = steps
        betas = torch.linspace(beta0, beta1, steps, dtype=torch.float64)
        acp = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("acp", acp.float())
        self.register_buffer("acp_prev", F.pad(acp[:-1], (1, 0), value=1.0).float())
        # 吸收态的反向"复活"概率**不能预先按单步算**：采样可以一次跳 stride 步，
        # 正确形式要用起点 t 和终点 t−Δ 的 ᾱ：
        #   P(g_{t−Δ}=1, g_t=0 | g₀=1) = ᾱ_{t−Δ} − ᾱ_t
        #   P(g_{t−Δ}=1 | g_t=0, g₀=1) = (ᾱ_{t−Δ} − ᾱ_t) / (1 − ᾱ_t)   ≤ 1
        # 旧代码写成 ᾱ_{t−1}(1−β_t)/(1−ᾱ_t) = ᾱ_t/(1−ᾱ_t)，是"g_t=1"的概率被当成了
        # "g_t=0 且 g_{t−1}=1"，量级也随 t 漂移（小 t 时 >1，大 t 时趋近 0）。
        # 后果：跳步采样时 g 几乎全程为 0，和训练时 g_t≈g₀（大多为 1）严重错配。
        # 现在改为在 sample() 里按实际的 (t, t−Δ) 现算，见那里的 rev_p。
        self.register_buffer("unabs", torch.zeros(steps))

    # ------------------------------------------------------------------ 前向
    def q_x(self, zz, t, eps):
        a = self.acp[t][:, None, None, None]
        return a.sqrt() * zz + (1 - a).sqrt() * eps

    def q_g(self, g0, t, u=None):
        """吸收态前向：原本为正的格子，到第 t 步以 ``1−ᾱ_t`` 的概率被吸收。"""
        a = self.acp[t][:, None, None, None]
        if u is None:
            u = torch.rand_like(g0)
        return g0 * (u < a).float()

    # ------------------------------------------------------------------ 训练
    def p_loss(self, net, zz, g0, cond, mask=None):
        """两个损失都是**全格元素均值**，各算各的。

        值损失算在全格上（零格目标本来就是 0，被误生成时也只吐 ≈0 —— 这条是做法④
        用误差预算拆出来的结论，见 train_csdi.py 里 ``zz = Z * g`` 那段，直接继承）。
        掩码损失用**未平滑**的真值：平滑会人为抬高 BCE，让它跟"常数基准"没法比。

        ``mask`` [B,1,N,H] 可选：为 0 的格子（合成直线段）两个损失都不计入，
        也不计入分母。
        """
        B = zz.shape[0]
        t = torch.randint(0, self.steps, (B,), device=zz.device)
        eps = torch.randn_like(zz)
        a = self.acp[t][:, None, None, None]
        x_t = self.q_x(zz, t, eps)
        v = a.sqrt() * eps - (1 - a).sqrt() * zz          # v-参数化，见 ScoreNet 文档
        g_t = self.q_g(g0, t)
        v_hat, logit = net(x_t, g_t, t, cond)
        e_v = ((v_hat - v) ** 2).clamp(max=LOSS_CLIP)
        e_g = F.binary_cross_entropy_with_logits(logit, g0, reduction="none")
        if mask is None:
            return e_v.mean(), e_g.mean()
        m = mask.expand_as(e_v)
        den = m.sum().clamp_min(1.0)
        return (e_v * m).sum() / den, (e_g * m).sum() / den

    @torch.no_grad()
    def mask_probe(self, net, zz, g0, cond, t_mid: int = None):
        """体检用的掩码预测：喂**同分布**的带噪输入，返回 logits。

        为什么不能图省事喂 ``x_t=0, g_t=0, t=0``：那组输入根本不在训练分布里
        （训练时 x_t 是高噪的正态扰动、g_t 是吸收态随机变量），网络见到"值为 0 且
        时间步为 0"只会答"这格没电"，于是报出一个远低于真值的正例率 —— 那是
        **探针的偏差，不是模型的偏差**。这里按训练前向的口径造输入。
        """
        B = zz.shape[0]
        t = torch.randint(0, self.steps, (B,), device=zz.device) if t_mid is None \
            else torch.full((B,), int(t_mid), device=zz.device, dtype=torch.long)
        eps = torch.randn_like(zz)
        x_t = self.q_x(zz, t, eps)
        g_t = self.q_g(g0, t)
        return net(x_t, g_t, t, cond)[1]

    # ------------------------------------------------------------------ 采样
    @torch.no_grad()
    def sample(self, net, cond, n_samples: int = 1, generator=None,
               clip: float = 8.0, stride: int = 1, g_thresh: float = 0.5,
               eta: float = 0.0):
        """生成 ``[S,B,1,N,H]`` 的整块场景。

        起点是 **t=T：整块被吸收、值通道是纯噪声**（ᾱ_T≈4e-5），不是做法④ 那种
        "先拿分类器切一刀再生成"。整条链上掩码和值互相修正。

        ``stride>1`` 走 DDIM（η=0，确定性），与做法④ 同一套 —— 评测必须用，
        否则 1000 步 × 每窗几十条在 LOA/SZH 上是好几个小时。
        """
        B = cond.shape[0]
        dev = cond.device
        cond = cond.repeat(n_samples, 1, 1, 1)
        shape = (n_samples * B, 1, *cond.shape[2:])
        x = torch.randn(shape, device=dev, generator=generator)
        g = torch.zeros(shape, device=dev)
        grid = list(range(self.steps - 1, -1, -stride))
        if grid[-1] != 0:
            grid.append(0)
        for k, i in enumerate(grid):
            t = torch.full((n_samples * B,), i, device=dev, dtype=torch.long)
            v_hat, logit = net(x, g, t, cond)
            a = self.acp[i]
            x0 = (a.sqrt() * x - (1 - a).sqrt() * v_hat).clamp(-clip, clip)
            prev = grid[k + 1] if k + 1 < len(grid) else -1
            if prev >= 0:
                a_prev = self.acp[prev]
                eps = (1 - a).sqrt() * x + a.sqrt() * v_hat
                if stride > 1:
                    # DDIM + η：与做法④ 逐字相同（见 csdi.py sample 的说明）。
                    sig2 = (eta ** 2) * ((1 - a_prev) / (1 - a)) \
                        * (1 - a / a_prev).clamp_min(0)
                    x = (a_prev.sqrt() * x0
                         + (1 - a_prev - sig2).clamp_min(0).sqrt() * eps
                         + sig2.sqrt() * torch.randn_like(x))
                else:
                    # DDPM 后验：ε 那一项不能漏，漏了就系统性欠离散（做法④ 踩过）。
                    beta = self.betas[i]
                    var = beta * (1 - a_prev) / (1 - a)
                    x = (a_prev.sqrt() * x0
                         + (1 - a_prev - var).clamp_min(0).sqrt() * eps
                         + var.sqrt() * torch.randn_like(x))
                # 掩码反向：已活的保持活（吸收不可逆）；已吸收的按后验概率复活，
                # 但只有网络认为"这格最终为正"时才可能 —— 两个条件相乘。
                # rev_p 用实际的跳步跨度 (t → t−Δ) 现算，见 __init__ 里那段推导。
                p0 = torch.sigmoid(logit)
                rev_p = ((a_prev - a) / (1 - a).clamp_min(1e-12)).clamp(0, 1)
                rev = (torch.rand(shape, device=dev, generator=generator)
                       < rev_p * p0).float()
                g = g + (1 - g) * rev
            else:
                # 最后一步：**按 p₀ 抽伯努利**，不要阈值化。阈值化会让 24 条样本
                # 在同一个歧义格上做出一模一样的生死决定，等于把 0 处的点质量
                # 从预测分布里删掉 —— 这正是扩散在真零格上系统性欠离散的直接来源。
                p0 = torch.sigmoid(logit)
                g = (torch.rand(shape, device=dev, generator=generator) < p0).float()
                x = x0
            # 值通道**不**因为被吸收而归零。训练时 x_t = √ᾱ·zz + √(1−ᾱ)·ε 对**所有**
            # 格子都是这个分布（zz 在非正格上本来就是 0），把被吸收格的值设成 0 会把
            # x_t 推离训练分布 —— 那等于在采样链上喂网络它没见过的输入。掩码只在最后
            # 一步落到输出上。
        return (x * g).view(n_samples, B, *x.shape[1:])
