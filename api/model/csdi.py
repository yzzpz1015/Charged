# -*- coding: utf-8 -*-
"""CSDI 式「整块扩散头」：一次生成 [站 × 未来小时] 的完整残差场景。

为什么必须是"整块"
------------------
做法①（点预测 + 残差自助）抽的是**整条残差向量**，站与站的关系原样来自数据，没有
自由度；做法③（分位头）只参数化每站自己的分布，"各站怎么拼成同一时刻"必须人为补
一条规则，而实测那条规则的值能差 20~34%（见 memory quantile-coupling-crps-sum）。

本模型的去噪网络直接在 [站 × 小时] 网格上做卷积，**站间依赖由模型自己学**，输出天然
就是一整个场景。所以 CRPS_sum 在这个模型上重新变成"考模型"，而不是"考拼法"——
这正是把 CRPS_sum 降级时留下的那个边界条件。

三个部件
--------
1. :func:`base_cond` / :func:`pack_cond` 条件打包：主干点预测 / 每站尺度 / 时段 sin-cos
   / 站级活动度两路（:func:`hist_feats`）/ on-off 概率
2. :class:`BlockTrunk` 共享卷积主干（FiLM + 全局因子），分类头和去噪头共用
3. :class:`GaussianDiffusion` DDPM 前向加噪 / 反向采样 / 训练损失

为什么要拆 on/off（"正值条件扩散"是什么意思）
--------------------------------------------
本数据里 y=0 是**真实的空闲状态**，不是缺失值（体积与时长 100% 一致，见 memory
data-zero-analysis）。而且它带持续性：现在零 → 下一小时仍零 84.6~90.4%，现在正 →
转零只有 0.4~6.4%（memory probabilistic-forecasting-direction 结论 1）。

连续扩散表达不了一个有持续性的概率质点：高斯噪声一定会把 0 抹成一条尾巴，预测出
"有一点点电"这种不存在的状态。所以拆成两步：

  * **on/off 分类**：给每个格子一个 P(该格为正)；
  * **正值条件扩散**：扩散模型**只在"被判为正"的格子上生成数值**，零格子作为
    **已知观测**（值恒为 0）喂进去，不参与生成。

这就是 CSDI 的 conditional-imputation 机制换个用法：原论文里"已知观测"是缺失掩码，
这里"已知观测"就是 on/off 里判为零的那些格子。扩散器因此永远不需要去生成一个 0，
它学的是"在已知哪些格是正的、其余条件都摆好的情况下，正值本身长什么样"。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 扩散损失的单格上限（v 误差平方）。逐站归一化后 z 的主体在 ±3 内，但验证段有
# 0.3% 的格子因分布漂移出现 |z|>5，这 0.3% 能贡献 87% 的 MSE —— 不封顶的话
# 验证损失既不下降、挑 epoch 也等于被这些格子绑架。|v|=5 以外的误差按常数计。
LOSS_CLIP = 25.0


# ------------------------------------------------------------------ 基础块
def sinu_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    """DDPM 的步数嵌入：整数步 t -> dim 维 sin/cos。"""
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None, :]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class FiLM2d(nn.Module):
    """[N,H] 网格上的 3x3 卷积，用时间步嵌入做 FiLM 调制，带残差。

    FiLM 的两个映射零初始化 -> 初始时是纯残差块，扩散网络从"什么都不做"开始学，
    比一开始就往特征里灌随机调制稳得多。
    """

    def __init__(self, cin: int, cout: int, temb: int, dilation: int = 1, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, cin)
        self.conv = nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation)
        self.emb = nn.Linear(temb, cout * 2)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.emb.weight)
        nn.init.zeros_(self.emb.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv(F.silu(self.norm(x)))
        s, b = self.emb(temb)[:, :, None, None].chunk(2, dim=1)
        return F.silu(h * (1 + s) + b + self.skip(x))


class BlockTrunk(nn.Module):
    """共享卷积主干：[B,C,N,H] 条件张量 + 步数 t -> [B,base,N,H] 特征。

    站点轴上的卷积要求相邻下标在空间上相邻，所以调用方必须按**空间序**（本仓库用
    Hilbert 曲线，见 api/trainer/common.py:232）排好站点列。见 :func:`hilbert_order`。
    """

    def __init__(self, cin: int, base: int = 64, temb: int = 128,
                 dilations=(1, 2, 4, 8), n_global: int = 2):
        super().__init__()
        self.temb_dim = temb
        self.inp = nn.Conv2d(cin, base, 3, padding=1)
        self.blocks = nn.ModuleList([FiLM2d(base, base, temb, d) for d in dilations])
        # 全局因子：实测残差相关矩阵的主特征值占比 8~17%（script/prob_resid_corr.py，
        # 六城），即存在一个横跨所有站的公共分量。膨胀卷积的感受野再大也是局部的，
        # 补一条池化->广播的通道调制（SE 风格）把这个公共分量显式喂回去。
        self.gl = nn.ModuleList([nn.Sequential(
            nn.Linear(base, base), nn.SiLU(), nn.Linear(base, base))
            for _ in range(n_global)])
        self.temb_mlp = nn.Sequential(nn.Linear(temb, temb), nn.SiLU(),
                                      nn.Linear(temb, temb))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.temb_mlp(sinu_emb(t, self.temb_dim))
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, te)
        for g in self.gl:
            h = h + g(h.mean(dim=(2, 3)))[:, :, None, None]
        return h


class ScoreNet(nn.Module):
    """去噪网络 **v̂(x_t, t, cond) -> [B,1,N,H]**（v-预测）。

    为什么要 v-预测 —— 前两种参数化各塌一边，实测记录如下，别再走回头路：

    * **ε-预测**：``x̂₀ = (x_t − √(1−ᾱ)·ε̂)/√ᾱ``。t→999 处 √ᾱ≈0.0063，分母把 ε̂ 的
      **任何**误差放大 ~157 倍。实测（JHB，训练分布上）模型的 ``ε̂ − ε`` 恒定偏置
      −0.037（真值 ε 均值是 0），而该处 MSE 只有 0.005 —— "照抄 x_t"在最高噪声端是
      白给分，loss 对 x0 的均值毫无约束。0.037×157 = +6，第一步 x̂₀ 直接顶到 clip；
      低噪声端又改不回来（t≈0 时 ε̂≈0，"给什么认什么"是不动点），样本被钉在 +0.22
      个 z（真值 +0.03）→ 点预测 R² = −0.43。
    * **x0-预测**：高噪声端 x0 本来就不可预测，均匀 t 的 MSE 逼网络输出条件均值，
      结果全时段一起塌：MPIW@50/80/90 = 1.83/3.40/4.23（真值该是几十），
      PICP 0.12/0.20/0.23。

    v-预测是 (x0, ε) 的**旋转**：``v = √ᾱ·ε − √(1−ᾱ)·x0``。反解
    ``x̂₀ = √ᾱ·x_t − √(1−ᾱ)·v̂``、``ε̂ = √(1−ᾱ)·x_t + √ᾱ·v̂``，两个系数都 ≤1，
    **没有除法也就没有放大**；而 v 的目标在所有 t 上尺度一致（标准化数据下 ≈N(0,1)），
    高噪声端不是白给分，梯度不会消失。
    """

    PARAM = "v"                      # ckpt 里存这个，防止误用旧参数化权重

    def __init__(self, cond_ch: int, base: int = 64, temb: int = 128, **kw):
        super().__init__()
        self.trunk = BlockTrunk(1 + cond_ch, base=base, temb=temb, **kw)
        self.out = nn.Conv2d(base, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.out(self.trunk(torch.cat([x_t, cond], dim=1), t))


class OnOffNet(nn.Module):
    """on/off 分类器：给每个格子的 P(为正)。输出 logits，[B,1,N,H]。"""

    def __init__(self, cond_ch: int, base: int = 64, temb: int = 128, **kw):
        super().__init__()
        self.trunk = BlockTrunk(cond_ch, base=base, temb=temb, **kw)
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, cond: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        if t is None:
            t = torch.zeros(cond.shape[0], device=cond.device, dtype=torch.long)
        return self.out(self.trunk(cond, t))


# ------------------------------------------------------------------ 条件打包
BASE_COND_NAMES = ("pred", "log_scale", "hod_sin", "hod_cos", "act24", "on_now")
COND_NAMES = BASE_COND_NAMES + ("p_on",)
LS = 0.05                      # on/off 目标的标签平滑（见 train_csdi 里的说明）


def hist_feats(hist_on):
    """从历史 on/off 历史里提出两路**站级活动度**。

    为什么非要这两路：实测（JHB 验证集，无泄漏）把 on/off 的可预测性拆开看，
    点预测/尺度/时段三路加起来的天花板是 BCE 0.396（常数 0.410），基本没有信息；
    而"这个站前 24h 是不是一直在动"把验证集切成三段、有电率分别是 0.94 / 0.74 / 0.54
    —— 这才是 on/off 唯一真正可迁移的特征（站身份本身不迁移，用它反而比常数更差，
    0.522 vs 0.410，因为三段时期不同、各站零率在漂）。

    点预测是**被磨平的量**，24 小时里的零非零图案在它身上已经被抹掉了，所以必须
    单独把历史图案喂进来，不能指望网络从点预测里反推。

    ``hist_on`` : [B, N, Hh] 的 0/1 历史（Hh 是历史小时数）→ 返回 [B,N] 两路。
    """
    act = hist_on[:, :, -24:].float().mean(dim=2)          # 前 24h 为正的比例
    on_now = hist_on[:, :, -1].float()                     # 预测原点那小时是否为正
    return act, on_now


def base_cond(pred, scale, hod, act, on_now, n_station, pre_len):
    """分类器看得见的条件 [B,6,N,H]：点预测 / 每站尺度 / 时段 sin-cos / 活动度两路。

    刻意**不含** p_on —— 分类器自己就是产出 p_on 的那个部件，不能吃自己的输出。
    scale 是每站的残差尺度（训练期统计），取 log 是因为它跨站能差 3 个数量级。
    ``act`` / ``on_now`` 是 :func:`hist_feats` 给的站级量（[B,N]），沿 horizon 广播；
    它们在 h≤24 上有信息、h≥48 上基本没有，网络自己会决定用不用。
    """
    B = pred.shape[0]

    def _bcast(v):
        """[N] / [B,N] / [1,N,1] / [B,N,1] 都收，统一广播成 [B,N,H]。"""
        if v.dim() == 1:
            v = v[None, :, None]
        elif v.dim() == 2:
            v = v[:, :, None]
        return v.expand(B, n_station, pre_len)

    ls = torch.log(_bcast(scale).clamp_min(1e-6))
    hod = _bcast(hod) if hod.dim() > 1 else hod
    two_pi = 2 * math.pi * hod / 24.0
    return torch.stack([pred, ls, torch.sin(two_pi), torch.cos(two_pi),
                        _bcast(act), _bcast(on_now)], dim=1)


def pack_cond(base, p_on, n_station, pre_len, stats=None):
    """扩散器看的条件 [B,7,N,H] = base 六路 + 分类器给的 P(为正)（[B,1,N,H]）。

    ``stats=(mean, std)`` 时逐路标准化（形状 [1,7,1,1]）。**这一条是必须的**，不是
    美化：六路量纲差两三个数量级 —— ``pred`` 是原始用电量（JHB 上 8~60），``hod_sin``
    是 [-1,1]，而扩散的输入 ``x_t`` 自身是 O(1)。量纲差这么大时，第一层卷积的输入
    被条件那几路完全主导，网络会退化成"只看条件、不看输入噪声"的回归器。

    实测（script/csdi_traj_diag.py）：去噪输出与输入的相关系数 ``corr(v̂, x_t)``
    在低噪声端只有 0.13~0.18（本该接近 1）。后果是采样链里"靠输入噪声撑开分布"的
    通路是空的 —— 24 条样本被推向同一个不动点，逐格分散度 0.42（真值该是 0.83），
    直接造成 PICP@50 = 0.21、crpsΣsum 比做法① 高 49.8%。标准化后重训再量这两个数。
    """
    B = base.shape[0]
    cond = torch.cat([base, p_on.expand(B, 1, n_station, pre_len)], dim=1)
    if stats is not None:
        m, s = stats
        cond = (cond - m.to(cond.dtype)) / s.to(cond.dtype)
    return cond


# ------------------------------------------------------------------ 扩散
class GaussianDiffusion(nn.Module):
    """DDPM：线性 beta 表，**x0-预测**，条件修复式采样。

    采样时"已知观测"格子（on/off 判为零的那些）在每一步都被替换回它们的加噪真值，
    与 CSDI 的 conditional imputation 一致；这样扩散器只需要在剩下的格子上生成。

    **步数与 beta 上界必须配套**，这是踩过的坑：``acp[T-1] = ∏(1-β)`` 决定最后一步
    还剩多少原始信号。原先用 steps=50 + β≤0.02 → acp[49]≈0.605，即最高噪声态里
    **还有 78% 的原始信号**，模型从没学过"从纯噪声生成"，而采样却从 N(0,1) 起步
    —— 那个输入落在训练分布之外，模型把它往条件均值收缩，实测生成样本的 std 只有
    真值的 0.36~0.42 倍（把步数改成 11/26/50 都是这个数，所以不是跳步的锅）。
    标准 DDPM 的线性表（β: 1e-4→0.02, T=1000）给出 acp[999]≈4e-5，Σβ≈10，
    这才是"最后一步基本是纯噪声"。改步数时务必让 Σβ 维持在这个量级。
    """

    def __init__(self, steps: int = 1000, beta0: float = 1e-4, beta1: float = 0.02):
        super().__init__()
        self.steps = steps
        betas = torch.linspace(beta0, beta1, steps, dtype=torch.float64)
        acp = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("acp", acp.float())
        self.register_buffer("acp_prev", F.pad(acp[:-1], (1, 0), value=1.0).float())

    def q_sample(self, x0, t, eps):
        a = self.acp[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * eps

    # ---- 训练
    def p_loss(self, net, x0, cond, mask=None):
        """噪声预测损失，**算在全格上**。x0 里零格已被调用方置成 0（见 train_csdi）。

        目标 ``v = √ᾱ·ε − √(1−ᾱ)·x0``（v-参数化，见 :class:`ScoreNet` 的说明）。
        早先只把损失算在"真值为正"的格子上，结果分类器一旦误判为有电，扩散器在那格
        吐的是正值而不是 0（从没有损失要求它输出 0）；改成全格参与后零格目标本就是 0，
        误生成时也只吐 ≈0。

        ``mask`` [B,1,N,H]（或可广播到它）可选：为 0 的格子不计入损失，也不计入
        分母。用于剔除合成直线段 —— 这些格子的值是伪数据，既不该监督重建，也不该
        当作"真零"教模型。
        """
        B = x0.shape[0]
        t = torch.randint(0, self.steps, (B,), device=x0.device)
        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, eps)
        a = self.acp[t][:, None, None, None]
        v = a.sqrt() * eps - (1 - a).sqrt() * x0
        err = ((net(x_t, t, cond) - v) ** 2).clamp(max=LOSS_CLIP)
        if mask is None:
            return err.mean()
        m = mask.expand_as(err)
        return (err * m).sum() / m.sum().clamp_min(1.0)

    # ---- 采样
    @torch.no_grad()
    def sample(self, net, cond, gen_mask, x_obs, n_samples: int = 1,
               generator=None, clip: float = 8.0, stride: int = 1, eta: float = 0.0):
        """生成 [S,B,1,N,H]。

        cond/x_obs/gen_mask 都是 [B,...]；x_obs 是**已知观测**格子的真值（生成格子上的
        取值无所谓，会被盖掉）。同一个 batch 元素重复采样 n_samples 次，共用一套条件。

        gen_mask = 1 表示这一格要**生成**（on/off 判为正），= 0 表示已知观测（真值恒 0）。
        注意别写反：生成格子取随机噪声，观测格子取 x_obs。

        ``stride > 1`` 走 DDIM，只走 ``steps/stride`` 步。评测里这是必需的：2125 个
        测试窗 × 每窗几十个样本 × 1000 步在 LOA/SZH 上是好几个小时。

        ``eta`` 是 DDIM 的随机性系数（Song et al. 2020）：η=0 是确定性采样（旧默认），
        η=1 在每个跳步处注入与"被跳过的 DDPM 链"等方差的噪声。**η=0 会系统性欠离散**
        —— 去噪器给的是条件均值，确定性映射把每个样本都送到均值附近，区间自然比
        真实残差窄，PICP 全线低于名义值。评测要分散度时必须用 η>0。
        """
        B = cond.shape[0]
        dev = cond.device
        cond = cond.repeat(n_samples, 1, 1, 1)
        gen_mask = gen_mask.repeat(n_samples, 1, 1, 1)
        x_obs = x_obs.repeat(n_samples, 1, 1, 1)
        shape = (n_samples * B, 1, *cond.shape[2:])
        x = torch.randn(shape, device=dev, generator=generator) * gen_mask + x_obs * (1 - gen_mask)
        if stride > 1:
            # 从 T-1 起每 stride 步走一格，最后一定落到 0（含 0 那一步）。
            grid = list(range(self.steps - 1, -1, -stride))
            if grid[-1] != 0:
                grid.append(0)
        else:
            grid = list(range(self.steps - 1, -1, -1))
        for k, i in enumerate(grid):
            t = torch.full((n_samples * B,), i, device=dev, dtype=torch.long)
            a = self.acp[i]
            # v-预测反解（见 ScoreNet 说明）：系数都 ≤1，无 1/√ᾱ 放大。
            v = net(x, t, cond)
            x0 = (a.sqrt() * x - (1 - a).sqrt() * v).clamp(-clip, clip)
            prev = grid[k + 1] if k + 1 < len(grid) else -1
            if prev >= 0:
                a_prev = self.acp[prev]
                eps = (1 - a).sqrt() * x + a.sqrt() * v
                if stride > 1:
                    # DDIM：σ=0 时是确定性映射；σ>0 时按 η 注入噪声。
                    # σ² = η² · (1−ᾱ_prev)/(1−ᾱ) · (1−ᾱ/ᾱ_prev)
                    # 这个方差正好等于"被跳过的那 stride 步 DDPM"累积的噪声方差，
                    # 所以 η=1 不只是"抖一下"，是把跳步丢掉的随机性补回来。
                    sig2 = (eta ** 2) * ((1 - a_prev) / (1 - a)) \
                        * (1 - a / a_prev).clamp_min(0)
                    x = (a_prev.sqrt() * x0
                         + (1 - a_prev - sig2).clamp_min(0).sqrt() * eps
                         + sig2.sqrt() * torch.randn_like(x))
                else:
                    # DDPM 后验：均值 = sqrt(ᾱ_prev)·x0 + sqrt(1-ᾱ_prev-σ²)·ε，
                    # σ² = β(1-ᾱ_prev)/(1-ᾱ)。**ε 那一项不能漏** —— 只写
                    # sqrt(ᾱ_prev)·x0 + σ·noise 会系统性缩小幅度（欠离散）。
                    beta = self.betas[i]
                    var = beta * (1 - a_prev) / (1 - a)
                    x = (a_prev.sqrt() * x0
                         + (1 - a_prev - var).clamp_min(0).sqrt() * eps
                         + var.sqrt() * torch.randn_like(x))
                # 条件修复：已知观测格子每一步都被替换回它们在 prev 步的**加噪真值**，
                # 既是输入也是约束；扩散器只在生成格子上自由发挥。
                x = x * gen_mask + self.q_sample(
                    x_obs, torch.full_like(t, prev), torch.randn_like(x)) * (1 - gen_mask)
            else:
                x = x0 * gen_mask + x_obs * (1 - gen_mask)
        return x.view(n_samples, B, *x.shape[1:])


# ------------------------------------------------------------------ 站点空间序
def hilbert_order(lon, lat, bits: int = 16):
    """把站点按 Hilbert 曲线排成一维，**相邻下标在空间上相邻**。

    卷积要在站点轴上做，就必须先给站点一个"谁挨着谁"的次序。散点经纬度本身没有
    次序，Hilbert 曲线的性质是：一维上相邻的两点，二维距离也小（比行扫描好得多）。
    与 api/trainer/common.py:232 的 FreTS 系列用同一套做法。
    """
    import numpy as np
    n = 1 << bits
    lo, la = np.asarray(lon, float), np.asarray(lat, float)
    sx, sy = max(lo.max() - lo.min(), 1e-12), max(la.max() - la.min(), 1e-12)
    x = ((lo - lo.min()) / sx * (n - 1)).astype(np.int64)
    y = ((la - la.min()) / sy * (n - 1)).astype(np.int64)

    def xy2d(x, y):
        d = 0
        s = n >> 1
        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            s >>= 1
        return d

    return np.argsort([xy2d(int(a), int(b)) for a, b in zip(x, y)])
