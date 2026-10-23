# -*- coding: utf-8 -*-
"""把 DDIM 采样轨迹逐步拆开：每一步的 x̂₀ 宽度是在哪一步丢掉的？

为什么要单开一个脚本 —— 证据出现了矛盾：

  * 去噪网络本身在训练集上不差：t=950 处 corr(v̂,v)=0.80、解释 v 的 62% 方差
    （script/csdi_denoiser_check.py）；
  * 可它产出的样本宽度只有真值的 0.26（script/csdi_spread_diag.py），而且换成
    20 步 DDIM / 200 步 / 1000 步完整 DDPM 都是 0.26 —— 与采样步数无关。

"去噪器够好"却"样本塌掉"，通常意味着采样循环里有一步写错了。本脚本**逐字复刻**
``GaussianDiffusion.sample`` 的 DDIM 分支，但把每一步的中间量打出来：

  std(x_t)     进入该步时的样本宽度（x 是标准化残差，真值约 1.0~1.4）
  std(x̂₀)      该步反解出的干净估计宽度 —— **这是最该盯的一列**。它若一直 ≈1，
               说明网络每步都给出了正确宽度的 x̂₀，那宽度是在"回代成 x_{prev}"时丢的；
               它若一步就塌到 0.3，说明网络本身就吐窄东西。
  corr(x̂₀,真x0) 每步估计与真值的相关，看是"宽度丢了"还是"方向也偏了"。

用法：``python3 script/csdi_traj_diag.py [city] [device] [stride]``
"""
import os
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import kairos_prob_eval as ke                                        # noqa: E402
from api.model.csdi import base_cond, pack_cond, hist_feats           # noqa: E402

CITY = sys.argv[1] if len(sys.argv) > 1 else "JHB"
DEV = torch.device(f"cuda:{sys.argv[2] if len(sys.argv) > 2 else 0}")
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 50
SPLIT = sys.argv[4] if len(sys.argv) > 4 else "tr"
NWIN, NS = 24, 16


def main():
    from api.dataset.common import SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST
    core = ke.load_csdi(CITY, DEV)
    ds, raw, _ = ke.load_city(CITY)
    N = raw.shape[1]
    hid, H = core["hid"], ke.SEQ_L
    off = np.arange(H)[None, :]
    bb = ke.build_kairos(CITY, "plain", f"{BASE}/result/kairos_l1", DEV, 99)

    ste = ds.window_starts({"tr": SPLIT_TRAIN, "va": SPLIT_VALID,
                            "te": SPLIT_TEST}[SPLIT], ke.SEQ_L, ke.SEQ_L)
    ste = ste[::max(1, len(ste) // NWIN)][:NWIN]
    M = len(ste)
    P = ke.point_pass(bb, raw, ste, list(range(H)), DEV)
    pred = np.stack([P[h] for h in range(H)], 2)[:, hid, :]
    obs = np.transpose(raw[ste[:, None] + H + off], (0, 2, 1))[:, hid, :]
    hist = np.transpose((raw[ste[:, None] + np.arange(H)[None, :]] > 0), (0, 2, 1))[:, hid, :]
    hod = np.stack([ds.time[ste + H + h].hour.values for h in range(H)], 1).astype(np.float32)

    ca, cb = core["cal"]
    sc_h, bi_h = np.asarray(core["scale"])[hid], np.asarray(core["bias"])[hid]
    pr = torch.from_numpy(np.ascontiguousarray(pred)).float().to(DEV)
    hb = torch.from_numpy(hod).float().to(DEV)[:, None, :]
    a_, o_ = hist_feats(torch.from_numpy(hist).float().to(DEV))
    base = base_cond(pr, torch.from_numpy(sc_h).float().to(DEV), hb, a_, o_, N, H)
    with torch.no_grad():
        p_on = torch.sigmoid(ca * core["ono"](base) + cb)
    cond = pack_cond(base, p_on, N, H, core.get('cstats'))

    # 真值 x0（标准化残差，零格置 0）—— 用来算逐步 corr，以及"真值宽度"参照。
    zt = ((obs - pred) - bi_h[None, :, None]) / sc_h[None, :, None]
    x0_true = torch.from_numpy(np.ascontiguousarray(zt * (obs > 0))).float().to(DEV).unsqueeze(1)

    dif, net = core["dif"], core["net"]
    gen = torch.Generator(device=DEV).manual_seed(0)
    m = (torch.rand(p_on.shape, device=DEV, generator=gen) < p_on).float()
    cnd = cond.repeat(NS, 1, 1, 1)
    gm = m.repeat(NS, 1, 1, 1)
    x_obs = torch.zeros_like(gm)
    xt = x0_true.repeat(NS, 1, 1, 1)
    shape = (NS * M, 1, N, H)
    x = torch.randn(shape, device=DEV, generator=gen) * gm + x_obs * (1 - gm)

    # 只在生成格子上算宽度（零格恒 0，掺进来会把 std 拉低，是假信号）。
    sel = gm[:, 0].bool()
    x0t_sel = xt[:, 0][sel]
    # 两种宽度必须分开量，这是本脚本存在的理由：
    #   跨格 std  —— 一条样本内部的"空间花纹"幅度（消融里量的是这个的兄弟：单条样本）
    #   跨样本 std —— 同一格上 S 条样本的分散度（= 模型表达的"不确定性"，PICP 看这个）
    # 实测：前者 0.91（好），后者 0.37（塌）。所以要看多样性是在哪一步死的。
    sel_m = m[:, 0].bool()               # [M,N,H]（sel 是它沿样本维复制 NS 份）

    def two_std(mx):                     # mx: [S*M,1,N,H] -> (跨格, 跨样本)
        q = mx[:, 0].reshape(NS, M, N, H)[:, sel_m]           # [NS, K]
        return float(q.reshape(-1).std()), float(q.std(0).mean())

    grid = list(range(dif.steps - 1, -1, -STRIDE))
    if grid[-1] != 0:
        grid.append(0)
    print(f"\n{CITY} [{SPLIT}] 采样轨迹拆解（{M} 窗 × {NS} 样本，跳步 {STRIDE}"
          f"={len(grid)} 步，只统计生成格子）")
    print(f"真值 x0 在这些格子上的 std = {float(x0t_sel.std()):.3f}"
          f"（这是「该有的宽度」）\n")
    print(f"{'步':>4}{'t':>6}{'√ᾱ':>7}{'x_t 跨格':>10}{'x_t 跨样本':>11}"
          f"{'x̂₀ 跨格':>10}{'x̂₀ 跨样本':>11}{'corr(x̂₀,x0)':>12}{'corr(v̂,x_t)':>12}")
    for k, i in enumerate(grid):
        t = torch.full((NS * M,), i, device=DEV, dtype=torch.long)
        a = dif.acp[i]
        with torch.no_grad():
            v = net(x, t, cnd)
            x0 = (a.sqrt() * x - (1 - a).sqrt() * v).clamp(-8, 8)
        cx = float(torch.corrcoef(torch.stack([x0[:, 0][sel], x0t_sel]))[0, 1])
        # v̂ 对输入的敏感度：若 corr(v̂, x_t)≈0，说明网络根本不看 x_t，只照条件吐一个定值
        # —— 那就是模式坍塌的机制（所有样本被推向同一个不动点）。
        vv = v[:, 0].reshape(NS, M, N, H)[0][sel_m]
        cvx = float(torch.corrcoef(torch.stack([vv, x[:, 0].reshape(NS, M, N, H)[0][sel_m]]))[0, 1])
        if k % max(1, len(grid) // 10) == 0 or k == len(grid) - 1:
            gx, sx = two_std(x)
            g0, s0 = two_std(x0)
            print(f"{k:>4}{i:>6}{float(a.sqrt()):>7.3f}{gx:>10.3f}{sx:>11.3f}"
                  f"{g0:>10.3f}{s0:>11.3f}{cx:>12.3f}{cvx:>12.3f}")
        prev = grid[k + 1] if k + 1 < len(grid) else -1
        if prev >= 0:
            a_prev = dif.acp[prev]
            eps = (1 - a).sqrt() * x + a.sqrt() * v
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
            x = x * gm + dif.q_sample(x_obs, torch.full_like(t, prev),
                                      torch.randn_like(x)) * (1 - gm)
    print(f"\n最后一步：跨格 std = {two_std(x * gm)[0]:.3f}，跨样本 std = {two_std(x * gm)[1]:.3f}"
          f"（真值 {float(x0t_sel.std()):.3f}）")


if __name__ == "__main__":
    main()
