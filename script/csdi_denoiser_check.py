# -*- coding: utf-8 -*-
"""去噪网络到底学没学到东西：逐 t 比 v̂ 与真值 v，并跟「输出恒 0」这个白给基准对撞。

为什么需要这个 —— 离散度体检发现生成样本的 std 只有真值的 1/4，且**放平**（各时段
都卡在 0.33~0.37，完全不跟条件走）。这种「输出近乎常数」的现象有两个完全不同的成因：

  * 网络欠训练：梯度还没把它从零初始化的输出推出去 → 修训练（步数/lr/时长）
  * 采样链写错：网络学好了但 DDIM 迭代把它抹掉了 → 修 sampler

判据是 ``MSE(v̂, v)`` 与 ``MSE(0, v) = E[v²]`` 谁大。**0 是那个白给基准**：输出恒 0 时
loss 就是 E[v²]（标准化数据下 ≈ E[x0²]，因为 v 是 (x0,ε) 的旋转，两边能量守恒）。
网络若比它还差，说明它不但在白给分上没进步，还在主动犯错 —— 那是欠训练/优化问题；
若明显低于它，说明网络学好了，离散度丢失发生在采样链里。

顺带打印 corr(v̂, v)，直接看顺序对不对（欠训练时相关也接近 0）。
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
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "te"        # te / va / tr
NWIN = 60
TS = [50, 200, 400, 600, 800, 950]


def main():
    core = ke.load_csdi(CITY, DEV)
    ds, raw, starts = ke.load_city(CITY)
    N = raw.shape[1]
    hid, H = core["hid"], ke.SEQ_L
    off = np.arange(H)[None, :]
    bb = ke.build_kairos(CITY, "plain", f"{BASE}/result/kairos_l1", DEV, 99)

    from api.dataset.common import SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST
    sid = {"tr": SPLIT_TRAIN, "va": SPLIT_VALID, "te": SPLIT_TEST}[SPLIT]
    ste = ds.window_starts(sid, ke.SEQ_L, ke.SEQ_L)
    ste = ste[::max(1, len(ste) // NWIN)][:NWIN]
    M = len(ste)
    P = ke.point_pass(bb, raw, ste, list(range(H)), DEV)
    pred = np.stack([P[h] for h in range(H)], 2)[:, hid, :]
    obs = np.transpose(raw[ste[:, None] + H + off], (0, 2, 1))[:, hid, :]
    hist = np.transpose((raw[ste[:, None] + np.arange(H)[None, :]] > 0), (0, 2, 1))[:, hid, :]
    hod = np.stack([ds.time[ste + H + h].hour.values for h in range(H)], 1).astype(np.float32)

    ca, cb = core["cal"]
    sc_h = np.asarray(core["scale"])[hid]
    bi_h = np.asarray(core["bias"])[hid]
    pr = torch.from_numpy(np.ascontiguousarray(pred)).float().to(DEV)
    hb = torch.from_numpy(hod).float().to(DEV)[:, None, :]
    a_, o_ = hist_feats(torch.from_numpy(hist).float().to(DEV))
    base = base_cond(pr, torch.from_numpy(sc_h).float().to(DEV), hb, a_, o_, N, H)
    with torch.no_grad():
        p_on = torch.sigmoid(ca * core["ono"](base) + cb)
    cond = pack_cond(base, p_on, N, H, core.get('cstats'))

    # 真值 x0：与训练同口径（标准化残差，零格置 0）。
    zt = ((obs - pred) - bi_h[None, :, None]) / sc_h[None, :, None]
    x0 = torch.from_numpy(np.ascontiguousarray(zt * (obs > 0))).float().to(DEV).unsqueeze(1)
    dif, net = core["dif"], core["net"]
    a_all = dif.acp

    print(f"\n{CITY} [{SPLIT}]  去噪网络体检（{M} 窗 × {N} 站 × {H} 步）")
    print("基准：v̂≡0 的 loss = E[v²]。网络若高于它 = 它比「什么都不做」还差。")
    print(f"该段真值 x0 的均方 = {float((x0 ** 2).mean()):.3f}"
          f"（训练段按构造≈0.87；差得多就是这段的残差尺度变了）\n")
    print(f"{'t':>5}{'√ᾱ':>8}{'MSE(v̂,v)':>11}{'E[v²] (基准)':>14}{'比值':>8}"
          f"{'corr(v̂,v)':>12}{'std v̂':>9}{'std v':>8}")
    g = torch.Generator(device=DEV).manual_seed(0)
    for t in TS:
        tt = torch.full((M,), t, device=DEV, dtype=torch.long)
        eps = torch.randn(x0.shape, device=DEV, generator=g)
        x_t = dif.q_sample(x0, tt, eps)
        a = a_all[t]
        v = a.sqrt() * eps - (1 - a).sqrt() * x0
        with torch.no_grad():
            vh = net(x_t, tt, cond)
        mse = float(((vh - v) ** 2).mean())
        triv = float((v ** 2).mean())
        cv = float(torch.corrcoef(torch.stack([vh.flatten(), v.flatten()]))[0, 1])
        print(f"{t:>5}{float(a.sqrt()):>8.3f}{mse:>11.4f}{triv:>14.4f}"
              f"{mse / triv:>8.2f}{cv:>12.3f}{float(vh.std()):>9.3f}{float(v.std()):>8.3f}")


if __name__ == "__main__":
    main()
