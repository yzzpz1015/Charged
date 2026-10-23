# -*- coding: utf-8 -*-
"""CSDI 采样掩码消融：on/off 判为"有电"的格子才生成，其余钳成 0。

为什么值得单独一个脚本：JHB 上误差预算拆下来，扩散器本身在**真值掩码**下
R² 0.551 / MAE 3.207（比冻结主干点预测还好），可换成分类器掩码就掉到 0.26。
也就是说做法④当前的下限不是"扩散生成得像不像"，而是"on/off 判得准不准"。
这里把几种掩码口径并排跑，用来判断下一步该修哪一头。

口径
----
clf      分类器 p_on 做伯努利抽样（评测脚本用的就是这个）
top      p_on > 阈值 硬判（去掉抽样噪声，看"判错"本身值多少）
all      全生成，不做任何钳制（完全不依赖 on/off，全靠 p_on 条件去抑制零格）
oracle   真值掩码（无泄漏上限，标出"还剩多少"）
"""
import os
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import kairos_prob_eval as ke                                        # noqa: E402
import prob_baseline as pb                                           # noqa: E402
from api.model.csdi import base_cond, pack_cond, hist_feats           # noqa: E402

CITY = sys.argv[1] if len(sys.argv) > 1 else "JHB"
DEV = torch.device(f"cuda:{sys.argv[2] if len(sys.argv) > 2 else 0}")
NWIN, NS = 120, 16


def r2(a, b):
    return 1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum()


def main():
    core = ke.load_csdi(CITY, DEV)
    ds, raw, starts = ke.load_city(CITY)
    N = raw.shape[1]
    hid, H = core["hid"], ke.SEQ_L
    off = np.arange(H)[None, :]
    bb = ke.build_kairos(CITY, "plain", f"{BASE}/result/kairos_l1", DEV, 99)
    sc = np.asarray(core["scale"])[hid]
    bi = np.asarray(core["bias"])[hid]

    ste = starts["te"]
    ste = ste[::max(1, len(ste) // NWIN)][:NWIN]
    M = len(ste)
    P = ke.point_pass(bb, raw, ste, list(range(H)), DEV)
    pred = np.stack([P[h] for h in range(H)], 2)[:, hid, :]
    obs = np.transpose(raw[ste[:, None] + H + off], (0, 2, 1))[:, hid, :]
    hist = np.transpose((raw[ste[:, None] + np.arange(H)[None, :]] > 0), (0, 2, 1))[:, hid, :]
    hod = np.stack([ds.time[ste + H + h].hour.values for h in range(H)], 1).astype(np.float32)
    pos = obs > 0

    ca, cb = core["cal"]
    pr = torch.from_numpy(np.ascontiguousarray(pred)).float().to(DEV)
    hb = torch.from_numpy(hod).float().to(DEV)[:, None, :]
    a_, o_ = hist_feats(torch.from_numpy(hist).float().to(DEV))
    base = base_cond(pr, torch.from_numpy(sc).float().to(DEV), hb, a_, o_, N, H)
    with torch.no_grad():
        p_on = torch.sigmoid(ca * core["ono"](base) + cb)
    cond = pack_cond(base, p_on, N, H, core.get('cstats'))

    # CRPS 归一化分母：训练期站均量级，**同一套 live_cols 口径**（与评测脚本一致）。
    y0, t0 = pb.load(CITY)
    _ok, trm, _vam, _tem, _m, _s = pb.split_rows(y0, t0)
    _cols, smean = pb.live_cols(y0, trm)
    assert len(_cols) == N, f"站数不一致：Kairos {N} vs 基线 {len(_cols)}"
    sm_h = np.asarray(smean)[hid]           # 台站轴已是 Hilbert 序，分母跟着排

    J = 24

    def run(mode, thr=None):
        Y, MM = [], []
        cs = np.zeros(J)                    # 每 horizon 的「先加总再算」CRPS 累加
        per = np.zeros((J, N))              # 每 horizon 每站的边际 CRPS 累加
        for c0 in range(0, M, 20):
            hi = min(c0 + 20, M)
            with torch.no_grad():
                if mode == "clf":
                    mm = (torch.rand(p_on[c0:hi].shape, device=DEV,
                                     generator=torch.Generator(device=DEV).manual_seed(c0))
                          < p_on[c0:hi]).float()
                elif mode == "top":
                    mm = (p_on[c0:hi] > thr).float()
                elif mode == "oracle":
                    mm = torch.from_numpy(pos[c0:hi]).float().to(DEV)[:, None]
                else:
                    mm = torch.ones_like(p_on[c0:hi])
                smp = core["dif"].sample(core["net"], cond[c0:hi], mm, torch.zeros_like(mm),
                                         n_samples=NS, stride=ke.CSDI_STRIDE,
                                         generator=torch.Generator(device=DEV).manual_seed(0))
            y = (pred[c0:hi][None] + bi[None, None, :, None]
                 + sc[None, None, :, None] * smp[:, :, 0].cpu().numpy())
            y = y * mm[:, 0].cpu().numpy()[None]          # 钳成 0 的格子输出确实是 0
            Y.append(y)
            MM.append(mm[:, 0].cpu().numpy())
            # CRPS 口径与 kairos_prob_eval.csdia_pass 逐字一致：边际 crps 按站存、
            # sum 口径对每窗先跨站求和再算。obs 也是 Hilbert 序，两边对得上。
            for j in range(J):
                for k in range(hi - c0):
                    sm = y[:, k, :, j]
                    cs[j] += float(pb.crps_fair(sm.sum(1), obs[c0 + k, :, j].sum()))
                    per[j] += pb.crps_fair(sm, obs[c0 + k, :, j])
        return np.concatenate(Y, 1), np.concatenate(MM, 0), cs, per

    print(f"{'掩码':<8} {'R2 / MAE (H=1..24)':<26} {'PICP50/80':<12} "
          f"{'crpsΣsum':>9}{'crpsΣ':>8}  真零格输出(应≈0)")
    for mode, thr in (("clf", None), ("top", 0.5), ("top", 0.2), ("all", None), ("oracle", None)):
        Y, MM, cs, per = run(mode, thr)
        ym = Y.mean(0)[:, :, :24]
        O = obs[:, :, :24]
        lo = np.quantile(Y[:, :, :, :24], 0.25, 0)
        hi = np.quantile(Y[:, :, :, :24], 0.75, 0)
        lo8 = np.quantile(Y[:, :, :, :24], 0.10, 0)
        hi8 = np.quantile(Y[:, :, :, :24], 0.90, 0)
        # 跨 horizon 取平均（H=1..24），归一化分母同评测脚本。
        c_ss = float(np.mean(cs / M / sm_h.sum()))
        c_mg = float(np.mean(per.mean(0) / M / sm_h.mean()))
        tag = mode if thr is None else f"{mode}>{thr}"
        print(f"{tag:<8} R2 {r2(ym, O):+.4f}  MAE {np.abs(ym - O).mean():.3f}    "
              f"{((O > lo) & (O < hi)).mean():.2f}/"
              f"{((O > lo8) & (O < hi8)).mean():.2f}    "
              f"{c_ss:>9.4f}{c_mg:>8.4f}   {ym[~pos[:, :, :24]].mean():8.2f}")

    # p_on 到底看不看得见"零"：把真零格与真正格的概率分布分开放。
    poz = pos[:, :, :24]
    pv = p_on[:, 0, :, :24].cpu().numpy()
    print(f"\np_on 分布（分位数）：")
    print(f"{'':<10}{'5%':>7}{'25%':>7}{'50%':>7}{'75%':>7}{'95%':>7}{'均值':>7}")
    for nm, msk in (("真为正", poz), ("真为零", ~poz)):
        q = np.quantile(pv[msk], [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"{nm:<10}{q[0]:7.3f}{q[1]:7.3f}{q[2]:7.3f}{q[3]:7.3f}{q[4]:7.3f}"
              f"{pv[msk].mean():7.3f}")
    # AUC（真零当负类）：0.5 = 完全分不开，与正例率常数一样没用。
    from sklearn.metrics import roc_auc_score
    print(f"AUC(p_on | 真为正 vs 真为零) = {roc_auc_score(poz.ravel(), pv.ravel()):.4f}"
          f"（正例率 {poz.mean():.3f}；真零格中 p_on<0.5 的占 "
          f"{(pv[~poz] < 0.5).mean():.3f}）")

    zc = ~pos[:, :, :24]
    print(f"\n参考：真零格占比 {zc.mean():.3f}；真零格上冻结主干点预测均值 "
          f"{pred[:, :, :24][zc].mean():.2f}")


if __name__ == "__main__":
    main()
