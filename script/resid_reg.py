# -*- coding: utf-8 -*-
"""残差回归探针：先减掉「站 × 星期几 × 小时」基线，再看历史特征对残差还有多少增量。

设计：
  * 基线 b = 训练期 (站, 星期几, 小时) 的均值（只用训练期）；
  * 残差 r = y - b；
  * 用**因果**历史特征（截断在预测原点 t-H，复用 zero_peak_gate.build_features）
    回归 r；线性 + GBM；
  * 测试期报告 全部 / 真正 / 峰格（严格 y > 训练期该站正值 p95）上的
    R²、MAE，以及「残差恒为 0」的 MAE 作为对照。

判读：
  * R²≈0 → 历史对「比平时高还是低」没有增量，峰侧不可条件化；
  * R² 有 0.1~0.2 → 有信号，值得做条件化模型；
  * pooled 高、逐站低 → 信号主要来自站身份。

用法::

    python script/resid_reg.py --cities MEL LOA JHB SPO SZH --horizon 24
"""
import argparse
import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import prob_baseline as pb                                          # noqa: E402
from zero_peak_gate import build_features, flatten                   # noqa: E402

MAX_CELLS = 400_000
PEAK_MIN = 100
warnings.filterwarnings("ignore", category=RuntimeWarning)


def time_baseline(y, t0, trm):
    """训练期 (站, 星期几, 小时) 均值 -> [T,N]。"""
    g = t0.dayofweek.values * 24 + t0.hour.values
    valid = trm & np.isfinite(y).all(1)          # 行级：可用且无 NaN
    N = y.shape[1]
    sums = np.zeros((168, N))
    cnts = np.zeros((168, N))
    np.add.at(sums, g[valid], y[valid])
    np.add.at(cnts, g[valid], np.ones_like(y[valid]))
    b = sums / np.maximum(cnts, 1.0)
    sm = np.nanmean(y[valid], axis=0)
    b = np.where(cnts > 0, b, sm[None, :])
    gm = np.nanmean(sm)
    b = np.where(np.isfinite(b), b, gm)
    return b[g]


def peak_thr(y, trm, N):
    thr = np.full(N, np.nan)
    for s in range(N):
        v = y[trm & np.isfinite(y[:, s]), s]
        v = v[v > 0]
        if v.size >= PEAK_MIN:
            thr[s] = np.quantile(v, 0.95)
    return thr


def stat(pred, true):
    e = pred - true
    mae = np.mean(np.abs(e))
    sst = np.sum((true - true.mean()) ** 2)
    r2 = 1.0 - np.sum(e ** 2) / sst if sst > 0 else np.nan
    return r2, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["MEL"])
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()
    h = args.horizon
    print(f"H={h}（特征截断在 t-{h}；基线只用训练期）", flush=True)

    for city in args.cities:
        y0, t0 = pb.load(city)
        ok, trm, vam, tem, u, span = pb.split_rows(y0, t0)
        cols, smean = pb.live_cols(y0, trm)
        y = y0[:, cols]
        N = y.shape[1]
        b = time_baseline(y, t0, trm)
        r = y - b
        thr = peak_thr(y, trm, N)

        tr_rows = np.where(trm)[0]
        te_rows = np.where(tem)[0]
        tr_rows = tr_rows[tr_rows >= h + 72]
        te_rows = te_rows[te_rows >= h + 72]
        rng = np.random.default_rng(0)
        tr_rows = rng.choice(tr_rows, min(MAX_CELLS // N, len(tr_rows)), replace=False)
        te_rows = rng.choice(te_rows, min(MAX_CELLS // N, len(te_rows)), replace=False)
        tr_on = np.nanmean((y[tr_rows] > 0).astype(np.float32), 0)
        sm = np.nanmean(y[tr_rows], 0)
        Ftr = build_features(y, t0, tr_rows, sm, tr_on, h)
        Fte = build_features(y, t0, te_rows, sm, tr_on, h)
        rtr = r[tr_rows]
        rte = r[te_rows]
        Xtr, _, _ = flatten(Ftr, rtr, tr_rows)
        Xte, _, ste = flatten(Fte, rte, te_rows)
        yte = y[te_rows].reshape(-1)
        keeps = np.isfinite(yte) & np.isfinite(rte).reshape(-1)
        Xtr, rtrf = Xtr[np.isfinite(rtr.reshape(-1))], rtr.reshape(-1)[np.isfinite(rtr.reshape(-1))]
        Xte, rtef = Xte[keeps], rte.reshape(-1)[keeps]
        yte = yte[keeps]
        ste = ste[keeps]

        sc = StandardScaler().fit(Xtr)
        lin = LinearRegression().fit(sc.transform(Xtr), rtrf)
        gbm = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1,
                                            max_leaf_nodes=31).fit(Xtr, rtrf)
        pl = lin.predict(sc.transform(Xte))
        pg = gbm.predict(Xte)

        pk = np.zeros(len(yte), bool)
        okthr = np.isfinite(thr)
        pk[okthr[ste]] = yte[okthr[ste]] > thr[ste[okthr[ste]]]
        masks = {"全部": np.ones(len(yte), bool), "真正": yte > 0, "峰格": pk}
        print(f"\n===== {city}  测试格 {len(yte)}  峰格 {pk.sum()} "
              f"({100 * pk.mean():.1f}%) =====", flush=True)
        print(f"{'子集':<6}{'n':>9}{'零残差MAE':>11}{'线性R2':>9}{'线性MAE':>10}"
              f"{'GBM R2':>9}{'GBM MAE':>10}", flush=True)
        for nm, mm in masks.items():
            if not mm.any():
                continue
            r2l, mael = stat(pl[mm], rtef[mm])
            r2g, maeg = stat(pg[mm], rtef[mm])
            print(f"{nm:<6}{mm.sum():>9d}{np.mean(np.abs(rtef[mm])):>11.3f}"
                  f"{r2l:>9.3f}{mael:>10.3f}{r2g:>9.3f}{maeg:>10.3f}", flush=True)
        # 逐站 GBM R²（只统计样本足够的站，报中位数）
        vals = []
        for s in np.unique(ste):
            mm = (ste == s) & (yte > 0)
            if mm.sum() >= 200 and np.var(rtef[mm]) > 0:
                vals.append(stat(pg[mm], rtef[mm])[0])
        if vals:
            print(f"  逐站(真正格) GBM R² 中位数 {np.median(vals):.3f}  "
                  f"（{len(vals)} 站）", flush=True)


if __name__ == "__main__":
    main()
