# -*- coding: utf-8 -*-
"""正值幅度头探针：只在 y>0 的格子上训「有车时多少度」，再在真正/峰格上比。

对照：
  * const   : 逐站训练期正值均值（不给历史）
  * lin_pos : 正值格线性回归（目标 y）
  * gbm_pos : 正值格 GBM（目标 y，平方损失）
  * gbm_log : 正值格 GBM（目标 log1p(y)，重尾更稳）
  * gbm_all : 所有格子 GBM（含零，目标 y）——「不拆零」的对照

指标：测试期真正格 / 峰格上的 MAE、R²、bias。峰格 = y > 训练期该站正值 p95。

用法::

    python script/positive_mag.py --cities MEL LOA JHB SPO SZH --horizon 24
"""
import argparse
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import prob_baseline as pb                                          # noqa: E402
from zero_peak_gate import build_features, flatten                   # noqa: E402
from resid_reg import peak_thr, stat                                # noqa: E402

MAX_ALL = 400_000
MAX_POS = 300_000
PEAK_MIN = 100


def add_static(F, pos_mean, thr):
    N = pos_mean.shape[0]
    extra = np.stack([
        np.broadcast_to(pos_mean[None, :], (F.shape[0], N)),
        np.broadcast_to(thr[None, :], (F.shape[0], N)),
    ], axis=-1)
    return np.concatenate([F, extra], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["MEL"])
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()
    h = args.horizon

    for city in args.cities:
        y0, t0 = pb.load(city)
        ok, trm, vam, tem, u, span = pb.split_rows(y0, t0)
        cols, smean = pb.live_cols(y0, trm)
        y = y0[:, cols]
        N = y.shape[1]
        thr = peak_thr(y, trm, N)
        tr_all_rows = np.where(trm & np.isfinite(y).all(1))[0]
        tr_all_rows = tr_all_rows[tr_all_rows >= h + 72]
        te_rows = np.where(tem & np.isfinite(y).all(1))[0]
        te_rows = te_rows[te_rows >= h + 72]
        rng = np.random.default_rng(0)
        tr_all_rows = rng.choice(tr_all_rows, min(MAX_ALL // N, len(tr_all_rows)),
                                 replace=False)
        te_rows = rng.choice(te_rows, min(MAX_ALL // N, len(te_rows)), replace=False)

        pos_cells = y[tr_all_rows] > 0
        pos_mean = np.array([
            y[tr_all_rows][pos_cells[:, s], s].mean() if pos_cells[:, s].any() else 0.0
            for s in range(N)], np.float64)
        thr_f = np.where(np.isfinite(thr), thr, pos_mean)
        tr_on = np.nanmean((y[tr_all_rows] > 0).astype(np.float32), 0)
        sm = np.nanmean(y[tr_all_rows], 0)

        Ftr = add_static(build_features(y, t0, tr_all_rows, sm, tr_on, h),
                         pos_mean, thr_f)
        Fte = add_static(build_features(y, t0, te_rows, sm, tr_on, h),
                         pos_mean, thr_f)
        Xall, _, _ = flatten(Ftr, y[tr_all_rows], tr_all_rows)
        yall = y[tr_all_rows].reshape(-1)
        Xte, _, ste = flatten(Fte, y[te_rows], te_rows)
        yte = y[te_rows].reshape(-1)
        keep = np.isfinite(yall) & np.isfinite(Xall).all(1)
        Xall, yall = Xall[keep], yall[keep]
        keep = np.isfinite(yte) & np.isfinite(Xte).all(1)
        Xte, yte, ste = Xte[keep], yte[keep], ste[keep]

        mpos = yall > 0
        Xp, yp = Xall[mpos], yall[mpos]
        if len(yp) > MAX_POS:
            i = rng.choice(len(yp), MAX_POS, replace=False)
            Xp, yp = Xp[i], yp[i]
        i = rng.choice(len(yall), min(MAX_ALL, len(yall)), replace=False)
        Xa, ya = Xall[i], yall[i]

        sc = StandardScaler().fit(Xp)
        lin = LinearRegression().fit(sc.transform(Xp), yp)
        gbm = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1,
                                            max_leaf_nodes=31).fit(Xp, yp)
        gbm_l = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1,
                                              max_leaf_nodes=31).fit(Xp, np.log1p(yp))
        gbm_a = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1,
                                              max_leaf_nodes=31).fit(Xa, ya)
        preds = {
            "const": pos_mean[ste],
            "lin_pos": lin.predict(sc.transform(Xte)),
            "gbm_pos": gbm.predict(Xte),
            "gbm_log": np.expm1(gbm_l.predict(Xte)),
            "gbm_all": gbm_a.predict(Xte),
        }
        pk = np.zeros(len(yte), bool)
        okthr = np.isfinite(thr)
        pk[okthr[ste]] = yte[okthr[ste]] > thr[ste[okthr[ste]]]
        masks = {"真正": yte > 0, "峰格": pk}
        print(f"\n===== {city}  训练正格 {int(mpos.sum())}  "
              f"测试正格 {int((yte > 0).sum())}  峰格 {int(pk.sum())} =====", flush=True)
        print(f"{'口径':<10}{'MAE正':>9}{'R2正':>8}{'bias正':>9}"
              f"{'MAE峰':>9}{'R2峰':>8}{'bias峰':>9}", flush=True)
        for nm, p in preds.items():
            p = np.maximum(p, 0)
            row = []
            for mm in masks.values():
                r2, mae = stat(p[mm], yte[mm])
                row += [mae, r2, np.mean(p[mm] - yte[mm])]
            print(f"{nm:<10}" + "".join(f"{v:>9.3f}" for v in row), flush=True)


if __name__ == "__main__":
    main()
