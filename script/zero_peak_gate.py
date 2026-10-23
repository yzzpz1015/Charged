# -*- coding: utf-8 -*-
"""Z0 + P0 的快速 gating 探针（CPU，不训练时序模型）。

Z0：P(y>0) 到底可不可预测
  * 只用**因果**特征：目标时刻前 1/2/3/24/72h 的值、前 24/72h 活跃率、
    小时/星期周期、站点训练期均值与活跃率。
  * 训练段拟合，测试段评估；报 pooled AUC 与 **within-station AUC**。
  * static 基线（只给站身份 + 时间）与 full 的差 = 滞后历史的增量。

P0：峰高可不可预测
  * 峰格 = 测试格 y ≥ 该站**训练期**正值 p95。
  * 常数（逐站训练峰均值）/ 线性 / GBM，报测试峰格 R²。
  * 同时报 log1p(y) 口径，避免大站主导。

用法::

    python script/zero_peak_gate.py --cities LOA JHB MEL SPO SZH
"""
import argparse
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import prob_baseline as pb                                          # noqa: E402

MAX_CELLS = 400_000
MAX_PEAK = 300_000


def _roll_stats(y, w):
    """前 w 行（不含当前行）的活跃率与均值，NaN 跳过。"""
    valid = np.isfinite(y)
    pos = valid & (y > 0)
    cv = np.cumsum(valid, 0)
    cp = np.cumsum(pos, 0)
    cy = np.cumsum(np.nan_to_num(y), 0)
    T = len(y)
    on = np.full((T, y.shape[1]), np.nan, np.float32)
    mu = np.full((T, y.shape[1]), np.nan, np.float32)
    if T > w:
        z = np.zeros((1, y.shape[1]), cv.dtype)
        dv = cv[w - 1:T - 1] - np.vstack((z, cv[:T - w - 1]))
        dp = cp[w - 1:T - 1] - np.vstack((z, cp[:T - w - 1]))
        dy = cy[w - 1:T - 1] - np.vstack((z, cy[:T - w - 1]))
        on[w:] = dp / np.maximum(dv, 1)
        mu[w:] = dy / np.maximum(dv, 1)
    return on, mu


def build_features(y, t0, rows, smean, train_on, h):
    """[len(rows), N, F] 的因果特征 + 站点静态特征。

    **所有特征截断在预测原点 t-h**：预测 H=h 时，最近可观测的是 y[t-h]，
    lag 1/2/3 属于预测期内，不能用（上一版探针就是在这里泄漏的）。
    滚动活动率取原点前 24/72h，即目标行 t 的 ``t-h+1`` 处。
    """
    T, N = y.shape
    lag = {}
    for k in (h, h + 1, h + 2, h + 24, h + 72):
        v = np.full_like(y, np.nan)
        v[k:] = y[:-k]
        lag[k] = v
    on24, mu24 = _roll_stats(y, 24)
    on72, mu72 = _roll_stats(y, 72)
    hod = t0.hour.values
    dow = t0.dayofweek.values
    feats = []
    rr = rows
    for k in (h, h + 1, h + 2, h + 24, h + 72):
        feats.append(np.nan_to_num(np.log1p(np.maximum(lag[k][rr], 0)), nan=0.0))
    io = np.maximum(rr - h + 1, 0)       # 原点前一日/三日的滚动统计
    for a in (on24, on72, mu24, mu72):
        feats.append(np.nan_to_num(a[io], nan=0.0))
    feats.append(np.broadcast_to(np.sin(2 * np.pi * hod[rr] / 24)[:, None], (len(rr), N)))
    feats.append(np.broadcast_to(np.cos(2 * np.pi * hod[rr] / 24)[:, None], (len(rr), N)))
    feats.append(np.broadcast_to(np.sin(2 * np.pi * dow[rr] / 7)[:, None], (len(rr), N)))
    feats.append(np.broadcast_to(np.cos(2 * np.pi * dow[rr] / 7)[:, None], (len(rr), N)))
    feats.append(np.broadcast_to(smean[None, :], (len(rr), N)))
    feats.append(np.broadcast_to(train_on[None, :], (len(rr), N)))
    return np.stack(feats, -1).astype(np.float32)


def flatten(X, Y, rows):
    """[M,N,F], [M,N] -> [M*N,F], [M*N], station id。"""
    N = X.shape[1]
    return (X.reshape(-1, X.shape[-1]), Y.reshape(-1),
            np.tile(np.arange(N), len(rows)))


def per_station_auc(y, p, st):
    vals, wts = [], []
    for s in np.unique(st):
        m = st == s
        if len(np.unique(y[m])) < 2:
            continue
        vals.append(roc_auc_score(y[m], p[m]))
        wts.append(m.sum())
    if not vals:
        return np.nan, np.nan
    vals = np.array(vals)
    return float(np.average(vals, weights=wts)), float(np.median(vals))


def fit_eval(Xtr, ytr, Xte, yte, kind="lr"):
    sc = StandardScaler().fit(Xtr)
    if kind == "lr":
        m = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs")
    else:
        m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1,
                                           max_leaf_nodes=31)
    m.fit(sc.transform(Xtr), ytr)
    p = m.predict_proba(sc.transform(Xte))[:, 1]
    return p


def z0(city, y, t0, trm, tem, h):
    tr_rows = np.where(trm)[0]
    te_rows = np.where(tem)[0]
    tr_rows = tr_rows[tr_rows >= h + 72]
    te_rows = te_rows[te_rows >= h + 72]
    rng = np.random.default_rng(0)
    tr_rows = tr_rows[rng.choice(len(tr_rows), min(MAX_CELLS // y.shape[1], len(tr_rows)),
                                 replace=False)]
    te_rows = te_rows[rng.choice(len(te_rows), min(MAX_CELLS // y.shape[1], len(te_rows)),
                                 replace=False)]
    tr_on = np.nanmean((y[tr_rows] > 0).astype(np.float32), 0)
    smean = np.nanmean(y[tr_rows], 0)
    Ftr = build_features(y, t0, tr_rows, smean, tr_on, h)
    Fte = build_features(y, t0, te_rows, smean, tr_on, h)
    Xtr, Ytr, _ = flatten(Ftr, (y[tr_rows] > 0).astype(np.int8), tr_rows)
    Xte, Yte, ste = flatten(Fte, (y[te_rows] > 0).astype(np.int8), te_rows)
    # static：只保留小时/星期 + 站点静态（最后 6 列）
    Xtr_s, Xte_s = Xtr[:, -6:], Xte[:, -6:]
    out = {"base": float(Yte.mean())}
    for nm, a, b in (("static", Xtr_s, Xte_s), ("full", Xtr, Xte)):
        p = fit_eval(a, Ytr, b, Yte)
        out[nm] = (roc_auc_score(Yte, p),) + per_station_auc(Yte, p, ste)
        if nm == "full":
            pred = p >= 0.5
            tp = np.sum(pred & (Yte == 1)); fp = np.sum(pred & (Yte == 0))
            fn = np.sum(~pred & (Yte == 1))
            out["prec"] = tp / max(1, tp + fp)
            out["rec"] = tp / max(1, tp + fn)
    # GBM（只跑 full，验证非线性有没有增量）
    pg = fit_eval(Xtr, Ytr, Xte, Yte, kind="gbm")
    out["gbm"] = (roc_auc_score(Yte, pg),) + per_station_auc(Yte, pg, ste)
    return out


def p0_simple(city, y, t0, trm, tem, h):
    """峰高可预测性：逐站常数 / 全局线性 / 全局 GBM。"""
    tr_rows = np.where(trm)[0]
    te_rows = np.where(tem)[0]
    tr_rows = tr_rows[tr_rows >= h + 72]
    te_rows = te_rows[te_rows >= h + 72]
    N = y.shape[1]
    thr = np.full(N, np.nan)
    for j in range(N):
        v = y[tr_rows, j]
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) >= 50:
            thr[j] = np.quantile(v, 0.95)
    ok = np.isfinite(thr)
    tr_peak = np.isfinite(y[tr_rows]) & (y[tr_rows] >= thr[None, :]) & ok[None, :]
    te_peak = np.isfinite(y[te_rows]) & (y[te_rows] >= thr[None, :]) & ok[None, :]
    if tr_peak.sum() < 100 or te_peak.sum() < 100:
        return None
    cst = np.full(N, np.nan)
    for j in range(N):
        v = y[tr_rows, j][tr_peak[:, j]]
        if len(v) >= 10:
            cst[j] = v.mean()
    mte = te_peak & np.isfinite(cst[None, :])
    yte = y[te_rows][mte]
    ste = np.tile(np.arange(N), len(te_rows)).reshape(len(te_rows), N)[mte]
    const = cst[ste]
    tr_on = np.nanmean((y[tr_rows] > 0).astype(np.float32), 0)
    smean = np.nanmean(y[tr_rows], 0)
    Ftr = build_features(y, t0, tr_rows, smean, tr_on, h)
    Fte = build_features(y, t0, te_rows, smean, tr_on, h)
    mtr = tr_peak.reshape(-1)
    mtef = mte.reshape(-1)
    Xtr, ytr = Ftr.reshape(-1, Ftr.shape[-1])[mtr], y[tr_rows].reshape(-1)[mtr]
    Xte, yte2 = Fte.reshape(-1, Fte.shape[-1])[mtef], y[te_rows].reshape(-1)[mtef]
    rng = np.random.default_rng(0)
    if len(ytr) > MAX_PEAK:
        i = rng.choice(len(ytr), MAX_PEAK, replace=False); Xtr, ytr = Xtr[i], ytr[i]
    if len(yte2) > MAX_PEAK:
        i = rng.choice(len(yte2), MAX_PEAK, replace=False)
        Xte, yte2, const = Xte[i], yte2[i], const[i]
    sc = StandardScaler().fit(Xtr)
    lin = LinearRegression().fit(sc.transform(Xtr), ytr)
    gbm = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1,
                                        max_leaf_nodes=31).fit(sc.transform(Xtr), ytr)
    out = {
        "n_tr": int(mtr.sum()), "n_te": int(mtef.sum()),
        "r2_const_raw": float(r2_score(yte2, const)),
        "r2_lin_raw": float(r2_score(yte2, lin.predict(sc.transform(Xte)))),
        "r2_gbm_raw": float(r2_score(yte2, gbm.predict(sc.transform(Xte)))),
    }
    yl, cl = np.log1p(yte2), np.log1p(const)
    out["r2_const_log"] = float(r2_score(yl, cl))
    out["r2_lin_log"] = float(r2_score(yl, np.log1p(np.maximum(lin.predict(sc.transform(Xte)), 0))))
    out["r2_gbm_log"] = float(r2_score(yl, np.log1p(np.maximum(gbm.predict(sc.transform(Xte)), 0))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["MEL"])
    ap.add_argument("--horizon", type=int, default=24,
                    help="提前量 H；所有特征截断在预测原点 t-H")
    args = ap.parse_args()
    h = args.horizon
    print(f"H={h}（特征截断在原点 t-{h}）")
    print(f"{'城':<5}{'base':>7}{'AUCstatic':>10}{'AUCfull':>9}{'AUCgbm':>8}"
          f"{'wAUCfull':>10}{'wAUCgbm':>9}{'prec':>7}{'rec':>7}", flush=True)
    for city in args.cities:
        y0, t0 = pb.load(city)
        ok, trm, vam, tem, u, span = pb.split_rows(y0, t0)
        cols, smean = pb.live_cols(y0, trm)
        y = y0[:, cols]
        z = z0(city, y, t0, trm, tem, h)
        print(f"{city:<5}{z['base']:>7.3f}{z['static'][0]:>10.3f}{z['full'][0]:>9.3f}"
              f"{z['gbm'][0]:>8.3f}{z['full'][1]:>10.3f}{z['gbm'][1]:>9.3f}"
              f"{z['prec']:>7.3f}{z['rec']:>7.3f}", flush=True)
        p = p0_simple(city, y, t0, trm, tem, h)
        if p:
            print(f"  P0 peak: n_tr={p['n_tr']} n_te={p['n_te']}  "
                  f"R2 raw const/lin/gbm = {p['r2_const_raw']:.3f}/"
                  f"{p['r2_lin_raw']:.3f}/{p['r2_gbm_raw']:.3f}  "
                  f"log = {p['r2_const_log']:.3f}/{p['r2_lin_log']:.3f}/"
                  f"{p['r2_gbm_log']:.3f}", flush=True)


if __name__ == "__main__":
    main()
