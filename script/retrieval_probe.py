# -*- coding: utf-8 -*-
"""Self-history retrieval probe (kNN on lag windows).

Question (user): many works retrieve similar windows from the target's OWN
history and get gains — does historical charging-volume retrieval
(similarity) add forecast value beyond what the same history already gives
through an autoregressive / full-window model?

Design:
- Library = training-period windows [t-L+1..t] with future block [t+1..t+h].
  Similarity = Euclidean on z-normalized window SHAPE (standard TS kNN).
- Query = test-period window; retrieve k nearest; forecast = mean of their
  z-normalized futures, rescaled by the query window's mean/std (raw space).
- Baselines on the SAME target: persistence, AR(3), full-72-lag ridge.
- Increment: OOS R^2 gain of adding the kNN forecast as a feature to AR(3)
  and to full-72-lag ridge.  Run on RAW (how good is retrieval alone) and on
  CYCLE-ADJUSTED (does episodic shape-recurrence add anything beyond the
  periodic schedule + short history — the decisive increment).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE = "/home/data/yzzpz/Charged/data"
CITIES = ["MEL", "SZH"]
SITE_CAP = {"MEL": 62, "SZH": 200}
L = 72
H = 24
KS = [5, 20]
SPLIT_SERIES = 0.7
RIDGE_ALPHA = 1.0
LAGS72 = [1, 2, 3, 6, 12, 24, 48, 72]   # sparse lags: stable full-history control


def load(city):
    vol = pd.read_csv(f"{BASE}/{city}/volume.csv", index_col=0, parse_dates=True).astype(float)
    vol.columns = vol.columns.astype(int)
    cols = [c for c in vol.columns if vol[c].std() > 0]
    return vol[cols]


def cycle_adjust(vol):
    return vol - vol.groupby(vol.index.hour).transform("mean")


def _windows(s, n):
    """[T,L] z-norm windows, [T,H] z-norm futures, raw per-window mu/sd."""
    T = n - L - H + 1
    X = np.empty((T, L))
    Y = np.empty((T, H))
    for i in range(T):
        X[i] = s[i:i + L]
        Y[i] = s[i + L:i + L + H]
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd, (Y - mu) / sd, mu[:, 0], sd[:, 0]


def knn_future(Xn, Yn, mu, sd, train_idx, query_idx, k):
    """Raw-space h-step forecast [Q,H] for the query windows."""
    Xq = Xn[query_idx]
    mu_q, sd_q = mu[query_idx], sd[query_idx]
    dots = Xq @ Xn[train_idx].T          # z-norm Euclidean^2 = 2L - 2*dot
    nn = np.argsort(-dots, axis=1)[:, :k]
    f = np.mean(Yn[train_idx][nn], axis=1)   # [Q,H] normalized future
    return f * sd_q[:, None] + mu_q[:, None]


def r2(y_true, y_hat):
    ss_res = float(((y_true - y_hat) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def ridge_score(Xtr, ytr, Xte, yte, alpha=RIDGE_ALPHA):
    """Closed-form ridge OOS R^2 with feature standardization (alpha on
    unit-variance scale, so the highly-collinear 72 lags are handled)."""
    m = Xtr.mean(0, keepdims=True)
    s = Xtr.std(0, keepdims=True)
    s[s < 1e-9] = 1.0
    Xtr = (Xtr - m) / s
    Xte = (Xte - m) / s
    ymu = ytr.mean()                 # intercept (fit on centered y)
    ytr_c = ytr - ymu
    XtX = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    w = np.linalg.solve(XtX, Xtr.T @ ytr_c)
    return r2(yte, Xte @ w + ymu)


def run_city(city, rng):
    vol = load(city)
    cols = list(vol.columns)
    if len(cols) > SITE_CAP[city]:
        cols = [cols[i] for i in rng.choice(len(cols), SITE_CAP[city], replace=False)]
    n = len(vol)
    split = int(SPLIT_SERIES * n)
    train_idx = np.arange(0, split - L - H + 1)
    query_idx = np.arange(split, n - L - H + 1)
    sq = int(0.7 * len(query_idx))
    print(f"\n== {city} (n_sites={len(cols)}) ==")
    base = cycle_adjust(vol)   # cycle-adjusted only: isolates episodic signal beyond the schedule
    agg = {k: [] for k in ["knnalone1", "knnalone6", "knnalone24", "persist1", "ar1", "ar6", "ar24", "hour6", "hour24",
                            "inc_ar1", "inc_hour1", "inc_ar6", "inc_hour6", "inc_ar24", "inc_hour24"]}
    for s_col in cols:
        s = base[s_col].values.astype(float)
        Xn, Yn, mu, sd = _windows(s, n)
        tq = query_idx + L - 1              # last observed step of each query
        y1 = s[tq + 1]
        y6 = s[tq + 6]
        y24 = s[tq + 24]
        Xar = np.stack([s[tq], s[tq - 1], s[tq - 2]], axis=1)
        # hour-aware control: AR3 + same-hour-yesterday deviation relative to the target
        Xhr1 = np.c_[Xar, s[tq + 1 - 24]]
        Xhr6 = np.c_[Xar, s[tq + 6 - 24]]
        Xhr24 = np.c_[Xar, s[tq + 24 - 24]]   # = s[tq] (last observed, persistence-ish)
        # controls
        r_ar1 = LinearRegression().fit(Xar[:sq], y1[:sq]).score(Xar[sq:], y1[sq:])
        r_ar6 = LinearRegression().fit(Xar[:sq], y6[:sq]).score(Xar[sq:], y6[sq:])
        r_ar24 = LinearRegression().fit(Xar[:sq], y24[:sq]).score(Xar[sq:], y24[sq:])
        r_hr1 = LinearRegression().fit(Xhr1[:sq], y1[:sq]).score(Xhr1[sq:], y1[sq:])
        r_hr6 = LinearRegression().fit(Xhr6[:sq], y6[:sq]).score(Xhr6[sq:], y6[sq:])
        r_hr24 = LinearRegression().fit(Xhr24[:sq], y24[:sq]).score(Xhr24[sq:], y24[sq:])
        agg["ar1"].append(r_ar1)
        agg["ar6"].append(r_ar6)
        agg["ar24"].append(r_ar24)
        agg["hour6"].append(r_hr6)
        agg["hour24"].append(r_hr24)
        for k in KS:
            F = knn_future(Xn, Yn, mu, sd, train_idx, query_idx, k)
            f1, f6, f24 = F[:, 0], F[:, 5], F[:, 23]
            if k == KS[0]:
                agg["knnalone1"].append(r2(y1, f1))
                agg["knnalone6"].append(r2(y6, f6))
                agg["knnalone24"].append(r2(y24, f24))
                agg["persist1"].append(r2(y1, s[tq]))
            c = LinearRegression().fit(np.c_[Xar[:sq], f1[:sq]], y1[:sq])
            agg["inc_ar1"].append(c.score(np.c_[Xar[sq:], f1[sq:]], y1[sq:]) - r_ar1)
            c = LinearRegression().fit(np.c_[Xhr1[:sq], f1[:sq]], y1[:sq])
            agg["inc_hour1"].append(c.score(np.c_[Xhr1[sq:], f1[sq:]], y1[sq:]) - r_hr1)
            c = LinearRegression().fit(np.c_[Xar[:sq], f6[:sq]], y6[:sq])
            agg["inc_ar6"].append(c.score(np.c_[Xar[sq:], f6[sq:]], y6[sq:]) - r_ar6)
            c = LinearRegression().fit(np.c_[Xhr6[:sq], f6[:sq]], y6[:sq])
            agg["inc_hour6"].append(c.score(np.c_[Xhr6[sq:], f6[sq:]], y6[sq:]) - r_hr6)
            c = LinearRegression().fit(np.c_[Xar[:sq], f24[:sq]], y24[:sq])
            agg["inc_ar24"].append(c.score(np.c_[Xar[sq:], f24[sq:]], y24[sq:]) - r_ar24)
            c = LinearRegression().fit(np.c_[Xhr24[:sq], f24[:sq]], y24[:sq])
            agg["inc_hour24"].append(c.score(np.c_[Xhr24[sq:], f24[sq:]], y24[sq:]) - r_hr24)
    nm = lambda x: np.nanmean(agg[x])
    print(f"  [cycle] kNN-alone h1 {nm('knnalone1'):.4f} h6 {nm('knnalone6'):.4f} h24 {nm('knnalone24'):.4f}"
          f" | persist1 {nm('persist1'):.4f} | AR3 h1 {nm('ar1'):.4f} h6 {nm('ar6'):.4f} h24 {nm('ar24'):.4f}"
          f" | AR3+hour h6 {nm('hour6'):.4f} h24 {nm('hour24'):.4f}")
    print(f"          incr over AR3:       h1 {nm('inc_ar1'):+.4f} h6 {nm('inc_ar6'):+.4f} h24 {nm('inc_ar24'):+.4f}"
          f" | incr over AR3+hour: h1 {nm('inc_hour1'):+.4f} h6 {nm('inc_hour6'):+.4f} h24 {nm('inc_hour24'):+.4f}")


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    for c in CITIES:
        run_city(c, rng)
