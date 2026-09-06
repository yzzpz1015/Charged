# -*- coding: utf-8 -*-
"""Neighbor relationship probe #2 — cheap complementary checks.

Probes (all data-level, no training):
  A. Regime conditioning: is spatial correlation concentrated in active
     hours / both-active timestamps? (zero-inflation dilutes the raw numbers)
  B. Lagged MI: temporal structure of the information transfer (nonlinear),
     net of the shared diurnal cycle.
  C. Joint forecast increment: target_adj[t] ~ target_adj[t-1] (control) vs
     + 5 nearest neighbors at t-1 (control+neighbors), OOS R^2 increment.
     Answers "neighbor past beyond own past", in forecasting form, joint.
  D. Nonlinearity: distance correlation (dCor, Székely) vs |Pearson| on
     cycle-adjusted values for k=5 nearest pairs.  dCor >> |pear| -> the
     relationship is nonlinear (binned-MI edge is real).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA = "/home/data/yzzpz/Charged/data/MEL_remove_zero"


def load():
    vol = pd.read_csv(f"{DATA}/volume.csv", index_col=0, parse_dates=True).astype(float)
    dst = pd.read_csv(f"{DATA}/distance.csv", index_col=0)
    dst.index = dst.index.astype(int)
    dst.columns = dst.columns.astype(int)
    vol.columns = vol.columns.astype(int)
    cols = [c for c in dst.columns if c in vol.columns]
    return vol[cols], dst.loc[cols, cols]


def cycle_adjust(vol):
    return vol - vol.groupby(vol.index.hour).transform("mean")


def safe_corr(a, b):
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0 or np.isnan(sa) or np.isnan(sb):
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def binned_mi(x, y, nbins=8, seed=0, nmax=8000):
    rng = np.random.RandomState(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[m], float), np.asarray(y[m], float)
    if len(x) > nmax:
        idx = rng.choice(len(x), nmax, replace=False)
        x, y = x[idx], y[idx]
    qx = np.asarray(pd.qcut(x, nbins, labels=False, duplicates="drop"), dtype=float)
    qy = np.asarray(pd.qcut(y, nbins, labels=False, duplicates="drop"), dtype=float)
    if np.isnan(qx).any() or np.isnan(qy).any():
        return np.nan
    k = int(max(qx.max(), qy.max())) + 1
    H = np.histogram2d(qx, qy, bins=[k, k])[0] + 1.0
    pxy = H / H.sum()
    px = pxy.sum(1, keepdims=True)
    py = pxy.sum(0, keepdims=True)
    return float((pxy * np.log2(pxy / (px * py))).sum())


def dcor(x, y, nmax=800, seed=0):
    """Distance correlation (Székely & Rizzo), subsampled."""
    rng = np.random.RandomState(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[m], float), np.asarray(y[m], float)
    if len(x) > nmax:
        idx = rng.choice(len(x), nmax, replace=False)
        x, y = x[idx], y[idx]
    x = x - x.mean()
    y = y - y.mean()
    A = np.abs(x[:, None] - x[None, :])
    B = np.abs(y[:, None] - y[None, :])
    A = A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()
    B = B - B.mean(0, keepdims=True) - B.mean(1, keepdims=True) + B.mean()
    dcov2 = float((A * B).mean())
    dvarx2 = float((A * A).mean())
    dvary2 = float((B * B).mean())
    denom = np.sqrt(dvarx2 * dvary2)
    return dcov2 / denom if denom > 0 else np.nan


def main():
    vol, dst = load()
    cols = list(vol.columns)
    N = len(cols)
    adj = cycle_adjust(vol)
    dmat = dst.values.copy()
    np.fill_diagonal(dmat, np.inf)
    nn5 = np.argsort(dmat, axis=1)[:, :5]
    hour = vol.index.hour.values

    # ---- A. regime conditioning ----
    print("== A. regime conditioning (k=5 nearest, pair-averaged) ==")
    print(f"{'regime':<26}{'n':>6}{'pear':>8}{'cycleAdj':>10}{'MI':>8}")
    for name, mask in [
        ("all hours", np.ones(len(vol), bool)),
        ("active window 07-23", (hour >= 7) & (hour <= 23)),
        ("both nonzero", None),  # per-pair
    ]:
        pcs, acs, mis = [], [], []
        for i in range(N):
            s = cols[i]
            for j in nn5[i]:
                n = cols[j]
                if mask is None:
                    m = (vol[s].values > 0) & (vol[n].values > 0)
                else:
                    m = mask
                if m.sum() < 200:
                    continue
                pcs.append(safe_corr(vol[s].values[m], vol[n].values[m]))
                acs.append(safe_corr(adj[s].values[m], adj[n].values[m]))
                mis.append(binned_mi(vol[s].values[m], vol[n].values[m]))
        print(f"{name:<26}{len(pcs):>6}{np.nanmean(pcs):>8.4f}{np.nanmean(acs):>10.4f}{np.nanmean(mis):>8.4f}")

    # ---- B. lagged MI (k=5 nearest, cycle-adjusted) ----
    print("\n== B. lagged MI: MI(mean_nbr[t-h], target[t]), cycle-adjusted ==")
    print(f"{'h':<4}{'MI_bits':>9}")
    for h in [0, 1, 6, 12, 24, 48]:
        mis = []
        for i in range(N):
            s = cols[i]
            nb = adj[[cols[j] for j in nn5[i]]].shift(h).mean(axis=1)
            mis.append(binned_mi(nb.values, adj[s].values))
        print(f"{h:<4}{np.nanmean(mis):>9.4f}")

    # ---- C. joint forecast increment (OOS), OLS vs Ridge, k=1/3/5, lag-1 vs same-time ----
    print("\n== C. forecast increment: target_adj[t] = control(own lag-1) [+ neighbors] (OOS R^2) ==")
    from sklearn.linear_model import RidgeCV
    split = int(0.7 * len(vol))
    for k in [1, 3, 5]:
        nni = np.argsort(dmat, axis=1)[:, :k]
        for lag, tag in [(1, "nbrs at t-1 (usable)"), (0, "nbrs at t (ceiling)")]:
            inc_o, inc_r, r2b_r = [], [], []
            for i in range(N):
                s = cols[i]
                y = adj[s].values[1:]
                own = adj[s].shift(1).values[1:]
                nbr = np.stack([adj[cols[j]].shift(lag).values[1:] for j in nni[i]], axis=1)
                Xa = own.reshape(-1, 1)
                Xb = np.concatenate([Xa, nbr], axis=1)
                m = np.isfinite(Xb).all(1) & np.isfinite(y)
                y_, Xa_, Xb_ = y[m], Xa[m], Xb[m]
                la = LinearRegression().fit(Xa_[:split], y_[:split])
                lb = LinearRegression().fit(Xb_[:split], y_[:split])
                rr = RidgeCV(alphas=[1e-2, 1e-1, 1, 10, 100]).fit(Xb_[:split], y_[:split])
                inc_o.append(lb.score(Xb_[split:], y_[split:]) - la.score(Xa_[split:], y_[split:]))
                r2b_r.append(rr.score(Xb_[split:], y_[split:]))
                inc_r.append(rr.score(Xb_[split:], y_[split:]) - la.score(Xa_[split:], y_[split:]))
            print(f"k={k} {tag:<20} incr_OLS {np.nanmean(inc_o):+.4f} | incr_Ridge {np.nanmean(inc_r):+.4f} | R2_Ridge {np.nanmean(r2b_r):.4f}")

    # ---- D. nonlinearity check (dCor vs |Pearson|) on k=5 nearest pairs ----
    print("\n== D. nonlinearity: distance corr vs |Pearson| (cycle-adjusted, k=5 nearest unique pairs) ==")
    seen = set()
    pc, dc = [], []
    for i in range(N):
        s = cols[i]
        for j in nn5[i]:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            n = cols[j]
            pc.append(abs(safe_corr(adj[s], adj[n])))
            dc.append(dcor(adj[s].values, adj[n].values))
    print(f"mean |Pearson| {np.nanmean(pc):.4f} | mean dCor {np.nanmean(dc):.4f} | ratio {np.nanmean(dc)/np.nanmean(pc):.2f}")
    print("(ratio ~1 => relationship is essentially linear; ratio >>1 => nonlinear structure)")


if __name__ == "__main__":
    main()
