# -*- coding: utf-8 -*-
"""High-utilization check: does the substitution/spillover mechanism appear
for busy stations?

For MEL/SZH/LOA: split sites by utilization (nonzero-hour fraction) into
terciles.  Per group report:
  - k=1 same-time cycleAdj correlation with nearest neighbor
  - OOS lead-lag increment: control(own lag-1) vs + [neighbor lag-1 value,
    neighbor-full indicator at t-1].  If spillover (A full -> B rises) exists,
    busy stations should show a positive increment.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from neighbor_probe_cities import load, cycle_adjust, safe_corr

CITIES = ["MEL", "SZH", "LOA"]


def run(city):
    vol, dst = load(city)
    cols = list(vol.columns)
    adj = cycle_adjust(vol)
    N = len(cols)
    util = (vol > 0).mean()                     # nonzero-hour fraction
    order = np.argsort(util.values)             # low -> high
    t3 = N // 3
    groups = {
        "low-util": order[:t3],
        "mid-util": order[t3:2 * t3],
        "high-util": order[2 * t3:],
    }
    dmat = dst.values.copy()
    np.fill_diagonal(dmat, np.inf)
    nn1 = np.argsort(dmat, axis=1)[:, :1]
    split = int(0.7 * len(vol))
    print(f"\n== {city} (N={N}) — by utilization tercile (nonzero-hour frac) ==")
    print(f"{'group':<11}{'n_sites':>8}{'util':>8}{'k1_pear':>9}{'k1_cycle':>10}{'incr_val':>10}{'incr_full':>10}")
    for gname, idx in groups.items():
        pc, ac, iv, ifull = [], [], [], []
        for i in idx:
            s = cols[i]
            n = cols[nn1[i][0]]
            pc.append(safe_corr(vol[s], vol[n]))
            ac.append(safe_corr(adj[s], adj[n]))
            y = adj[s].values[1:]
            own = adj[s].shift(1).values[1:]
            nbr = adj[n].shift(1).values[1:]
            nbr_full = (vol[n].shift(1).values >= vol[n].quantile(0.9)).astype(float)[1:]
            Xa = own.reshape(-1, 1)
            Xb = np.stack([own, nbr, nbr_full], axis=1)
            m = np.isfinite(Xb).all(1) & np.isfinite(y)
            y_, A_, B_ = y[m], Xa[m], Xb[m]
            if len(y_) < 500:
                continue
            la = LinearRegression().fit(A_[:split], y_[:split])
            lb = LinearRegression().fit(B_[:split], y_[:split])
            rA = la.score(A_[split:], y_[split:])
            iv.append(lb.score(B_[split:], y_[split:]) - rA)
            # full-indicator only
            Xf = np.stack([own, nbr_full], axis=1)[m]
            lf = LinearRegression().fit(Xf[:split], y_[:split])
            ifull.append(lf.score(Xf[split:], y_[split:]) - rA)
        print(f"{gname:<11}{len(idx):>8}{np.mean(util.values[idx]):>8.3f}"
              f"{np.nanmean(pc):>9.4f}{np.nanmean(ac):>10.4f}"
              f"{np.nanmean(iv):>10.4f}{np.nanmean(ifull):>10.4f}")


if __name__ == "__main__":
    for c in CITIES:
        run(c)
