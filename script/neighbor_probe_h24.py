# -*- coding: utf-8 -*-
"""Neighbor lead-lag probe at horizon 24.

Does the 5 nearest neighbors' flow at time t add OOS forecast value for the
target at time t+24, beyond the target's OWN history (lags 1..48 sparse)?
Mirrors neighbor_probe2 section C, but the forecast target is t+24 (24h ahead)
instead of 1-step. Cycle-adjusted so the shared diurnal cycle isn't counted.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

DATA = "/home/data/yzzpz/Charged/data/MEL_remove_zero"
LAGS = [1, 2, 3, 24, 48]     # sparse own-history control lags


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


def main():
    vol, dst = load()
    cols = list(vol.columns)
    N = len(cols)
    adj = cycle_adjust(vol)
    dmat = dst.values.copy()
    np.fill_diagonal(dmat, np.inf)
    split = int(0.7 * len(vol))

    for k in [1, 3, 5]:
        nni = np.argsort(dmat, axis=1)[:, :k]
        inc_ridge, inc_ols = [], []
        for i in range(N):
            s = cols[i]
            y = adj[s].values[24:]                      # target at t+24
            own = np.stack([adj[s].shift(l).values[24:] for l in LAGS], axis=1)
            nbr = np.stack([adj[cols[j]].values[24:] for j in nni[i]], axis=1)  # nbr at t (=lag-24)
            Xa = own
            Xb = np.concatenate([own, nbr], axis=1)
            m = np.isfinite(Xb).all(1) & np.isfinite(y)
            y_, Xa_, Xb_ = y[m], Xa[m], Xb[m]
            la = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(Xa_[:split], y_[:split])
            lb = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(Xb_[:split], y_[:split])
            inc_ridge.append(lb.score(Xb_[split:], y_[split:]) - la.score(Xa_[split:], y_[split:]))
            inc_ols.append(lb.score(Xb_[split:], y_[split:]) - la.score(Xa_[split:], y_[split:]))
        print(f"k={k} horizon=24: neighbor increment OOS R2 = {np.nanmean(inc_ridge):+.4f}  "
              f"(n_pts={int((~np.isfinite(dmat)).sum()) if False else N} sites)")


if __name__ == "__main__":
    main()
