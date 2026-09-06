# -*- coding: utf-8 -*-
"""Cross-city neighbor signal probe.

For each city (raw dirs): same-time spatial correlation, distance decay,
and the decisive forecast increment (neighbor[t-1] -> target[t], control
own lag-1).  AMS has no distance.csv -> haversine from sites.csv.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE = "/home/data/yzzpz/Charged/data"
CITIES = ["MEL", "JHB", "SPO", "LOA", "SZH", "AMS"]
SITE_CAP = 400  # cap sites used in the increment regression (big cities)


def load(city):
    vol = pd.read_csv(f"{BASE}/{city}/volume.csv", index_col=0, parse_dates=True).astype(float)
    vol.columns = vol.columns.astype(int)
    sites = pd.read_csv(f"{BASE}/{city}/sites.csv")
    try:
        dst = pd.read_csv(f"{BASE}/{city}/distance.csv", index_col=0)
        dst.index = dst.index.astype(int)
        dst.columns = dst.columns.astype(int)
    except FileNotFoundError:
        # haversine from sites.csv
        s = sites.set_index(sites.columns[0])
        s.index = s.index.astype(int)
        lon, lat = s.loc[vol.columns, "longitude"].values, s.loc[vol.columns, "latitude"].values
        R = 6371.0
        la1, lo1 = np.radians(lat)[:, None], np.radians(lon)[:, None]
        la2, lo2 = np.radians(lat)[None, :], np.radians(lon)[None, :]
        dlat, dlon = la2 - la1, lo2 - lo1
        a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
        dst = pd.DataFrame(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)),
                           index=vol.columns, columns=vol.columns)
    cols = [c for c in dst.columns if c in vol.columns]
    # drop dead/constant sites (std==0 -> correlation undefined)
    cols = [c for c in cols if vol[c].std() > 0]
    return vol[cols], dst.loc[cols, cols]


def cycle_adjust(vol):
    return vol - vol.groupby(vol.index.hour).transform("mean")


def safe_corr(a, b):
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0 or np.isnan(sa) or np.isnan(sb):
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def increment(vol, adj, dst, cols, k, rng=None):
    """OOS R^2 increment of neighbor[t-1] over control own lag-1, mean over sites."""
    N = len(cols)
    if rng is not None and N > SITE_CAP:
        idx = rng.choice(N, SITE_CAP, replace=False)
        cols = [cols[i] for i in idx]
        N = len(cols)
    dmat = dst.loc[cols, cols].values.copy()
    np.fill_diagonal(dmat, np.inf)
    nn = np.argsort(dmat, axis=1)[:, :k]
    split = int(0.7 * len(vol))
    incs = []
    for i in range(N):
        s = cols[i]
        y = adj[s].values[1:]
        own = adj[s].shift(1).values[1:]
        nbr = np.stack([adj[cols[j]].shift(1).values[1:] for j in nn[i]], axis=1)
        Xa = own.reshape(-1, 1)
        Xb = np.concatenate([Xa, nbr], axis=1)
        m = np.isfinite(Xb).all(1) & np.isfinite(y)
        y_, A_, B_ = y[m], Xa[m], Xb[m]
        if len(y_) < 500:
            continue
        la = LinearRegression().fit(A_[:split], y_[:split])
        lb = LinearRegression().fit(B_[:split], y_[:split])
        incs.append(lb.score(B_[split:], y_[split:]) - la.score(A_[split:], y_[split:]))
    return float(np.nanmean(incs)) if incs else np.nan


def main():
    rng = np.random.RandomState(0)
    print(f"{'city':<5}{'N':>5}{'zero%':>7}{'far_pcar':>9}{'far_cycle':>10}"
          f"{'k5_pear':>9}{'k5_cycle':>10}{'nn5_km':>8}{'incr_k1':>9}{'incr_k3':>9}")
    for city in CITIES:
        vol, dst = load(city)
        cols = list(vol.columns)
        N = len(cols)
        adj = cycle_adjust(vol)
        # far baseline (pairs > 40km), subsample pairs
        iu = np.triu_indices(N, k=1)
        d = dst.values[iu]
        m = d > 40
        idx = np.where(m)[0]
        if len(idx) > 0:
            sub = rng.choice(idx, min(1500, len(idx)), replace=False)
            pcar = np.nanmean([safe_corr(vol.iloc[:, iu[0][q]], vol.iloc[:, iu[1][q]]) for q in sub])
            pc_cy = np.nanmean([safe_corr(adj.iloc[:, iu[0][q]], adj.iloc[:, iu[1][q]]) for q in sub])
        else:
            pcar = pc_cy = np.nan
        # k=5 nearest same-time
        dmat = dst.values.copy()
        np.fill_diagonal(dmat, np.inf)
        nn5 = np.argsort(dmat, axis=1)[:, :5]
        pc, ac = [], []
        for i in range(N):
            s = cols[i]
            nbrs = [cols[j] for j in nn5[i]]
            pc.append(np.nanmean([safe_corr(vol[s], vol[n]) for n in nbrs]))
            ac.append(np.nanmean([safe_corr(adj[s], adj[n]) for n in nbrs]))
        nn5_km = np.median(dmat[np.arange(N), nn5[:, -1]])
        inc1 = increment(vol, adj, dst, cols, 1, rng)
        inc3 = increment(vol, adj, dst, cols, 3, rng)
        zero = 100 * (vol == 0).mean().mean()
        print(f"{city:<5}{N:>5}{zero:>7.1f}{pcar:>9.4f}{pc_cy:>10.4f}"
              f"{np.nanmean(pc):>9.4f}{np.nanmean(ac):>10.4f}{nn5_km:>8.1f}{inc1:>9.4f}{inc3:>9.4f}")


if __name__ == "__main__":
    main()
