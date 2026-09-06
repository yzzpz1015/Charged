# -*- coding: utf-8 -*-
"""Neighbor-station signal probe (data-level, no training).

Questions:
1. Does neighbor flow carry spatial signal beyond the common diurnal cycle?
2. How does correlation / MI decay with distance -> what radius is appropriate?
3. Does neighbor PAST help predict target FUTURE (lead-lag), net of the
   target's own autoregression and hour-of-day?

Zero values are genuine idle (established), so both raw and hour-of-day
adjusted numbers are reported.  Hour-of-day adjustment removes each site's
own mean at each hour -> isolates spatial structure net of the shared rhythm.
"""
import numpy as np
import pandas as pd

DATA = "/home/data/yzzpz/Charged/data/MEL_remove_zero"

def load():
    vol = pd.read_csv(f"{DATA}/volume.csv", index_col=0, parse_dates=True).astype(float)
    dst = pd.read_csv(f"{DATA}/distance.csv", index_col=0)
    dst.index = dst.index.astype(int)
    dst.columns = dst.columns.astype(int)
    vol.columns = vol.columns.astype(int)
    # align site columns
    cols = [c for c in dst.columns if c in vol.columns]
    vol = vol[cols].astype(float)
    dst = dst.loc[cols, cols].astype(float)
    return vol, dst

def cycle_adjust(vol):
    """Subtract each site's hour-of-day mean (keep the residual)."""
    hm = vol.groupby(vol.index.hour).transform("mean")
    return vol - hm

def safe_corr(a, b):
    """Pearson corr, NaN if either series is constant."""
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0 or np.isnan(sa) or np.isnan(sb):
        return np.nan
    return np.corrcoef(a, b)[0, 1]

def binned_mi(x, y, nbins=8, seed=0):
    """Fast binned MI (bits), quantile bins + Laplace smoothing."""
    rng = np.random.RandomState(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x[m]), np.asarray(y[m])
    n = len(x)
    if n > 8000:
        idx = rng.choice(n, 8000, replace=False)
        x, y = x[idx], y[idx]
    qx = pd.qcut(x, nbins, labels=False, duplicates="drop")
    qy = pd.qcut(y, nbins, labels=False, duplicates="drop")
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    if np.isnan(qx).any() or np.isnan(qy).any():
        return np.nan
    k = int(max(qx.max(), qy.max())) + 1
    H = np.histogram2d(qx, qy, bins=[k, k])[0] + 1.0  # Laplace
    pxy = H / H.sum()
    px = pxy.sum(1, keepdims=True)
    py = pxy.sum(0, keepdims=True)
    return float((pxy * np.log2(pxy / (px * py))).sum())

def partial_corr_signal(vol, cols):
    """Per-site incremental info of neighbors beyond own lag-1 and hour-of-day.

    For each site t: regress out hour-of-day mean and lag-1 residual, then
    correlate the target's innovation with the mean of its k-nearest
    neighbors' same-step innovation.  Returns mean |partial r|.
    """
    res = vol.groupby(vol.index.hour).transform("mean").rsub(vol)  # residual vs hour mean
    lag1 = res.shift(1)
    out = {}
    for s in cols:
        y = res[s] - lag1[s]           # innovation net of own AR(1)
        out[s] = y
    return pd.DataFrame(out)

def main():
    vol, dst = load()
    cols = list(vol.columns)
    T, N = vol.shape
    print(f"MEL: {N} sites x {T} hourly steps, distances {dst.values.min():.1f}..{dst.values.max():.0f} km")
    print(f"zero fraction: {100*(vol==0).mean().mean():.1f}%")

    adj = cycle_adjust(vol)
    rng = np.random.RandomState(0)

    # ---- pairwise stats, bucketed by distance ----
    iu = np.triu_indices(N, k=1)
    i_src, i_tgt = iu
    dist_ij = dst.values[iu]
    raw_p = np.array([safe_corr(vol.iloc[:, a], vol.iloc[:, b]) for a, b in zip(i_src, i_tgt)])
    adj_p = np.array([safe_corr(adj.iloc[:, a], adj.iloc[:, b]) for a, b in zip(i_src, i_tgt)])
    spr = np.array([vol.iloc[:, a].corr(vol.iloc[:, b], method="spearman") for a, b in zip(i_src, i_tgt)])
    mi = np.array([binned_mi(vol.iloc[:, a].values, vol.iloc[:, b].values) for a, b in zip(i_src, i_tgt)])

    bins = [0, 2, 5, 10, 20, 40, np.inf]
    lbl = ["0-2", "2-5", "5-10", "10-20", "20-40", "40+"]
    print("\n== correlation / MI vs distance (all pairs) ==")
    print(f"{'dist(km)':<9}{'pairs':>6}{'mean_pear':>10}{'mean_spear':>11}{'pear_cycleAdj':>14}{'MI_bits':>9}")
    for b0, b1, L in zip(bins[:-1], bins[1:], lbl):
        m = (dist_ij >= b0) & (dist_ij < b1)
        print(f"{L:<9}{m.sum():>6}{np.nanmean(raw_p[m]):>10.4f}{np.nanmean(spr[m]):>11.4f}{np.nanmean(adj_p[m]):>14.4f}{np.nanmean(mi[m]):>9.4f}")

    # far baseline = pairs > 40km
    far = dist_ij > 40
    print(f"\nfar baseline (>40km): pear {np.nanmean(raw_p[far]):.4f} | spearman {np.nanmean(spr[far]):.4f} | cycleAdj {np.nanmean(adj_p[far]):.4f} | MI {np.nanmean(mi[far]):.4f}")

    # ---- radius recommendation: mean corr within radius ----
    print("\n== mean signal vs radius (within-radius pairs) ==")
    print(f"{'radius(km)':<11}{'n_pairs':>8}{'pear':>8}{'cycleAdj':>10}{'MI':>7}")
    for r in [1, 2, 3, 5, 7, 10, 15, 20]:
        m = dist_ij <= r
        print(f"{r:<11}{m.sum():>8}{np.nanmean(raw_p[m]):>8.4f}{np.nanmean(adj_p[m]):>10.4f}{np.nanmean(mi[m]):>7.4f}")

    # ---- nearest-k aggregate (the usable signal) ----
    print("\n== mean correlation of target with its k-nearest neighbors (per site, then averaged) ==")
    kvals = [3, 5, 8, 12]
    dmat = dst.values.copy()
    np.fill_diagonal(dmat, np.inf)
    for k in kvals:
        nn = np.argsort(dmat, axis=1)[:, :k]
        pc, ac = [], []
        for i in range(N):
            s = cols[i]
            nbrs = [cols[j] for j in nn[i]]
            pc.append(np.nanmean([safe_corr(vol[s], vol[n]) for n in nbrs]))
            ac.append(np.nanmean([safe_corr(adj[s], adj[n]) for n in nbrs]))
        print(f"k={k:<3} mean pear {np.nanmean(pc):.4f} | mean cycleAdj {np.nanmean(ac):.4f} | median nn-dist "
              f"{np.median(dmat[np.arange(N), nn[:, -1]]):.1f} km")

    # ---- lead-lag: does neighbor past help target future? ----
    print("\n== lead-lag (k=5 nearest): corr(mean_nbr[t-h], target[t]) after cycle adjust ==")
    dmat2 = dst.values.copy()
    np.fill_diagonal(dmat2, np.inf)
    nn5 = np.argsort(dmat2, axis=1)[:, :5]
    adj = cycle_adjust(vol)
    for h in [0, 1, 6, 12, 24, 48, 72]:
        rs = []
        for i in range(N):
            s = cols[i]
            nbr_mean = adj[[cols[j] for j in nn5[i]]].shift(h).mean(axis=1)
            rs.append(safe_corr(nbr_mean.dropna(), adj[s].loc[nbr_mean.dropna().index]))
        print(f"h={h:<3} mean r {np.nanmean(rs):.4f}")

    # ---- beyond own AR(1)+hour: partial signal ----
    print("\n== partial signal: corr(target innovation vs mean of k=5 nbrs' innovation), cycle+AR1 removed ==")
    res = adj  # cycle-adjusted already
    lag = res.shift(1)
    innov = res - lag
    dmat3 = dst.values.copy()
    np.fill_diagonal(dmat3, np.inf)
    nn5b = np.argsort(dmat3, axis=1)[:, :5]
    prs = []
    for i in range(N):
        s = cols[i]
        nbr_innov = innov[[cols[j] for j in nn5b[i]]].mean(axis=1)
        prs.append(safe_corr(nbr_innov.dropna(), innov[s].loc[nbr_innov.dropna().index]))
    print(f"mean partial r {np.nanmean(prs):.4f} (0.0 = neighbor adds nothing beyond own past + diurnal)")

if __name__ == "__main__":
    main()
