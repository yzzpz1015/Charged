# -*- coding: utf-8 -*-
"""Cross-city satellite -> busy-ness probe.

Hypothesis: satellite predicts EV busy-ness only when stations span DIVERSE
types (charger count / land-use). MEL (62 sites, all 1 charger) gave ~0; LOA
(224 sites, charger_num 1..16) should give a positive signal if the hypothesis
holds.

For each city: DINOv2 CLS -> {charger_num (log), mean duration, mean volume,
duration-per-charger} via LOO ridge (PCA 15 + alpha grid). Also report
cross-site cosine (visual diversity) and charger/power spread.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

BASE = "/home/data/yzzpz/Charged"
CITIES = ["MEL", "JHB", "LOA"]
PCA_DIM = 15
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def pca(X, n):
    return X if X.shape[1] <= n else PCA(n_components=n).fit_transform(X)


def loo_ridge(X, y, alphas=ALPHAS):
    X = np.asarray(X, float); y = np.asarray(y, float)
    N = len(y); best_r2 = -1e18
    for a in alphas:
        r2s = []
        for i in range(N):
            tr = np.arange(N) != i
            m = X[tr].mean(0, keepdims=True); s = X[tr].std(0, keepdims=True); s[s < 1e-9] = 1.0
            r = Ridge(alpha=a).fit((X[tr] - m) / s, y[tr])
            r2s.append((y[i] - r.predict((X[i:i + 1] - m) / s)[0]) ** 2)
        ss = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float(np.sum(r2s)) / ss if ss > 0 else np.nan
        if r2 > best_r2:
            best_r2 = r2
    return best_r2


def cosine_spread(reps):
    C = reps @ reps.T
    np.fill_diagonal(C, np.nan)
    off = C[np.triu_indices(reps.shape[0], 1)]
    return np.nanmean(off)


def load_city(city):
    data = f"{BASE}/data/{city}_remove_zero"
    dur = pd.read_csv(f"{data}/duration.csv", index_col=0)
    vol = pd.read_csv(f"{data}/volume.csv", index_col=0)
    sites = pd.read_csv(f"{data}/sites.csv")
    site_col = 'site_id' if 'site_id' in sites.columns else 'site'
    site_ids = sites[site_col].astype(int).tolist()
    chg_map = {int(r[site_col]): float(r['charger_num']) for _, r in sites.iterrows()}
    charger = np.array([chg_map[s] for s in site_ids])
    mu_dur = dur[[str(s) for s in site_ids]].mean(0).values
    mu_vol = vol[[str(s) for s in site_ids]].mean(0).values

    # DINOv2 CLS, numeric-sorted mosaic-file order (extract_dinov2.py uses key=int)
    emb = np.load(f"{BASE}/tiles_dataset/{city}/embedding/z16/sat_embed_dinov2_768.npy")
    mosaic = f"{BASE}/tiles_dataset/{city}/mosaic/z16"
    fids = sorted([int(f.split('.')[0]) for f in os.listdir(mosaic) if f.endswith('.png')])
    emb_idx = {fid: i for i, fid in enumerate(fids)}
    X = np.stack([emb[emb_idx[s]] for s in site_ids])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return site_ids, X, charger, mu_dur, mu_vol


def main():
    hdr = (f"{'city':<5}{'n':>4}{'cos':>7}{'charger#':>9}{'power':>11} | "
           f"{'R2(chg#)':>9}{'R2(dur)':>9}{'R2(vol)':>9}{'R2(dur/chg)':>12}")
    print(hdr)
    print("-" * len(hdr))
    for city in CITIES:
        site_ids, X, charger, mu_dur, mu_vol = load_city(city)
        n = len(site_ids)
        cos = cosine_spread(X)
        chg = f"{int(charger.min())}-{int(charger.max())}"
        pwr = f"{mu_vol.min():.0f}-{mu_vol.max():.0f}"
        Xp = pca(X, PCA_DIM)
        r_chg = loo_ridge(Xp, np.log1p(charger))
        r_dur = loo_ridge(Xp, mu_dur)
        r_vol = loo_ridge(Xp, mu_vol)
        r_durc = loo_ridge(Xp, mu_dur / charger)
        print(f"{city:<5}{n:>4}{cos:>7.3f}{chg:>9}{pwr:>11} | "
              f"{r_chg:>9.3f}{r_dur:>9.3f}{r_vol:>9.3f}{r_durc:>12.3f}")


if __name__ == "__main__":
    main()
