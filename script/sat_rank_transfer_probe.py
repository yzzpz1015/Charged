# -*- coding: utf-8 -*-
"""Decisive probe for "satellite transfers RELATIVE STRUCTURE (ranking)".

Prior probes regressed satellite -> LEVEL with R^2 and found ~0/negative.
R^2 is scale-sensitive; the user's claim is ordinal (ranking).  This probe
re-tests the claim with RANK correlation, cross-city leave-one-city-out:

  train ridge(satellite -> busyness) on 2 source cities,
  predict the 3rd city's sites, rank them, Spearman vs true rank.

Baselines: (a) rank by charger_num (mechanical capacity->throughput signal,
the only signal ever found), (b) random.  If satellite rank <= charger_num
rank, the "relative structure" claim is dead; if it beats charger_num, there
is a genuine satellite-visible signal.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

BASE = "/home/data/yzzpz/Charged"
CITIES = ["JHB", "MEL", "LOA"]
PCA_DIM = 15
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def site_col(df):
    return "site_id" if "site_id" in df.columns else "site"


def load_city(city):
    data = f"{BASE}/data/{city}_remove_const"
    dur = pd.read_csv(f"{data}/duration.csv", index_col=0)
    sites = pd.read_csv(f"{data}/sites.csv")
    sc = site_col(sites)
    site_ids = sites[sc].astype(int).tolist()
    chg = np.array([float(sites[sites[sc] == s]["charger_num"].iloc[0]) for s in site_ids])
    y = np.array([dur[str(s)].mean() for s in site_ids])          # mean duration = busyness

    emb = np.load(f"{BASE}/tiles_dataset/{city}/embedding/z16/sat_embed_dinov2_768.npy")
    mosaic = f"{BASE}/tiles_dataset/{city}/mosaic/z16"
    fids = sorted([int(f.split('.')[0]) for f in os.listdir(mosaic) if f.endswith('.png')])
    emb_idx = {fid: i for i, fid in enumerate(fids)}
    common = sorted(set(site_ids) & set(emb_idx.keys()))
    sid = np.array(common)
    y = np.array([dur[str(s)].mean() for s in common])
    chg = np.array([float(sites[sites[sc] == s]["charger_num"].iloc[0]) for s in common])
    X = np.stack([emb[emb_idx[s]] for s in common])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return sid, X, chg, y


def loo_cross_city_ridge(tgt):
    trn = [c for c in CITIES if c != tgt]
    Xtr, ytr = [], []
    for c in trn:
        _, X, chg, y = load_city(c)
        Xtr.append(X); ytr.append(y)
    Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
    pca = PCA(n_components=min(PCA_DIM, Xtr.shape[1])).fit(Xtr)
    Xtr_p = pca.transform(Xtr)

    _, Xte, chg_te, yte = load_city(tgt)
    Xte_p = pca.transform(Xte)

    best = None
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Xtr_p, ytr)
        best = m if best is None else best  # keep last (weak reg irrelevant for rank)

    # multi-alpha: pick by minimizing CV MSE on training (simple LOO within train)
    def loo_cv(a):
        from sklearn.model_selection import LeaveOneOut
        errs = []
        for tr, va in LeaveOneOut().split(Xtr_p):
            m = Ridge(alpha=a).fit(Xtr_p[tr], ytr[tr])
            errs.append((m.predict(Xtr_p[va])[0] - ytr[va]) ** 2)
        return np.mean(errs)
    best_a = min(ALPHAS, key=loo_cv)
    model = Ridge(alpha=best_a).fit(Xtr_p, ytr)
    yhat = model.predict(Xte_p)
    return yhat, chg_te, yte, best_a


def main():
    print("busyness = mean duration per site; rank metric = Spearman rho")
    print(f"{'target':<6}{'n':>4} | {'sat_rho':>8}{'chg_rho':>8}{'sat-chg':>9} | best_a")
    print("-" * 48)
    for tgt in CITIES:
        yhat, chg, y, a = loo_cross_city_ridge(tgt)
        r_sat = spearmanr(yhat, y).statistic
        r_chg = spearmanr(chg, y).statistic
        print(f"{tgt:<6}{len(y):>4} | {r_sat:>8.3f}{r_chg:>8.3f}{r_sat-r_chg:>+9.3f} | {a}")

    # pooled (all cities together) rank corr as a summary
    print("\n--- pooled (all 3 cities, LOO within-pool) ---")
    sid, X, chg, y = [], [], [], []
    for c in CITIES:
        s, Xc, cc, yc = load_city(c)
        sid.append(s); X.append(Xc); chg.append(cc); y.append(yc)
    X = np.vstack(X); chg = np.concatenate(chg); y = np.concatenate(y)
    from sklearn.model_selection import LeaveOneOut
    pca = PCA(n_components=PCA_DIM).fit(X); Xp = pca.transform(X)
    yhat = np.zeros_like(y)
    for tr, va in LeaveOneOut().split(Xp):
        m = Ridge(alpha=10.0).fit(Xp[tr], y[tr])
        yhat[va] = m.predict(Xp[va])[0]
    print(f"pooled satellite spearman: {spearmanr(yhat, y).statistic:.3f}")
    print(f"pooled charger_num spearman: {spearmanr(chg, y).statistic:.3f}")


if __name__ == "__main__":
    main()
