# -*- coding: utf-8 -*-
"""Test the user's proposal: new city + K calibrated sites (satellite + observed
value) -> propagate to the remaining sites via SATELLITE SIMILARITY.

If satellite-similar sites have similar demand, kNN-in-satellite-space from K
labeled sites should predict the held-out sites better than simply predicting
the labeled mean.  Baseline = labeled mean (no satellite).  Also runs kNN in a
RANDOM space as a null, and ridge on satellite as a second estimator.

Run per city (within-city: the user acknowledges cross-city SCALE is not
satellite-explainable, so calibration is local).
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

BASE = "/home/data/yzzpz/Charged"
CITIES = ["JHB", "MEL", "LOA"]
KS = [5, 10, 20, 40]
SEEDS = 30
KN = 5  # neighbours used inside kNN


def sc(df):
    return "site_id" if "site_id" in df.columns else "site"


def load(city):
    d = f"{BASE}/data/{city}_remove_const"
    dur = pd.read_csv(f"{d}/duration.csv", index_col=0)
    vol = pd.read_csv(f"{d}/volume.csv", index_col=0)
    sites = pd.read_csv(f"{d}/sites.csv"); c = sc(sites)
    ids = sites[c].astype(int).tolist()
    E = np.load(f"{BASE}/tiles_dataset/{city}/embedding/z16/sat_embed_dinov2_768.npy")
    fids = sorted(int(f.split('.')[0]) for f in os.listdir(f"{BASE}/tiles_dataset/{city}/mosaic/z16") if f.endswith(".png"))
    ei = {f: i for i, f in enumerate(fids)}
    common = sorted(set(ids) & set(ei))
    X = np.stack([E[ei[s]] for s in common])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    y = np.array([dur[str(s)].mean() for s in common])
    return X, y


def r2(y, yh):
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - yh) ** 2).sum() / ss if ss > 0 else np.nan


def knn_predict(Xl, yl, Xu, k):
    S = Xu @ Xl.T                                   # cosine sim
    idx = np.argsort(-S, axis=1)[:, :k]
    w = np.take_along_axis(S, idx, axis=1).clip(0, None)
    w = w / (w.sum(1, keepdims=True) + 1e-12)
    return (yl[idx] * w).sum(1)


def main():
    for city in CITIES:
        X, y = load(city)
        n = len(y)
        print(f"\n### {city}  n={n}  (target = mean duration)")
        hdr = f"  {'K':>4} | {'base':>7}{'satknn':>8}{'randknn':>9}{'ridge':>7} | {'rho_sat':>8}{'rho_base':>9}"
        print(hdr)
        for K in KS:
            if K >= n - 5:
                continue
            acc = {k: [] for k in ("base", "sat", "rand", "ridge")}
            rhos = {"sat": [], "base": []}
            rng = np.random.RandomState(0)
            for _ in range(SEEDS):
                lab = rng.choice(n, K, replace=False)
                m = np.ones(n, bool); m[lab] = False
                Xl, yl, Xu, yu = X[lab], y[lab], X[m], y[m]
                # base: labeled mean
                yb = np.full(len(yu), yl.mean())
                # satellite kNN
                ys = knn_predict(Xl, yl, Xu, min(KN, K))
                # random-space kNN (null)
                Rl, Ru = rng.randn(*Xl.shape), rng.randn(*Xu.shape)
                Rl /= np.linalg.norm(Rl, axis=1, keepdims=True) + 1e-9
                Ru /= np.linalg.norm(Ru, axis=1, keepdims=True) + 1e-9
                yr = knn_predict(Rl, yl, Ru, min(KN, K))
                # ridge on satellite (PCA to min(15,K-1))
                p = PCA(n_components=max(2, min(15, K - 1))).fit(Xl)
                yg = Ridge(alpha=10.0).fit(p.transform(Xl), yl).predict(p.transform(Xu))
                for k, v in (("base", yb), ("sat", ys), ("rand", yr), ("ridge", yg)):
                    acc[k].append(r2(yu, v))
                rhos["sat"].append(spearmanr(ys, yu).statistic)
                rhos["base"].append(np.nan)  # constant -> undefined
            print(f"  {K:>4} | {np.mean(acc['base']):>7.3f}{np.mean(acc['sat']):>8.3f}"
                  f"{np.mean(acc['rand']):>9.3f}{np.mean(acc['ridge']):>7.3f} | "
                  f"{np.nanmean(rhos['sat']):>8.3f}{'  n/a':>9}")


if __name__ == "__main__":
    main()
