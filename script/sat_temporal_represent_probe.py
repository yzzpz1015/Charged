# -*- coding: utf-8 -*-
"""Probe: does satellite imagery represent the *temporal shape* of demand,
transferable ACROSS cities?

Prior probes (static_context_cities.py) regressed satellite -> *level*
(charger_num / mean duration / mean volume) and found ~0.  This probe asks a
different question: does satellite encode the *time-series shape* (diurnal /
weekly profile, scale-free) of a site, and does that shape transfer across
cities via nearest neighbours in satellite-embedding space?

Task: for each target site, predict its normalized demand profile from the
profiles of its k nearest neighbours (by satellite embedding) among the OTHER
two cities.  Metric = cosine(predicted profile, true profile).  Baseline =
cosine(global-mean profile, true profile).  If satellite carries temporal
structure, kNN cosine should exceed the baseline.
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

BASE = "/home/data/yzzpz/Charged"
CITIES = ["JHB", "MEL", "LOA"]
K = 10


def site_col(df):
    return "site_id" if "site_id" in df.columns else "site"


def load_profiles(city, granularity="diurnal"):
    """Return (site_ids[int], profiles[N x D] L1-normalized)."""
    data = f"{BASE}/data/{city}_remove_const"
    dur = pd.read_csv(f"{data}/duration.csv", index_col=0)
    sites = pd.read_csv(f"{data}/sites.csv")
    sc = site_col(sites)
    site_ids = sites[sc].astype(int).tolist()

    t = pd.to_datetime(dur.index)
    hour = t.hour.values
    dow = t.dayofweek.values
    D = 24 if granularity == "diurnal" else 24 * 7
    key = hour if granularity == "diurnal" else (hour + 24 * dow)

    profs = []
    for s in site_ids:
        col = str(s)
        if col not in dur.columns:
            profs.append(np.zeros(D))
            continue
        v = dur[col].values.astype(float)
        acc = np.bincount(key, weights=v, minlength=D)
        cnt = np.bincount(key, minlength=D).astype(float)
        cnt[cnt == 0] = 1.0
        p = acc / cnt                      # mean per bin
        p = p / (p.sum() + 1e-9)           # L1-normalize -> shape only
        profs.append(p)
    return np.array(site_ids), np.array(profs)


def load_emb(city):
    """Return (site_ids[int], X[N x d]) aligned by mosaic-file id (numeric)."""
    emb = np.load(f"{BASE}/tiles_dataset/{city}/embedding/z16/sat_embed_dinov2_768.npy")
    mosaic = f"{BASE}/tiles_dataset/{city}/mosaic/z16"
    fids = sorted([int(f.split('.')[0]) for f in os.listdir(mosaic) if f.endswith('.png')])
    emb_idx = {fid: i for i, fid in enumerate(fids)}
    # keep only ids present in embedding AND profile
    return np.array(list(emb_idx.keys())), emb, emb_idx


def main():
    for gran in ["diurnal", "weekly"]:
        print(f"\n===== granularity = {gran} =====")
        # collect all cities
        ids, profs, Xs = {}, {}, {}
        for c in CITIES:
            sid, P = load_profiles(c, gran)
            eid, E, emb_idx = load_emb(c)
            common = sorted(set(sid) & set(eid))
            ids[c] = np.array(common)
            profs[c] = np.stack([P[list(sid).index(i)] for i in common])
            Xs[c] = np.stack([E[emb_idx[i]] for i in common])
            # L2-normalize embeddings for cosine NN
            Xs[c] = Xs[c] / (np.linalg.norm(Xs[c], axis=1, keepdims=True) + 1e-9)

        print(f"  site counts: " + " ".join(f"{c}={len(ids[c])}" for c in CITIES))

        # cross-site shape variance (is there shape variance to predict?)
        allP = np.vstack([profs[c] for c in CITIES])
        Cmat = allP @ allP.T
        np.fill_diagonal(Cmat, np.nan)
        off = Cmat[np.triu_indices(Cmat.shape[0], 1)]
        print(f"  cross-site profile cosine (all pairs): {np.nanmean(off):.4f}")

        print(f"  {'target':<5}{'n':>4} | {'base':>6}{'kNN':>6}{'delta':>7}")
        for tgt in CITIES:
            trn = [c for c in CITIES if c != tgt]
            Xtr = np.vstack([Xs[c] for c in trn])
            Ptr = np.vstack([profs[c] for c in trn])
            Xte, Pte = Xs[tgt], profs[tgt]

            # baseline: global mean profile of training sites
            gmean = Ptr.mean(0)
            gmean = gmean / (gmean.sum() + 1e-9)
            base_cos = Pte @ gmean

            # kNN by satellite embedding (cosine)
            nn = NearestNeighbors(n_neighbors=K, metric="cosine").fit(Xtr)
            dist, idx = nn.kneighbors(Xte)
            pred = np.stack([Ptr[idx[i]].mean(0) for i in range(len(idx))])
            pred = pred / (pred.sum(1, keepdims=True) + 1e-9)
            knn_cos = np.sum(Pte * pred, axis=1)  # both unit L1, ~cosine

            print(f"  {tgt:<5}{len(Pte):>4} | {base_cos.mean():>6.3f}"
                  f"{knn_cos.mean():>6.3f}{knn_cos.mean()-base_cos.mean():>+7.3f}")


if __name__ == "__main__":
    main()
