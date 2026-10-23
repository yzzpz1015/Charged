# -*- coding: utf-8 -*-
"""Corrected probe (v2) for "does satellite / land-use reveal demand LEVEL"?

Fixes over v1:
  - target z-scored WITHIN city (removes the 6x cross-city scale confound)
  - volume log1p'd (v1 had a 990 outlier wrecking R^2)
  - satellite kept at higher dim (no aggressive PCA-to-15); report both
  - charger_num also z-scored within city (so a near-constant control
    correctly yields ~no signal instead of free cross-city variance)
  - city identity as an explicit ceiling: how much does "knowing the city"
    already explain?
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

BASE = "/home/data/yzzpz/Charged"
CITIES = ["JHB", "MEL", "LOA"]
RADIUS_M = 1500.0
NOISE = {"other", "bench", "waste_basket", "post_box", "toilets", "drinking_water",
         "telephone", "shelter", "recycling", "waste", "vending_machine",
         "bicycle_parking", "waste_disposal", "clock", "fire_hydrant",
         "give_way", "traffic_signals", "crossing", "street_lamp", "grit_bin"}


def site_col(df):
    return "site_id" if "site_id" in df.columns else "site"


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def zscore(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-9 else 1.0)


def load_city(city):
    d = f"{BASE}/data/{city}_remove_const"
    dur, vol = (pd.read_csv(f"{d}/{f}.csv", index_col=0) for f in ("duration", "volume"))
    sites = pd.read_csv(f"{d}/sites.csv"); sc = site_col(sites)
    ids = sites[sc].astype(int).tolist()
    slat, slon = sites["latitude"].values.astype(float), sites["longitude"].values.astype(float)

    poi = pd.read_csv(f"{d}/poi.csv")
    pt, pla, plo = poi["type"].astype(str).values, poi["latitude"].values.astype(float), poi["longitude"].values.astype(float)
    k = ~np.isin(pt, list(NOISE)); pt, pla, plo = pt[k], pla[k], plo[k]
    from collections import Counter
    pc = []
    for i in range(len(ids)):
        pc.append(dict(Counter(pt[haversine(slat[i], slon[i], pla, plo) <= RADIUS_M])))

    E = np.load(f"{BASE}/tiles_dataset/{city}/embedding/z16/sat_embed_dinov2_768.npy")
    fids = sorted(int(f.split('.')[0]) for f in os.listdir(f"{BASE}/tiles_dataset/{city}/mosaic/z16") if f.endswith(".png"))
    ei = {f: i for i, f in enumerate(fids)}
    common = sorted(set(ids) & set(ei))
    rows = []
    for s in common:
        i = ids.index(s)
        rows.append({"city": city, "site": s,
                     "mu_dur": dur[str(s)].mean(), "mu_vol": vol[str(s)].mean(),
                     "charger": float(sites[sites[sc] == s]["charger_num"].iloc[0]),
                     "poi": pc[i], "emb": E[ei[s]]})
    return rows


def loo_ridge(X, y, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    X, y = np.asarray(X, float), np.asarray(y, float)
    best = min(alphas, key=lambda a: sum(
        (Ridge(alpha=a).fit(X[tr], y[tr]).predict(X[va])[0] - y[va]) ** 2
        for tr, va in LeaveOneOut().split(X)))
    yh = np.zeros(len(y))
    for tr, va in LeaveOneOut().split(X):
        yh[va] = Ridge(alpha=best).fit(X[tr], y[tr]).predict(X[va])[0]
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - yh) ** 2).sum() / ss if ss > 0 else np.nan
    return spearmanr(yh, y).statistic, r2, best, yh


def _loo_detail(X, y, alpha):
    X, y = np.asarray(X, float), np.asarray(y, float)
    yh = np.zeros(len(y))
    for tr, va in LeaveOneOut().split(X):
        yh[va] = Ridge(alpha=alpha).fit(X[tr], y[tr]).predict(X[va])[0]
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - yh) ** 2).sum() / ss if ss > 0 else np.nan
    return spearmanr(yh, y).statistic, r2, alpha, yh


def main():
    allr = [r for c in CITIES for r in load_city(c)]
    # within-city z-score of every target/feature-of-interest
    for key in ("mu_dur", "mu_vol", "charger"):
        vals = np.array([r[key] for r in allr], float)
        out = np.zeros_like(vals)
        for c in CITIES:
            m = np.array([r["city"] == c for r in allr])
            out[m] = zscore(vals[m])
        for i, r in enumerate(allr):
            r[key + "_z"] = out[i]

    from collections import Counter
    cnt = Counter()
    for r in allr:
        cnt.update(r["poi"])
    cats = [k for k, _ in cnt.most_common(15)]
    LU = np.stack([np.log1p([r["poi"].get(c, 0) for c in cats]) for r in allr])
    LU = (LU - LU.mean(0)) / (LU.std(0) + 1e-9)
    E = np.stack([r["emb"] for r in allr])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    CHG = np.array([r["charger_z"] for r in allr]).reshape(-1, 1)

    y_dur = np.array([r["mu_dur_z"] for r in allr])
    y_vol = np.array([r["mu_vol_z"] for r in allr])
    # log1p THEN within-city z-score (v1 had this as a global z-score: label bug)
    _lv = np.log1p([r["mu_vol"] for r in allr])
    y_logvol = np.zeros_like(_lv)
    for c in CITIES:
        m = np.array([r["city"] == c for r in allr])
        y_logvol[m] = zscore(_lv[m])

    def rep(name, X, y):
        rho, r2, a, yh = loo_ridge(X, y)
        # guard: near-constant predictions make Spearman meaningless
        flag = "  <-- yhat degenerate" if yh.std() < 0.05 * y.std() else ""
        print(f"  {name:<26} rho={rho:+.3f}  R2={r2:+.3f}  yhat_std={yh.std():.3f}{flag}")

    for tag, y in [("mean duration (within-city z)", y_dur),
                   ("log mean volume (within-city z)", y_logvol)]:
        print(f"\n=== {tag}   n={len(allr)} ===")
        rep("land-use (POI 1500m)", LU, y)
        rep("satellite PCA-15", PCA(n_components=15).fit_transform(E), y)
        rep("satellite PCA-50", PCA(n_components=50).fit_transform(E), y)
        rep("charger_num (near-const ctrl)", CHG, y)

    # ceiling: city identity alone (one-hot) explains how much?
    CITY = np.stack([[1.0 if r["city"] == c else 0.0 for c in CITIES] for r in allr])
    print("\n=== ceiling: city-identity one-hot ===")
    rep("city one-hot -> dur_z", CITY, y_dur)
    rep("city one-hot -> logvol_z", CITY, y_logvol)


if __name__ == "__main__":
    main()
