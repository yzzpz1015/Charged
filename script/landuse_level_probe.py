# -*- coding: utf-8 -*-
"""Probe: does land-use (POI within 1500m) explain demand LEVEL?

Tests the user's chain "satellite -> land-use -> demand level" at the OUTCOME
edge: land-use -> mean duration / mean volume.  Side by side with satellite
(DINOv2 768) and charger_num (mechanical control).  LOO ridge, Spearman + R^2.

Land-use feature = log1p count of meaningful POI categories within 1500m,
computed uniformly from poi.csv (type/lon/lat) for all three cities.
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
PCA_DIM = 15
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
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_city(city):
    data = f"{BASE}/data/{city}_remove_const"
    dur = pd.read_csv(f"{data}/duration.csv", index_col=0)
    vol = pd.read_csv(f"{data}/volume.csv", index_col=0)
    sites = pd.read_csv(f"{data}/sites.csv")
    sc = site_col(sites)
    site_ids = sites[sc].astype(int).tolist()
    slat = sites["latitude"].values.astype(float)
    slon = sites["longitude"].values.astype(float)

    poi = pd.read_csv(f"{data}/poi.csv")
    ptype = poi["type"].astype(str).values
    plat = poi["latitude"].values.astype(float)
    plon = poi["longitude"].values.astype(float)
    keep = ~np.isin(ptype, list(NOISE))
    ptype, plat, plon = ptype[keep], plat[keep], plon[keep]

    # per-site POI category counts within radius
    from collections import Counter
    poi_counts = []
    for i in range(len(site_ids)):
        d = haversine(slat[i], slon[i], plat, plon)
        cnt = Counter(ptype[d <= RADIUS_M])
        poi_counts.append(dict(cnt))

    emb = np.load(f"{BASE}/tiles_dataset/{city}/embedding/z16/sat_embed_dinov2_768.npy")
    mosaic = f"{BASE}/tiles_dataset/{city}/mosaic/z16"
    fids = sorted([int(f.split('.')[0]) for f in os.listdir(mosaic) if f.endswith('.png')])
    emb_idx = {fid: i for i, fid in enumerate(fids)}

    common = sorted(set(site_ids) & set(emb_idx.keys()))
    rows = []
    for s in common:
        i = site_ids.index(s)
        chg = float(sites[sites[sc] == s]["charger_num"].iloc[0])
        rows.append({
            "site": s, "mu_dur": dur[str(s)].mean(), "mu_vol": vol[str(s)].mean(),
            "charger": chg, "poi": poi_counts[i], "emb": emb[emb_idx[s]],
        })
    return rows


def loo_ridge(X, y, alphas=[0.1, 1.0, 10.0, 100.0]):
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = len(y)
    best = min(alphas, key=lambda a: sum(
        (Ridge(alpha=a).fit(X[tr], y[tr]).predict(X[va])[0] - y[va]) ** 2
        for tr, va in LeaveOneOut().split(X)))
    yhat = np.zeros(n)
    for tr, va in LeaveOneOut().split(X):
        m = Ridge(alpha=best).fit(X[tr], y[tr])
        yhat[va] = m.predict(X[va])[0]
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - yhat) ** 2).sum() / ss if ss > 0 else np.nan
    return spearmanr(yhat, y).statistic, r2, best


def main():
    all_rows = []
    city_of = []
    for c in CITIES:
        rs = load_city(c)
        all_rows.extend(rs); city_of += [c] * len(rs)
    city_of = np.array(city_of)

    # top meaningful categories across pool
    from collections import Counter
    cnt = Counter()
    for r in all_rows:
        cnt.update(r["poi"])
    cats = [k for k, _ in cnt.most_common(15)]
    print("land-use categories:", cats)

    y_dur = np.array([r["mu_dur"] for r in all_rows])
    y_vol = np.array([r["mu_vol"] for r in all_rows])
    chg = np.array([r["charger"] for r in all_rows]).reshape(-1, 1)
    emb = np.stack([r["emb"] for r in all_rows])
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    LU = np.zeros((len(all_rows), len(cats)))
    for i, r in enumerate(all_rows):
        for j, c in enumerate(cats):
            LU[i, j] = np.log1p(r["poi"].get(c, 0))
    SAT = PCA(n_components=PCA_DIM).fit_transform(emb)

    def report(name, X, y):
        rho, r2, a = loo_ridge(X, y)
        print(f"  {name:<20} rho={rho:+.3f}  R2={r2:+.3f}")

    for label, y in [("mean duration", y_dur), ("mean volume", y_vol)]:
        print(f"\n=== POOLED (n={len(all_rows)})  target = {label} ===")
        report("land-use (POI)", LU, y)
        report("satellite (DINOv2)", SAT, y)
        report("charger_num", chg, y)

    print("\n=== per-city satellite -> mean duration (LOO) ===")
    for c in CITIES:
        m = city_of == c
        rho, r2, _ = loo_ridge(SAT[m], y_dur[m])
        print(f"  {c:<5} rho={rho:+.3f}  R2={r2:+.3f}")


if __name__ == "__main__":
    main()
