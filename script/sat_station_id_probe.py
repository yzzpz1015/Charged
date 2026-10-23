# -*- coding: utf-8 -*-
"""Decisive test: is VisionEV's "imagery helps" effect STATION IDENTITY, not vision?

VisionEV (189 stations x ~121 weeks) feeds a per-station CONSTANT image into a
trainable model, split by station-WEEK rows.  Because the image is constant per
station, a random station-week split puts station A's image in the training set
AND in the test set -> any trainable map can memorize station A's level from its
training weeks and replay it on its test weeks.  That is a station-ID / random
effect, and it CANNOT transfer to a new station or a new city.

Prediction if the mechanism is station-ID:
  split A (random station-week rows):  image ridge ~= station-train-mean baseline
  split B (leave-station-out):         image ridge ~= 0

Also reports whether the image adds anything ON TOP of the station's own train
mean (it cannot: a constant feature per station can only explain a constant).
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

BASE = "/home/data/yzzpz/Charged"
CITIES = ["JHB", "MEL", "LOA"]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def r2(y, yh):
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - yh) ** 2).sum() / ss if ss > 1e-12 else np.nan


def load(city):
    d = f"{BASE}/data/{city}_remove_const"
    vol = pd.read_csv(f"{d}/volume.csv", index_col=0)
    dur = pd.read_csv(f"{d}/duration.csv", index_col=0)
    vol.index = pd.to_datetime(vol.index)
    dur.index = pd.to_datetime(dur.index)
    sites = pd.read_csv(f"{d}/sites.csv")
    sc = "site_id" if "site_id" in sites.columns else "site"
    ids = sites[sc].astype(int).tolist()

    E = np.load(f"{BASE}/tiles_dataset/{city}/embedding/z16/sat_embed_dinov2_768.npy")
    fids = sorted(int(f.split('.')[0]) for f in os.listdir(f"{BASE}/tiles_dataset/{city}/mosaic/z16") if f.endswith(".png"))
    ei = {f: i for i, f in enumerate(fids)}
    common = [s for s in ids if s in ei]
    if not common:
        return None

    # ---- weekly panel: rows = (station, week) ----
    wv = vol[[str(s) for s in common]].resample("W").sum()      # weekly total volume
    wd = dur[[str(s) for s in common]].resample("W").mean()     # weekly mean duration
    keep = (wv.sum(1) > 0) & (wd.notna().sum(1) > 0)
    wv, wd = wv[keep], wd[keep]
    S, W = wv.shape[1], len(wv)
    Xs = np.stack([E[ei[s]] for s in common])
    Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)
    rows = []
    for j, s in enumerate(common):
        for i in range(W):
            rows.append((j, i,
                         np.log1p(wv.iloc[i, j]) if np.isfinite(wv.iloc[i, j]) else np.nan,
                         wd.iloc[i, j] if np.isfinite(wd.iloc[i, j]) else np.nan))
    df = pd.DataFrame(rows, columns=["sj", "wi", "vol", "dur"]).dropna()
    return Xs, df, S, W


def split_A(df, seed=0):                      # random station-WEEK rows (VisionEV-like)
    rng = np.random.RandomState(seed)
    m = rng.rand(len(df)) < 0.7
    return df[m].index.values, df[~m].index.values


def split_B(df, nfold=5, seed=0):             # leave-STATION-out
    rng = np.random.RandomState(seed)
    st = df["sj"].unique()
    rng.shuffle(st)
    fold = {s: i % nfold for i, s in enumerate(st)}
    return df["sj"].map(fold).values


def run(Xs, df, ycol):
    y = df[ycol].values
    out = {"n_rows": len(df), "n_sta": df["sj"].nunique()}

    # ---------- split A ----------
    tr, te = split_A(df)
    Xtr, Xte = Xs[df["sj"].values[tr]], Xs[df["sj"].values[te]]
    ytr, yte = y[tr], y[te]
    best = min(ALPHAS, key=lambda a: ((Ridge(alpha=a).fit(Xtr, ytr).predict(Xtr) - ytr) ** 2).mean())
    yh_img = Ridge(alpha=best).fit(Xtr, ytr).predict(Xte)
    # station train-mean baseline (unseen station -> global train mean)
    gm = ytr.mean()
    smean = pd.Series(ytr).groupby(df["sj"].values[tr]).mean()
    yh_sta = np.array([smean.get(s, gm) for s in df["sj"].values[te]])
    out["A_img"] = r2(yte, yh_img)
    out["A_sta"] = r2(yte, yh_sta)
    out["A_glo"] = r2(yte, np.full(len(yte), gm))
    # image ON TOP of station mean: does the image explain any WITHIN-station residual?
    yh_combo = yh_sta + Ridge(alpha=best).fit(Xtr, ytr - np.array([smean.get(s, gm) for s in df["sj"].values[tr]])).predict(Xte)
    out["A_sta+img"] = r2(yte, yh_combo)

    # ---------- split B ----------
    fold = split_B(df)
    yhB = np.zeros(len(df))
    for f in range(fold.max() + 1):
        tr, te = fold != f, fold == f
        b = min(ALPHAS, key=lambda a: ((Ridge(alpha=a).fit(Xs[df["sj"].values[tr]], y[tr]).predict(Xs[df["sj"].values[tr]]) - y[tr]) ** 2).mean())
        yhB[te] = Ridge(alpha=b).fit(Xs[df["sj"].values[tr]], y[tr]).predict(Xs[df["sj"].values[te]])
    out["B_img"] = r2(y, yhB)
    out["B_glo"] = r2(y, np.full(len(y), y.mean()))
    return out


def main():
    for ycol, lab in [("vol", "weekly VOLUME (log1p)"), ("dur", "weekly mean DURATION")]:
        print(f"\n{'='*74}\nTARGET = {lab}\n{'='*74}")
        print(f"{'city':<6}{'rows':>7}{'sta':>6} | {'A_img':>8}{'A_sta':>8}{'A_sta+img':>10} | {'B_img':>8}")
        print("-" * 74)
        for c in CITIES:
            r = load(c)
            if r is None:
                print(f"{c:<6}  (no satellite)")
                continue
            Xs, df, S, W = r
            o = run(Xs, df, ycol)
            print(f"{c:<6}{o['n_rows']:>7}{o['n_sta']:>6} | {o['A_img']:>8.3f}{o['A_sta']:>8.3f}"
                  f"{o['A_sta+img']:>10.3f} | {o['B_img']:>8.3f}")
        print("\n  A_* = random station-week split (VisionEV-like); B_* = leave-station-out (cross-city-like)")
        print("  A_sta+img = station train-mean + image ridge on the residual")


if __name__ == "__main__":
    main()
