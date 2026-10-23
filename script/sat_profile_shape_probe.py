# -*- coding: utf-8 -*-
"""Does the satellite image reveal a station's TEMPORAL PROFILE SHAPE
(not just its mean level)?  This is the user's challenge to the
"image can only explain the station mean" claim -- that claim holds only
for a LINEAR additive model.  Here the target IS the within-station
week-to-week shape, so a linear map can express it.

Decomposition of log1p weekly volume Y[s,w]:
    Y = mu          (global)
      + a_s         (station main effect  = LEVEL)
      + b_w         (week main effect     = common calendar)
      + R[s,w]      (interaction residual = station-specific SHAPE)
We test whether the image predicts R (shape) and a_s (level/value), with a
RANDOM per-station code as the null.  LOO over stations = leave-station-out.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

BASE = "/home/data/yzzpz/Charged"
CITIES = ["JHB", "MEL", "LOA"]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def load(city):
    d = f"{BASE}/data/{city}_remove_const"
    vol = pd.read_csv(f"{d}/volume.csv", index_col=0)
    vol.index = pd.to_datetime(vol.index)
    sites = pd.read_csv(f"{d}/sites.csv")
    sc = "site_id" if "site_id" in sites.columns else "site"
    ids = sites[sc].astype(int).tolist()
    E = np.load(f"{BASE}/tiles_dataset/{city}/embedding/z16/sat_embed_dinov2_768.npy")
    fids = sorted(int(f.split('.')[0]) for f in os.listdir(f"{BASE}/tiles_dataset/{city}/mosaic/z16") if f.endswith(".png"))
    ei = {f: i for i, f in enumerate(fids)}
    common = [s for s in ids if s in ei]
    W = vol[[str(s) for s in common]].resample("W").sum()
    W = W[W.sum(1) > 0]
    Y = np.log1p(W.values.T)                       # [S, T]
    X = np.stack([E[ei[s]] for s in common])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X, Y, common


def decomp(Y):
    mu = Y.mean()
    A = Y.mean(1, keepdims=True)                   # station main effect (LEVEL)
    B = (Y - A).mean(0, keepdims=True)             # week main effect (calendar)
    R = Y - mu - (A - mu) - B                      # interaction residual (SHAPE)
    return A - mu, B, R


def loo_ridge(X, y, alphas=ALPHAS):
    X, y = np.asarray(X, float), np.asarray(y, float)
    best = min(alphas, key=lambda a: np.mean([
        ((Ridge(alpha=a).fit(X[tr], y[tr]).predict(X[va]) - y[va]) ** 2).mean()
        for tr, va in LeaveOneOut().split(X)]))
    yh = np.zeros_like(y)
    for tr, va in LeaveOneOut().split(X):
        yh[va] = Ridge(alpha=best).fit(X[tr], y[tr]).predict(X[va])
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum(), best, yh


def main():
    print("目标: 站水平 a_s / 站特有形状 R[s,w]。LOO=leave-station-out。")
    print("null = 每站一个随机码（同容量）\n")
    for city in CITIES:
        X, Y, ids = load(city)
        A, B, R = decomp(Y)
        n, T = Y.shape
        print(f"### {city}  n={n} stations x {T} weeks")
        rng = np.random.RandomState(0)
        Xr = rng.randn(n, 768); Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
        for tag, y in [("站水平 a_s [n,1]", A),
                       ("站特有形状 R [n,T]", R)]:
            r2s, _a, _ = loo_ridge(X, y)
            r2r, _b, _ = loo_ridge(Xr, y)
            print(f"  {tag:<22} 真图 R2={r2s:+.3f}   随机码 R2={r2r:+.3f}")
        print(f"  （形状 R 的方差占比: {R.var()/Y.var():.3f}，即去掉水平和日历后还剩多少）\n")


if __name__ == "__main__":
    main()
