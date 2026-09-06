# -*- coding: utf-8 -*-
"""Decisive static-context re-test.

Question (after the "label + encoder" correction): does a DISCRIMINATIVE
geo-encoder (RemoteCLIP patch grid, cosine 0.67) recover a busy-ness signal
(mean DURATION, power-independent) where DINOv2 CLS (cosine 0.87) was ~0 —
and, decisively, does that signal survive into the FORECAST RESIDUAL (what the
frozen Kairos + RevIN cannot already predict)?

If counterfactual (->duration) turns positive but residual (->e_s) stays ~0,
then satellite's busy-ness signal is RevIN-redundant for the forecast task.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

BASE = "/home/data/yzzpz/Charged"
sys.path.insert(0, BASE)
from api.dataset.common import EVDataset  # noqa: E402

CITY = "MEL"
SEQ_L, PRE_LEN = 72, 6
ALPHAS = [0.1, 1.0, 10.0, 100.0]
PCA_DIM = 15


def pca(X, n):
    return X if X.shape[1] <= n else PCA(n_components=n).fit_transform(X)


def loo_ridge(X, y, alphas=ALPHAS):
    X = np.asarray(X, float); y = np.asarray(y, float)
    N = len(y); best_a, best_r2 = alphas[0], -1e18
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
            best_r2, best_a = r2, a
    return best_r2, best_a


def perm_pval(X, y, n=200, alphas=ALPHAS):
    base, _ = loo_ridge(X, y, alphas)
    rng = np.random.RandomState(0); hits = 0
    for _ in range(n):
        r2, _ = loo_ridge(X, rng.permutation(y), alphas)
        if r2 >= base:
            hits += 1
    return base, hits / n


def load_residual():
    ev = EVDataset(feature="volume", auxiliary="None",
                   data_path=f"{BASE}/data/{CITY}_remove_zero/")
    ev.split_cross_validation(fold=6, total_fold=6, train_ratio=0.8, valid_ratio=0.1)
    T, N = ev.feat.shape
    te_s, te_e = ev.test_start, ev.test_end
    num_windows = te_e - te_s - SEQ_L - PRE_LEN
    lab = np.load(f"{BASE}/result/kairos_l1/label_plain_pre6_fold6.npy")
    pred = np.load(f"{BASE}/result/kairos_l1/predict_plain_pre6_fold6.npy")
    assert lab.shape[0] == num_windows * N
    resid = (lab - pred).reshape(num_windows, N, PRE_LEN)
    return resid, N


def load_sat():
    import os
    vol = pd.read_csv(f"{BASE}/data/{CITY}_remove_zero/volume.csv", index_col=0)
    dur = pd.read_csv(f"{BASE}/data/{CITY}_remove_zero/duration.csv", index_col=0)
    site_ids = vol.columns.astype(int).tolist()
    # labels reindexed to site_ids order (duration columns may be string-sorted)
    mu_vol = vol[[str(s) for s in site_ids]].mean(0).values
    mu_dur = dur[[str(s) for s in site_ids]].mean(0).values
    # DINOv2 CLS: remap via sat_sites id -> row
    dino_raw = np.load(f"{BASE}/tiles_dataset/{CITY}/embedding/z16/sat_embed_dinov2_768.npy")
    sat_sites = pd.read_csv(f"{BASE}/tiles_dataset/{CITY}/embedding/z16/sat_embed_256_sites.csv")["site_id"].tolist()
    dino_idx = {int(s): i for i, s in enumerate(sat_sites)}
    dino = np.stack([dino_raw[dino_idx[s]] for s in site_ids])
    dino = dino / (np.linalg.norm(dino, axis=1, keepdims=True) + 1e-9)
    # RemoteCLIP: saved in mosaic-file (numeric) order -> remap via file id
    rc_cls_raw = np.load(f"{BASE}/tiles_dataset/{CITY}/embedding/z16/remoteclip_cls.npy")
    rc_g7_raw = np.load(f"{BASE}/tiles_dataset/{CITY}/embedding/z16/remoteclip_grid7.npy")
    mosaic = f"{BASE}/tiles_dataset/{CITY}/mosaic/z16"
    fids = sorted([int(f.split('.')[0]) for f in os.listdir(mosaic) if f.endswith('.png')])
    rc_idx = {fid: i for i, fid in enumerate(fids)}
    rc_cls = np.stack([rc_cls_raw[rc_idx[s]] for s in site_ids])
    rc_g7 = np.stack([rc_g7_raw[rc_idx[s]] for s in site_ids])
    return site_ids, mu_vol, mu_dur, dino, rc_cls, rc_g7


def main():
    site_ids, mu_vol, mu_dur, dino, rc_cls, rc_g7 = load_sat()
    resid, N = load_residual()
    e_s = resid[:, :, 0].mean(axis=0)          # per-site mean 1-step residual
    print(f"== {CITY}: {N} sites ==")

    reps = {"DINOv2 CLS": dino, "RC CLS": rc_cls, "RC grid7x7": rc_g7}

    print("\n[counterfactual] context -> busy-ness (mean volume / duration)")
    for name, R in reps.items():
        X = pca(R, PCA_DIM)
        rv, _ = loo_ridge(X, mu_vol)
        rd, _ = loo_ridge(X, mu_dur)
        print(f"  {name:<14} vol R2={rv:+.3f}   dur R2={rd:+.3f}")
    base, p = perm_pval(pca(rc_g7, PCA_DIM), mu_dur, n=200)
    print(f"  [perm] RC grid7x7 -> duration: R2={base:+.3f} p={p:.3f}")

    print("\n[DECISIVE probe A] context -> mean residual e_s")
    for name, R in reps.items():
        r2, a = loo_ridge(pca(R, PCA_DIM), e_s)
        print(f"  {name:<14} LOO R2 = {r2:+.3f} (alpha={a})")
    base, p = perm_pval(pca(rc_g7, PCA_DIM), e_s, n=200)
    print(f"  [perm] RC grid7x7 -> residual: R2={base:+.3f} p={p:.3f}")


if __name__ == "__main__":
    main()
