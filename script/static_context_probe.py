# -*- coding: utf-8 -*-
"""Static context (satellite + POI) vs L1 residual probe.

Core question (user's design): can static context predict the residual of the
FROZEN-TSFM base forecast?  If load LEVEL is predictable by context but the
RESIDUAL is not, the base forecast has already absorbed all context info -> no
increment for static modalities (the negative-result reading of L2's +0.005).

Method (all linear/ridge, no complex models):
  Residual: e_{s,t} = label - predict of L1 (frozen Kairos + trained point head).
    Predictions are saved (predict_plain / label_plain, fold6, pre_len=6).
    Ordering is window-major: sample k -> window i = k//N, site n = k%N.
  Static context s_s = [satellite DINOv2 768 (L2-norm) | POI count vector].
  Probe A (site bias):  mean residual e_s ~ ridge(s_s)           (LOO CV + permutation)
  Probe B (site pattern): (weekend - weekday) residual d_s ~ ridge(s_s)
  Probe C (per-time):    e_{s,t} ~ ridge(s_s + calendar + s_s*calendar)  (time-split)
  Counterfactual:        mean raw load level mu_s ~ ridge(s_s)  (must be high R^2)
                         mean residual e_s ~ ridge(s_s)          (the decisive number)
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.decomposition import PCA

BASE = "/home/data/yzzpz/Charged"
sys.path.insert(0, BASE)
from api.dataset.common import EVDataset  # noqa: E402

CITY = "MEL"
SEQ_L = 72
PRE_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 6
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
POI_MAX_DIM = 32


def pca(X, n):
    if X.shape[1] <= n:
        return X
    return PCA(n_components=n).fit_transform(X)


def load_static():
    """Return (site_ids in volume-column order, sat[N,768], poi[N,V], poi_vocab)."""
    vol = pd.read_csv(f"{BASE}/data/{CITY}_remove_zero/volume.csv", index_col=0)
    site_ids = vol.columns.astype(int).tolist()          # string-sorted order (feat column order)
    # satellite
    sat = np.load(f"{BASE}/tiles_dataset/{CITY}/embedding/z16/sat_embed_dinov2_768.npy")
    sat_sites = pd.read_csv(f"{BASE}/tiles_dataset/{CITY}/embedding/z16/sat_embed_256_sites.csv")["site_id"].tolist()
    sat_idx = {s: i for i, s in enumerate(sat_sites)}    # numeric order -> row
    # POI count vector from site_poi_1500m.csv string
    poi_df = pd.read_csv(f"{BASE}/data/{CITY}_remove_zero/site_poi_1500m.csv")
    poi_rows = {int(r["site_id"]): r["poi_str"] for _, r in poi_df.iterrows()}
    # union vocabulary
    vocab = {}
    for sid in site_ids:
        for tok in poi_rows[sid].split(","):
            k = tok.split(":")[0].strip()
            if k and k not in vocab:
                vocab[k] = len(vocab)
    V = len(vocab)
    poi = np.zeros((len(site_ids), V))
    for i, sid in enumerate(site_ids):
        for tok in poi_rows[sid].split(","):
            k, v = tok.split(":")[0].strip(), tok.split(":")[1].strip()
            if k in vocab:
                poi[i, vocab[k]] = float(v)
    # normalize POI rows (density vs count ambiguity -> L2 norm per site)
    nrm = np.linalg.norm(poi, axis=1, keepdims=True)
    nrm[nrm < 1e-9] = 1.0
    poi = poi / nrm
    sat_out = np.stack([sat[sat_idx[sid]] for sid in site_ids], axis=0)
    sat_out = sat_out / (np.linalg.norm(sat_out, axis=1, keepdims=True) + 1e-9)
    return site_ids, sat_out, poi, vocab


def load_residual():
    """Residual e = label - predict, reshaped [num_windows, N, pre_len]; also returns time axis."""
    ev = EVDataset(feature="volume", auxiliary="None", data_path=f"{BASE}/data/{CITY}_remove_zero/")
    ev.split_cross_validation(fold=6, total_fold=6, train_ratio=0.8, valid_ratio=0.1)
    raw = ev.feat                       # [T, N]
    T, N = raw.shape
    te_s, te_e = ev.test_start, ev.test_end
    num_windows = te_e - te_s - SEQ_L - PRE_LEN
    t_idx = pd.to_datetime(ev.time)[te_s + SEQ_L: te_s + SEQ_L + num_windows]  # forecast-start times per window
    lab = np.load(f"{BASE}/result/kairos_l1/label_plain_pre{PRE_LEN}_fold6.npy")
    pred = np.load(f"{BASE}/result/kairos_l1/predict_plain_pre{PRE_LEN}_fold6.npy")
    assert lab.shape[0] == num_windows * N, f"sample count mismatch {lab.shape[0]} vs {num_windows*N}"
    resid = (lab - pred).reshape(num_windows, N, PRE_LEN)   # [W, N, H]
    return resid, lab.reshape(num_windows, N, PRE_LEN), t_idx, N


def loo_ridge(X, y, alphas=RIDGE_ALPHAS):
    """Leave-one-out CV R^2 with alpha grid (unbiased, safe for N=62 high-dim)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    N = len(y)
    best_alpha, best_r2 = alphas[0], -1e18
    for a in alphas:
        r2s = []
        for i in range(N):
            tr = np.arange(N) != i
            m = X[tr].mean(0, keepdims=True); s = X[tr].std(0, keepdims=True); s[s < 1e-9] = 1.0
            r = Ridge(alpha=a).fit((X[tr] - m) / s, y[tr])
            p = r.predict((X[i:i + 1] - m) / s)[0]
            r2s.append((y[i] - p) ** 2)
        ss = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float(np.sum(r2s)) / ss if ss > 0 else np.nan
        if r2 > best_r2:
            best_r2, best_alpha = r2, a
    return best_r2, best_alpha


def perm_pval(X, y, n_perm=200, alphas=RIDGE_ALPHAS):
    """Permutation p-value: shuffle y, recompute LOO R^2."""
    base, _ = loo_ridge(X, y, alphas)
    rng = np.random.RandomState(0)
    hits = 0
    for _ in range(n_perm):
        r2, _ = loo_ridge(X, rng.permutation(y), alphas)
        if r2 >= base:
            hits += 1
    return base, hits / n_perm


def main():
    site_ids, sat, poi, vocab = load_static()
    resid, lab, t_idx, N = load_residual()
    W = resid.shape[0]
    print(f"== {CITY}: {N} sites, {W} test windows, POI vocab {len(vocab)} ==")

    # ---- static context matrix: sat | poi (PCA-reduced for N=62) ----
    X_full = np.concatenate([pca(sat, 15), pca(poi, 10)], axis=1)   # [N, 25]
    X_sat = pca(sat, 15)
    X_poi = pca(poi, 10)

    # site-level aggregates
    e1 = resid[:, :, -1]                             # [W, N] residual at last horizon step (pre_len-ahead)
    y1 = lab[:, :, -1]                               # [W, N] label at last horizon step (raw level)
    e_s = e1.mean(axis=0)                            # per-site mean residual
    mu_s = y1.mean(axis=0)                           # per-site mean raw load level
    sd_s = y1.std(axis=0)                            # per-site load scale

    # weekend/weekday split on forecast-start times
    is_wknd = np.asarray(t_idx.dayofweek >= 5)
    e_wknd = e1[is_wknd].mean(axis=0) if is_wknd.any() else np.full(N, np.nan)
    e_wkdy = e1[~is_wknd].mean(axis=0) if (~is_wknd).any() else np.full(N, np.nan)
    d_s = e_wknd - e_wkdy                             # weekend minus weekday residual differential

    print("\n[Counterfactual] context -> load LEVEL (must be HIGH = context knows busy-ness)")
    for name, X in [("sat", X_sat), ("poi", X_poi), ("sat+poi", X_full)]:
        r2, a = loo_ridge(X, mu_s)
        print(f"  mean raw load level mu_s  <- {name:<8} LOO R^2 = {r2:+.4f} (alpha={a})")

    print("\n[Probe A] context -> mean RESIDUAL (decisive: HIGH = increment exists)")
    for name, X in [("sat", X_sat), ("poi", X_poi), ("sat+poi", X_full)]:
        r2, a = loo_ridge(X, e_s)
        print(f"  mean residual e_s         <- {name:<8} LOO R^2 = {r2:+.4f} (alpha={a})")
    baseA, pvalA = perm_pval(X_full, e_s, n_perm=100)
    print(f"  [perm] sat+poi vs e_s: p={pvalA:.3f}")

    print("\n[Probe B] context -> (weekend - weekday) residual differential")
    for name, X in [("sat", X_sat), ("poi", X_poi), ("sat+poi", X_full)]:
        r2, a = loo_ridge(X, d_s)
        print(f"  wknd-wkdy residual diff   <- {name:<8} LOO R^2 = {r2:+.4f} (alpha={a})")
    baseB, pvalB = perm_pval(X_full, d_s, n_perm=100)
    print(f"  [perm] sat+poi vs d_s: p={pvalB:.3f}")

    print("\n[Probe C] per-time: e_{s,t} ~ ridge(context + calendar + context*calendar), time-split")
    # per-sample: flatten [W,N] -> [W*N]
    e_flat = e1.reshape(-1)
    hr = np.tile(t_idx.hour.values[:, None], (1, N)).reshape(-1)
    wk = np.tile(is_wknd[:, None], (1, N)).reshape(-1)
    # context per sample
    sat_flat = np.tile(sat[None, :, :], (W, 1, 1)).reshape(-1, sat.shape[1])
    poi_flat = np.tile(poi[None, :, :], (W, 1, 1)).reshape(-1, poi.shape[1])
    # calendar features: hour one-hot + weekend flag
    from sklearn.preprocessing import OneHotEncoder
    hr_oh = OneHotEncoder(sparse_output=False, categories="auto").fit_transform(hr[:, None])
    cal = np.concatenate([hr_oh, wk[:, None].astype(float)], axis=1)
    # split by time (first 70% windows)
    split = int(0.7 * W)
    tr = np.arange(split * N)
    te = np.arange(split * N, W * N)
    def ridge_te(A_tr, y_tr, A_te, y_te, a=10.0):
        m = A_tr.mean(0, keepdims=True); s = A_tr.std(0, keepdims=True); s[s < 1e-9] = 1.0
        r = Ridge(alpha=a).fit((A_tr - m) / s, y_tr)
        p = r.predict((A_te - m) / s)
        return 1 - float(((y_te - p) ** 2).sum()) / float(((y_te - y_te.mean()) ** 2).sum())
    # controls
    r_cal = ridge_te(cal[tr], e_flat[tr], cal[te], e_flat[te])
    r_cal_ctx = ridge_te(np.c_[cal[tr], sat_flat[tr], poi_flat[tr]], e_flat[tr],
                         np.c_[cal[te], sat_flat[te], poi_flat[te]], e_flat[te])
    # interaction: sat+poi reduced (PCA 10) x calendar (hour one-hot * wknd)
    ctx_pca = PCA(n_components=10).fit_transform(np.c_[sat_flat, poi_flat])
    inter = ctx_pca * wk[:, None]                    # context x weekend interaction
    r_cal_inter = ridge_te(np.c_[cal[tr], inter[tr]], e_flat[tr],
                           np.c_[cal[te], inter[te]], e_flat[te])
    print(f"  R^2 calendar-only           = {r_cal:+.4f}")
    print(f"  R^2 calendar+context        = {r_cal_ctx:+.4f}  (delta {r_cal_ctx-r_cal:+.4f})")
    print(f"  R^2 calendar+ctx*weekend    = {r_cal_inter:+.4f}  (delta {r_cal_inter-r_cal:+.4f})")


if __name__ == "__main__":
    main()
