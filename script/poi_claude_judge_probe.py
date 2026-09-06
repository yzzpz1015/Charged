# -*- coding: utf-8 -*-
"""Claude (the assistant itself) reads POI as spatial functional semantics.

The user's point: instead of the weak local Qwen2-VL-2B, use MY OWN world
knowledge to judge each site's FUNCTION / BUSY(1-5) / PATTERN from its POI,
then test that judgment -> demand.  Judgments are hardcoded below (hand-read
from the per-site functional POI dump, ignoring pure parking/furniture).
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

BASE = "/home/data/yzzpz/Charged"
CITY = "MEL"
OUT = f"{BASE}/tiles_dataset/{CITY}/embedding/z16/poi_claude_judge.json"
PCA_DIM = 10
ALPHAS = [0.1, 1.0, 10.0, 100.0]

# hand judgment: site_id -> (function, busy 1..5, pattern)
JUDGE = {
    0:  ("mixed", 2, "balanced"),
    1:  ("commercial", 3, "weekday"),
    2:  ("transport", 1, "balanced"),
    3:  ("commercial", 4, "weekday"),
    4:  ("commercial", 3, "weekday"),
    5:  ("commercial", 3, "weekday"),
    6:  ("commercial", 4, "weekday"),
    7:  ("commercial", 4, "weekday"),
    8:  ("commercial", 3, "weekday"),
    9:  ("commercial", 2, "weekday"),
    10: ("commercial", 2, "weekday"),
    11: ("commercial", 3, "weekday"),
    12: ("transport", 1, "balanced"),
    13: ("commercial", 3, "weekday"),
    14: ("transport", 2, "balanced"),
    15: ("mixed", 3, "balanced"),
    16: ("commercial", 2, "weekday"),
    17: ("commercial", 5, "weekday"),
    18: ("transport", 2, "balanced"),
    19: ("mixed", 2, "balanced"),
    20: ("industrial", 2, "weekday"),
    21: ("commercial", 3, "weekday"),
    22: ("commercial", 3, "weekday"),
    23: ("commercial", 4, "weekday"),
    24: ("industrial", 1, "balanced"),
    25: ("commercial", 4, "weekday"),
    26: ("residential", 2, "weekday"),
    27: ("commercial", 3, "weekday"),
    28: ("commercial", 4, "weekday"),
    29: ("commercial", 3, "weekday"),
    30: ("commercial", 3, "weekday"),
    31: ("commercial", 3, "weekday"),
    32: ("transport", 1, "balanced"),
    33: ("commercial", 2, "weekday"),
    34: ("transport", 2, "balanced"),
    35: ("commercial", 2, "weekday"),
    36: ("commercial", 2, "weekday"),
    37: ("commercial", 4, "weekday"),
    38: ("mixed", 3, "weekend"),
    39: ("commercial", 3, "weekday"),
    40: ("transport", 1, "balanced"),
    41: ("transport", 1, "balanced"),
    42: ("commercial", 2, "weekday"),
    43: ("commercial", 3, "weekday"),
    45: ("mixed", 2, "balanced"),
    46: ("mixed", 2, "weekday"),
    47: ("commercial", 3, "weekday"),
    48: ("mixed", 2, "balanced"),
    49: ("commercial", 3, "weekday"),
    50: ("commercial", 5, "weekday"),
    51: ("commercial", 2, "weekday"),
    52: ("residential", 1, "balanced"),
    53: ("residential", 2, "weekend"),
    54: ("commercial", 4, "weekday"),
    55: ("commercial", 4, "weekday"),
    56: ("commercial", 2, "weekday"),
    57: ("commercial", 3, "weekday"),
    58: ("commercial", 2, "weekday"),
    59: ("commercial", 3, "weekday"),
    60: ("commercial", 2, "weekday"),
    61: ("transport", 1, "balanced"),
    62: ("commercial", 2, "weekday"),
}


def load_data():
    data = f"{BASE}/data/{CITY}_remove_zero"
    dur = pd.read_csv(f"{data}/duration.csv", index_col=0)
    vol = pd.read_csv(f"{data}/volume.csv", index_col=0)
    site_ids = vol.columns.astype(int).tolist()
    ts = pd.to_datetime(dur.index)
    wkd = np.asarray(ts.dayofweek < 5)
    m_wkd = dur.iloc[wkd].astype(float).mean(0)[[str(s) for s in site_ids]].to_numpy()
    m_wke = dur.iloc[~wkd].astype(float).mean(0)[[str(s) for s in site_ids]].to_numpy()
    mu_dur = dur.astype(float).mean(0)[[str(s) for s in site_ids]].to_numpy()
    ratio = m_wke / (m_wkd + 1e-6)
    return site_ids, mu_dur, ratio


def load_residual(site_ids):
    import sys
    sys.path.insert(0, BASE)
    from api.dataset.common import EVDataset
    ev = EVDataset(feature="volume", auxiliary="None",
                   data_path=f"{BASE}/data/{CITY}_remove_zero/")
    ev.split_cross_validation(fold=6, total_fold=6, train_ratio=0.8, valid_ratio=0.1)
    N = ev.feat.shape[1]
    te_s, te_e = ev.test_start, ev.test_end
    num_windows = te_e - te_s - 72 - 6
    lab = np.load(f"{BASE}/result/kairos_l1/label_plain_pre6_fold6.npy")
    pred = np.load(f"{BASE}/result/kairos_l1/predict_plain_pre6_fold6.npy")
    resid = (lab - pred).reshape(num_windows, N, 6)
    return resid[:, :, 0].mean(axis=0)


def onehot(vals, vocab):
    idx = {v: i for i, v in enumerate(vocab)}
    M = np.zeros((len(vals), len(vocab)))
    for i, v in enumerate(vals):
        if v in idx:
            M[i, idx[v]] = 1.0
    return M


def pca(X, n):
    return X if X.shape[1] <= n else PCA(n_components=n).fit_transform(X)


def loo(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    N = len(y); best = -1e18
    for a in ALPHAS:
        r2s = []
        for i in range(N):
            tr = np.arange(N) != i
            m = X[tr].mean(0, keepdims=True); s = X[tr].std(0, keepdims=True); s[s < 1e-9] = 1.0
            r = Ridge(alpha=a).fit((X[tr] - m) / s, y[tr])
            r2s.append((y[i] - r.predict((X[i:i + 1] - m) / s)[0]) ** 2)
        ss = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float(np.sum(r2s)) / ss if ss > 0 else np.nan
        if r2 > best:
            best = r2
    return best


def perm_p(X, y, n=2000, seed=0):
    """permutation p-value on LOO R2 (does judged feature beat random assignment)."""
    rng = np.random.default_rng(seed)
    obs = loo(X, y)
    cnt = 0
    for _ in range(n):
        yp = rng.permutation(y)
        if loo(X, yp) >= obs:
            cnt += 1
    return (cnt + 1) / (n + 1)


def main():
    site_ids, mu_dur, ratio = load_data()
    e_s = load_residual(site_ids)
    fs = [JUDGE[s][0] for s in site_ids]
    bs = np.array([JUDGE[s][1] for s in site_ids], float)
    ps = [JUDGE[s][2] for s in site_ids]

    json.dump({"site_ids": site_ids, "function": fs, "busy": [int(x) for x in bs],
               "pattern": ps}, open(OUT, "w"), indent=1)
    print(f"== Claude judges POI: {CITY} {len(site_ids)} sites ==\n")
    from collections import Counter
    print("FUNCTION dist:", dict(Counter(fs)))
    print("PATTERN  dist:", dict(Counter(ps)))
    print("BUSY     dist:", dict(Counter(int(b) for b in bs)))
    print()

    F = onehot(fs, sorted(set(fs)))
    P = onehot(ps, sorted(set(ps)))

    def report(name, X, y):
        Xp = pca(X, min(PCA_DIM, X.shape[1]))
        return f"{name:<22}{loo(Xp, y):+.3f}"

    print(f"{'judge feature':<22}{'->ratio':>9}{'->level':>9}{'->resid':>9}")
    print(report("FUNCTION one-hot", F, ratio), report("FUNCTION one-hot", F, mu_dur),
          report("FUNCTION one-hot", F, e_s))
    print(report("PATTERN one-hot", P, ratio), report("PATTERN one-hot", P, mu_dur),
          report("PATTERN one-hot", P, e_s))
    print(report("BUSY numeric", bs[:, None], ratio), report("BUSY numeric", bs[:, None], mu_dur),
          report("BUSY numeric", bs[:, None], e_s))

    print(f"\n  BUSY vs mu_dur pearson = {np.corrcoef(bs, mu_dur)[0,1]:+.3f}")
    print(f"  BUSY vs ratio  pearson = {np.corrcoef(bs, ratio)[0,1]:+.3f}")
    pb = perm_p(bs[:, None], mu_dur)
    pr = perm_p(bs[:, None], ratio)
    print(f"  BUSY->level perm-p = {pb:.3f}   BUSY->ratio perm-p = {pr:.3f}")


if __name__ == "__main__":
    main()
