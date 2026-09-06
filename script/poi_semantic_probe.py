# -*- coding: utf-8 -*-
"""POI-as-spatial-functional-semantics probe (LLM reads POI, not numeric vector).

The numeric POI count vector -> busy-ness is ~0.  The user's hypothesis: the
signal is in the FUNCTIONAL SEMANTICS of the POI ("restaurants + cafes + bars"
=> commercial district), which a count vector throws away (categories are
one-hot orthogonal) but an LLM's word embeddings recover (restaurant~cafe~bar
are all 'commercial').

Probe: turn each site's POI into a natural-language functional description,
read it with GPT-2 (full forward, mean-pool), and test the semantic vector ->
{mean duration level, weekend/weekday pattern}.  Compare to the numeric-vector
baseline (~0).
"""
import numpy as np
import pandas as pd
import torch
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

BASE = "/home/data/yzzpz/Charged"
CITY = "MEL"
GPT2 = f"{BASE}/api/model/weights/gpt2_model"
PCA_DIM = 15
ALPHAS = [0.1, 1.0, 10.0, 100.0]

# pure furniture / unclassified -> drop from the functional description
DROP = {"other", "bench", "waste_basket", "shelter", "drinking_water", "post_box",
        "telephone", "bbq", "bicycle_parking", "toilets", "fountain", "whirlpool"}


def load_labels():
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
    return site_ids, mu_dur, m_wkd, m_wke, ratio


def poi_texts(site_ids):
    """Per-site functional POI description (natural language)."""
    pdf = pd.read_csv(f"{BASE}/data/{CITY}_remove_zero/site_poi_1500m.csv")
    rows = {int(r["site_id"]): r["poi_str"] for _, r in pdf.iterrows()}
    texts = []
    for s in site_ids:
        counts = {}
        for tok in rows[s].split(","):
            k, cn = tok.split(":")
            k = k.strip()
            if k in DROP:
                continue
            counts[k] = counts.get(k, 0) + int(cn)
        # rank by count, keep top 12 functional types
        items = sorted(counts.items(), key=lambda kv: -kv[1])[:12]
        parts = [f"{n} {k.replace('_', ' ')}{'s' if n != 1 else ''}" for k, n in items]
        text = "The area within 1.5 km of this charging station contains: " + ", ".join(parts) + "."
        texts.append(text)
    return texts


def gpt2_embed(texts, device):
    tok = GPT2Tokenizer.from_pretrained(GPT2)
    tok.pad_token = tok.eos_token
    model = GPT2Model.from_pretrained(GPT2).to(device).eval()
    enc = tok(texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state                 # [N, S, 768]
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(1) / mask.sum(1)               # mean over real tokens
    return pooled.cpu().numpy()


def pca(X, n):
    return X if X.shape[1] <= n else PCA(n_components=n).fit_transform(X)


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def loo_ridge(X, y):
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


def load_residual(site_ids):
    """Per-site mean 1-step residual of L1 (the decisive forecast-increment target)."""
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
    e_s = resid[:, :, 0].mean(axis=0)          # per-site mean 1-step residual
    assert len(e_s) == len(site_ids)
    return e_s


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    site_ids, mu_dur, m_wkd, m_wke, ratio = load_labels()
    e_s = load_residual(site_ids)
    texts = poi_texts(site_ids)
    print(f"== {CITY}: {len(site_ids)} sites ==\n")
    print("example POI text (site 0):", texts[0], "\n")

    sem = gpt2_embed(texts, device)
    print(f"GPT-2 semantic vector: {sem.shape}")

    # numeric POI baseline (same functional categories, count vector, l2-normed)
    pdf = pd.read_csv(f"{BASE}/data/{CITY}_remove_zero/site_poi_1500m.csv")
    rows = {int(r["site_id"]): r["poi_str"] for _, r in pdf.iterrows()}
    vocab = {}
    for s in site_ids:
        for tok in rows[s].split(","):
            k = tok.split(":")[0].strip()
            if k not in DROP and k not in vocab:
                vocab[k] = len(vocab)
    num = np.zeros((len(site_ids), len(vocab)))
    for i, s in enumerate(site_ids):
        for tok in rows[s].split(","):
            k, cn = tok.split(":"); k = k.strip()
            if k in vocab:
                num[i, vocab[k]] = float(cn)
    num = l2(num)

    print(f"\n{'repr':<20}{'R2(level dur)':>13}{'R2(weekday)':>12}{'R2(weekend)':>12}{'R2(ratio)':>11}{'R2(resid)':>11}")
    for name, X, dim in [("numeric POI (count)", num, 10),
                         ("GPT-2 semantic", l2(sem), 15)]:
        Xp = pca(X, dim)
        print(f"{name:<20}{loo_ridge(Xp, mu_dur):>13.3f}{loo_ridge(Xp, m_wkd):>12.3f}"
              f"{loo_ridge(Xp, m_wke):>12.3f}{loo_ridge(Xp, ratio):>11.3f}{loo_ridge(Xp, e_s):>11.3f}")


if __name__ == "__main__":
    main()
