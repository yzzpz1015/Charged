# -*- coding: utf-8 -*-
"""LLM reads POI as spatial functional semantics -> demand (one-shot probe).

Numeric POI and frozen-GPT-2 embedding both gave ~0.  The missing ingredient is
WORLD-KNOWLEDGE judgment: an LLM that reasons "restaurants+cafes => commercial
=> weekday daytime demand".  Qwen2-VL-2B-Instruct (local, used for sat captions)
does text-only generation: read each site's POI, output FUNCTION / BUSY(1-5) /
PATTERN, and we test that judgment against demand level / weekend-weekday
pattern / L1 residual.
"""
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

BASE = "/home/data/yzzpz/Charged"
CITY = "MEL"
MODEL = "/home/guest3/modelscope/Qwen2-VL-2B-Instruct"
OUT = f"{BASE}/tiles_dataset/{CITY}/embedding/z16/poi_llm_judge.json"
PCA_DIM = 10
ALPHAS = [0.1, 1.0, 10.0, 100.0]
DROP = {"other", "bench", "waste_basket", "shelter", "drinking_water", "post_box",
        "telephone", "bbq", "bicycle_parking", "toilets", "fountain", "whirlpool"}


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
    return site_ids, mu_dur, m_wkd, m_wke, ratio


def load_residual(site_ids):
    import sys
    sys.path.insert(0, BASE)
    from api.dataset.common import EVDataset
    ev = EVDataset(feature="volume", auxiliary="None", data_path=f"{BASE}/data/{CITY}_remove_zero/")
    ev.split_cross_validation(fold=6, total_fold=6, train_ratio=0.8, valid_ratio=0.1)
    N = ev.feat.shape[1]
    te_s, te_e = ev.test_start, ev.test_end
    num_windows = te_e - te_s - 72 - 6
    lab = np.load(f"{BASE}/result/kairos_l1/label_plain_pre6_fold6.npy")
    pred = np.load(f"{BASE}/result/kairos_l1/predict_plain_pre6_fold6.npy")
    resid = (lab - pred).reshape(num_windows, N, 6)
    return resid[:, :, 0].mean(axis=0)


def poi_text(site_ids):
    pdf = pd.read_csv(f"{BASE}/data/{CITY}_remove_zero/site_poi_1500m.csv")
    rows = {int(r["site_id"]): r["poi_str"] for _, r in pdf.iterrows()}
    out = []
    for s in site_ids:
        c = {}
        for tok in rows[s].split(","):
            k, cn = tok.split(":"); k = k.strip()
            if k not in DROP:
                c[k] = c.get(k, 0) + int(cn)
        items = sorted(c.items(), key=lambda kv: -kv[1])[:12]
        out.append(", ".join(f"{n} {k.replace('_', ' ')}" for k, n in items))
    return out


def judge(texts):
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(MODEL, trust_remote_code=True,
                                                   torch_dtype=torch.bfloat16).cuda().eval()
    prompt_tmpl = ("An EV charging station has these nearby places within 1.5 km: {poi}. "
                   "Based ONLY on these, reply with exactly 3 comma-separated items, no sentences: "
                   "area_function (residential/commercial/industrial/transport/mixed/leisure), "
                   "busyness_1_to_5, usage_pattern (weekday/weekend/balanced).")
    results = []
    for t in texts:
        prompt = prompt_tmpl.format(poi=t)
        msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device).to(torch.bfloat16)
        with torch.no_grad():
            torch.manual_seed(42)
            gen = model.generate(**inputs, max_new_tokens=48, do_sample=True, temperature=0.8, top_p=0.9)
        cap = processor.batch_decode(gen[:, inputs['input_ids'].shape[1]:],
                                     skip_special_tokens=True)[0].strip()
        results.append(cap)
    return results


def parse(ans):
    import re
    parts = [p.strip().lower() for p in ans.split(",")]
    f = parts[0].split()[0] if parts and parts[0] else None
    b = None
    if len(parts) > 1:
        m = re.search(r"\d", parts[1])
        b = int(m.group()) if m else None
    p = parts[2].split()[0] if len(parts) > 2 and parts[2] else None
    return f, b, p


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


def main():
    site_ids, mu_dur, m_wkd, m_wke, ratio = load_data()
    e_s = load_residual(site_ids)
    texts = poi_text(site_ids)
    print(f"== {CITY}: {len(site_ids)} sites, Qwen2-VL-2B judging POI ==\n")
    print("site 0 POI:", texts[0], "\n")

    ans = judge(texts)
    fs, bs, ps = [], [], []
    for a in ans:
        f, b, p = parse(a)
        fs.append(f); bs.append(b); ps.append(p)
    json.dump({"site_ids": site_ids, "function": fs, "busy": bs, "pattern": ps,
               "raw": ans}, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"[saved] {OUT}\n")

    print("sample judgments:")
    for i in [0, 1, 2, 5, 10]:
        print(f"  site {site_ids[i]}: FUNC={fs[i]} BUSY={bs[i]} PATTERN={ps[i]}")
    print()

    F = onehot(fs, sorted(set(f for f in fs if f)))
    P = onehot(ps, sorted(set(p for p in ps if p)))
    busy = np.array([b if b is not None else 3 for b in bs], float)

    # does the LLM's functional judgment predict observed demand?
    def report(name, X, y):
        Xp = pca(X, min(PCA_DIM, X.shape[1]))
        return f"{name:<24}{loo(Xp, y):+.3f}"

    print(f"{'judge feature':<24}{'->ratio':>9}{'->level':>9}{'->resid':>9}")
    print(report("FUNCTION one-hot", F, ratio), report("FUNCTION one-hot", F, mu_dur),
          report("FUNCTION one-hot", F, e_s))
    print(report("PATTERN one-hot", P, ratio), report("PATTERN one-hot", P, mu_dur),
          report("PATTERN one-hot", P, e_s))
    print(report("BUSY numeric", busy[:, None], ratio), report("BUSY numeric", busy[:, None], mu_dur),
          report("BUSY numeric", busy[:, None], e_s))
    # direct: does BUSY correlate with actual level?
    print(f"\n  BUSY vs mu_dur pearson = {np.corrcoef(busy, mu_dur)[0,1]:+.3f}")
    print(f"  BUSY vs ratio  pearson = {np.corrcoef(busy, ratio)[0,1]:+.3f}")


if __name__ == "__main__":
    main()
