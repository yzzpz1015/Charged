# -*- coding: utf-8 -*-
"""POI judge kit: produce a pasteable prompt + ingest an EXTERNAL LLM's output.

The probe asks "does a strong LLM's semantic reading of POI carry demand signal?"
The judge must be a genuine external black box (not the assistant hardcoding
judgments, which risks its own priors).  So this script only:

  python script/poi_judge_kit.py --make                 # write prompt.txt + jsonl
  python script/poi_judge_kit.py --ingest <out.json>    # probe the LLM's output

User feeds `script/poi_judge_prompt.txt` (POI + task) to ANY strong LLM, gets a
JSON array [{site_id,function,busy,pattern}], saves it, and --ingest runs the
same LOO-ridge probe against {level, weekend/weekday ratio, L1 residual}.
"""
import json
import sys
import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, "/home/data/yzzpz/Charged/script")
from poi_claude_judge_probe import load_data, load_residual, onehot, pca, loo  # noqa: E402

BASE = "/home/data/yzzpz/Charged"
CITY = "MEL"
PROMPT = f"{BASE}/script/poi_judge_prompt.txt"
JSONL = f"{BASE}/tiles_dataset/{CITY}/embedding/z16/poi_for_judge.jsonl"
DROP = {"other", "bench", "waste_basket", "shelter", "drinking_water", "post_box",
        "telephone", "bbq", "bicycle_parking", "toilets", "fountain", "whirlpool"}

TASK = """An EV charging station has these nearby places within 1.5 km. Based ONLY on
these places, judge the station's demand profile. For each station return:
  function : residential | commercial | industrial | transport | mixed | leisure
  busy     : integer 1 (quietest) .. 5 (busiest)
  pattern  : weekday | weekend | balanced
Reply with NOTHING but a JSON array, one object per station, with keys exactly
site_id (int), function (str), busy (int), pattern (str). No prose, no markdown."""


def site_ids():
    vol = pd.read_csv(f"{BASE}/data/{CITY}_remove_zero/volume.csv", index_col=0)
    return vol.columns.astype(int).tolist()


def poi_text(sids):
    pdf = pd.read_csv(f"{BASE}/data/{CITY}_remove_zero/site_poi_1500m.csv")
    rows = {int(r["site_id"]): r["poi_str"] for _, r in pdf.iterrows()}
    out = []
    for s in sids:
        c = {}
        for tok in rows[s].split(","):
            k, cn = tok.split(":"); k = k.strip()
            if k not in DROP:
                c[k] = c.get(k, 0) + int(cn)
        items = sorted(c.items(), key=lambda kv: -kv[1])[:12]
        out.append(", ".join(f"{n} {k.replace('_', ' ')}" for k, n in items))
    return out


def make():
    sids = site_ids()
    texts = poi_text(sids)
    lines = ["=== EV CHARGING STATION POI JUDGING TASK ===\n", TASK, "\nSTATIONS:\n"]
    for s, t in zip(sids, texts):
        lines.append(f"  site {s}: {t}")
    lines.append("\nOUTPUT (JSON array, %d objects):\n" % len(sids))
    open(PROMPT, "w").write("\n".join(lines))
    with open(JSONL, "w") as f:
        for s, t in zip(sids, texts):
            f.write(json.dumps({"site_id": s, "poi": t}, ensure_ascii=False) + "\n")
    print(f"[wrote] {PROMPT}")
    print(f"[wrote] {JSONL}  ({len(sids)} sites)")


def probe(judge):
    sids = site_ids()
    mu_dur, ratio = load_data()
    e_s = load_residual(sids)
    fs = [judge[s][0] for s in sids]
    bs = np.array([judge[s][1] for s in sids], float)
    ps = [judge[s][2] for s in sids]
    print("FUNCTION", dict(Counter(fs)))
    print("PATTERN ", dict(Counter(ps)))
    print("BUSY    ", dict(Counter(int(b) for b in bs)))
    F = onehot(fs, sorted(set(fs)))
    P = onehot(ps, sorted(set(ps)))

    def rep(n, X, y):
        Xp = pca(X, min(10, X.shape[1]))
        return f"{n:<20}{loo(Xp, y):+.3f}"

    print(f"\n{'feature':<20}{'->ratio':>9}{'->level':>9}{'->resid':>9}")
    print(rep("FUNCTION", F, ratio), rep("FUNCTION", F, mu_dur), rep("FUNCTION", F, e_s))
    print(rep("PATTERN", P, ratio), rep("PATTERN", P, mu_dur), rep("PATTERN", P, e_s))
    print(rep("BUSY", bs[:, None], ratio), rep("BUSY", bs[:, None], mu_dur), rep("BUSY", bs[:, None], e_s))
    print(f"\nBUSY vs mu_dur pearson = {np.corrcoef(bs, mu_dur)[0,1]:+.3f}")
    print(f"BUSY vs ratio  pearson = {np.corrcoef(bs, ratio)[0,1]:+.3f}")


def ingest(path):
    rows = json.load(open(path))
    judge = {r["site_id"]: (r["function"], int(r["busy"]), r["pattern"]) for r in rows}
    missing = set(site_ids()) - set(judge)
    if missing:
        print(f"WARN: missing sites {sorted(missing)}")
    print(f"== ingesting external LLM output: {len(judge)} sites ==\n")
    probe(judge)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--make":
        make()
    elif len(sys.argv) >= 3 and sys.argv[1] == "--ingest":
        ingest(sys.argv[2])
    else:
        print(__doc__)
