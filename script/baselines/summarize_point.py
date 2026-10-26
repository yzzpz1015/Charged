# -*- coding: utf-8 -*-
"""汇总点预测指标（metrics_point，MAE/RMSE/R²）。

* 点预测口径：分位模型取 τ=0.5；采样模型取样本中位数（见 common.py）。
* 每城每个 H 的 MAE 单位为 raw kWh/h，跨城不可直接平均；
  五城平均用 nMAE = MAE / mean(y)（同一 clean cells 上的 y 均值）。
* 测试窗少于全量的方法（CSDI / TSDiff、DiffSTG 的 SZH）打 ``*``。

用法::

    python3 script/baselines/summarize_point.py --seeds 0,1,2
    python3 script/baselines/summarize_point.py --seeds 0,1,2 --metric r2
    python3 script/baselines/summarize_point.py --seeds 0,1,2 --markdown
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402

CITIES = ["LOA", "JHB", "MEL", "SPO", "SZH"]
HORIZONS = ["H24", "H48", "H72"]
JS = (23, 47, 71)
EXPECT_W = {"LOA": 2125, "JHB": 2125, "MEL": 1781, "SPO": 2125, "SZH": 2125}

LABEL = {
    "itransformer": "7 iTransformer",
    "stgcn": "8 STGCN",
    "stgcn_nograph": "9 STGCN-nograph",
    "gwnet": "10 Graph WaveNet",
    "tft": "11 TFT",
    "deepar_isqf": "12 DeepAR-ISQF",
    "deepar": "12b DeepAR-StudentT",
    "mqcnn": "13 MQ-CNN",
    "csdi_official": "14 CSDI (official)",
    "timegrad": "15 TimeGrad",
    "tsdiff": "16 TSDiff",
    "diffstg": "17 DiffSTG",
    "chronos_bolt": "18 Chronos-Bolt",
    "moirai": "19 Moirai",
    "patchtst_quant_pre72": "6 PatchTST+quant",
    "unist_pool": "20 UniTS-FT+pool",
    "opencity": "21 OpenCity-Plus",
}
ORDER = list(LABEL)


def load(tag, seeds):
    root = os.path.join(BASE, "result", "baselines",
                        "metrics" + (f"_{tag}" if tag else ""))
    out = {}
    for s in seeds:
        for p in glob.glob(os.path.join(root, "*", "*", f"H*_seed{s}.json")):
            with open(p) as f:
                d = json.load(f)
            if "mae" not in d:
                continue
            h = d["horizon"]
            h = f"H{int(h)}" if isinstance(h, (int, np.integer)) else h
            out[(d["method"], d["city"], h, s)] = d
    return out


def y_means():
    """每个 (city, H) 上 clean cells 的 y 均值，用于 nMAE。"""
    cache = os.path.join(BASE, "result", "baselines", "metrics_point",
                         "_y_mean.json")
    if os.path.exists(cache):
        with open(cache) as f:
            d = json.load(f)
        return {tuple(k.split("|")): v for k, v in d.items()}
    out = {}
    for c in CITIES:
        _, raw, starts, keep, _ = C.load_city(c)
        for h, j in zip(HORIZONS, JS):
            y = raw[starts["te"][:, None] + C.SEQ_L + j][:, 0, :]
            m = keep["te"]
            out[(c, h)] = float(y[m].mean())
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as f:
        json.dump({f"{c}|{h}": v for (c, h), v in out.items()}, f, indent=1)
    return out


def fmt(vals):
    a = np.asarray(vals, float)
    if len(a) == 1:
        return f"{a[0]:.4f}"
    return f"{a.mean():.4f}±{a.std():.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="point")
    ap.add_argument("--metric", choices=["mae", "rmse", "r2"], default="mae")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    tab = load(args.tag, seeds)
    ym = y_means()
    methods = sorted({k[0] for k in tab},
                     key=lambda m: (ORDER.index(m) if m in ORDER else 99, m))

    print(f"point metric = {args.metric}, seeds = {seeds}")
    if args.metric == "r2":
        print("R² 为 pooled clean cells；SZH 由站间水平差主导，"
              "跨方法比较请看 MAE/nMAE。")
    norm_name = {"mae": "nMAE", "rmse": "nRMSE", "r2": "五城平均 R²"}[
        args.metric]
    for h in HORIZONS:
        print(f"\n===== {h} =====")
        if args.markdown:
            print("| 方法 | " + " | ".join(CITIES)
                  + f" | {norm_name} |")
            print("|---|" + "---:|" * len(CITIES) + "---:|")
        else:
            print("method".ljust(26) + "".join(f"{c:>22}" for c in CITIES)
                  + f"   {norm_name}-avg")
        for m in methods:
            nm, cells = [], []
            for c in CITIES:
                recs = [tab[(m, c, h, s)] for s in seeds if (m, c, h, s) in tab]
                if not recs:
                    cells.append("—")
                    continue
                vals = [float(r[args.metric]) for r in recs]
                cell = fmt(vals)
                if any(int(r.get("n_test_windows", -1)) < EXPECT_W[c]
                       for r in recs):
                    cell += "*"
                cells.append(cell)
                if args.metric in ("mae", "rmse"):
                    nm.extend([v / ym[(c, h)] for v in vals])
                elif args.metric == "r2":
                    nm.extend(vals)
            ncity = sum(1 for c in CITIES
                        if any((m, c, h, s) in tab for s in seeds))
            avg = f"{np.mean(nm):.4f}" if nm else "—"
            if nm and ncity < len(CITIES):
                avg += f"({ncity})"
            name = LABEL.get(m, m)
            if args.markdown:
                print(f"| {name} | " + " | ".join(cells) + f" | {avg} |")
            else:
                print(name.ljust(26) + "".join(f"{v:>22}" for v in cells)
                      + f"   {avg:>8}")
    print("\n* = 测试窗少于该城全量；nMAE/nRMSE = 指标 / mean(y)"
          "（同一 clean cells）")
    print("点预测定义：分位模型 τ=0.5；采样模型样本中位数。")


if __name__ == "__main__":
    main()
