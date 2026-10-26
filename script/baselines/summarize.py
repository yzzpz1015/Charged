# -*- coding: utf-8 -*-
"""汇总 result/baselines/metrics → 主表（方法 × 城市 × H）。

用法::

    python3 script/baselines/summarize.py                 # seed 0，crpsΣ
    python3 script/baselines/summarize.py --metric crps_sum_norm --seed 0
    python3 script/baselines/summarize.py --markdown      # 生成 docs 用 md 表
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(BASE, "result", "baselines", "metrics")
CITIES = ["LOA", "JHB", "MEL", "SPO", "SZH"]
HORIZONS = ["H24", "H48", "H72"]
EXPECT_W = {"LOA": 2125, "JHB": 2125, "MEL": 1781, "SPO": 2125, "SZH": 2125}


def load(seed, metric, root=None):
    out = {}
    for p in glob.glob(os.path.join(root or ROOT, "*", "*", f"H*_seed{seed}.json")):
        with open(p) as f:
            d = json.load(f)
        h = d["horizon"]
        # UniTS 写的是整数 24/48/72，其余方法写 "H24"/"H48"/"H72"；统一成后者。
        if isinstance(h, (int, np.integer)):
            h = f"H{int(h)}"
        out[(d["method"], d["city"], h)] = (d[metric], d)
    return out


def mark(v, d, city):
    """测试窗数少于协议全量时打 *，冒烟结果不会伪装成全量。"""
    nw = int(d.get("n_test_windows", -1))
    return f"{v:.4f}" + ("*" if nw >= 0 and nw < EXPECT_W.get(city, 0) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metric", default="crps_norm_sum")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--methods", default="", help="逗号分隔，默认全部")
    ap.add_argument("--seeds", default="",
                    help="逗号分隔；给出时输出 mean±std（按可用 seed 聚合）")
    ap.add_argument("--tag", default="",
                    help="读 metrics_{tag}/；如 point 读 metrics_point（点预测重评）")
    args = ap.parse_args()
    root = os.path.join(BASE, "result", "baselines",
                        "metrics" + (f"_{args.tag}" if args.tag else ""))
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",")]
        tabs = {s: load(s, args.metric, root) for s in seeds}
        tab = tabs[seeds[0]]
        methods = sorted({k[0] for t in tabs.values() for k in t})
        if args.methods:
            keep = set(args.methods.split(","))
            methods = [m for m in methods if m in keep]
        w = max(len(m) for m in methods) + 2
        print(f"metric = {args.metric}, seeds = {seeds} (mean±std)")
        print("method".ljust(w) + "  ".join(f"{c:>26}" for c in CITIES))
        for m in methods:
            row = m.ljust(w)
            for c in CITIES:
                cells = []
                for h in HORIZONS:
                    vals = [tabs[s][(m, c, h)] for s in seeds if (m, c, h) in tabs[s]]
                    if not vals:
                        cells.append(f"{'—':>8}")
                    elif len(vals) == 1:
                        cells.append(f"{mark(*vals[0], c):>8}")
                    else:
                        arr = np.array([v[0] for v in vals], dtype=float)
                        cell = f"{arr.mean():.4f}±{arr.std():.4f}"
                        if any(int(v[1].get('n_test_windows', -1)) < EXPECT_W.get(c, 0)
                               for v in vals):
                            cell += "*"
                        cells.append(cell)
                row += "  ".join(cells) + "    "
            print(row)
        print("\n* = 至少一个 seed 的测试窗数少于该城全量")
        return
    tab = load(args.seed, args.metric, root)
    methods = sorted({k[0] for k in tab})
    if args.methods:
        keep = set(args.methods.split(","))
        methods = [m for m in methods if m in keep]

    if args.markdown:
        print(f"| 方法 | 城市 | H24 | H48 | H72 |")
        print("|---|---|---|---|---|")
        for m in methods:
            for c in CITIES:
                if not any((m, c, h) in tab for h in HORIZONS):
                    continue
                vals = [mark(*tab[(m, c, h)], c) if (m, c, h) in tab else "—"
                        for h in HORIZONS]
                print(f"| {m} | {c} | " + " | ".join(vals) + " |")
        return

    w = max(len(m) for m in methods) + 2
    print(f"metric = {args.metric}, seed = {args.seed}")
    print("method".ljust(w) + "  ".join(f"{c:>22}" for c in CITIES))
    for m in methods:
        row = m.ljust(w)
        for c in CITIES:
            vals = []
            for h in HORIZONS:
                v = tab.get((m, c, h))
                vals.append(mark(*v, c) if v else "   —  ")
            row += "  ".join(f"{v:>6}" for v in vals) + "    "
        print(row)
    # 缺什么
    missing = [(m, c) for m in methods for c in CITIES
               if not any((m, c, h) in tab for h in HORIZONS)]
    if missing:
        print("\n缺失：", ", ".join(f"{m}/{c}" for m, c in missing))
    print("\n* = 测试窗数少于该城全量（扩散方法按固定子集、或冒烟未重跑）")


if __name__ == "__main__":
    main()
