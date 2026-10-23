# -*- coding: utf-8 -*-
"""时序里到底有多少零——六个城市的直接统计。

口径与主实验完全一致：`prob_baseline.load`（停摆行置 NaN）+ `split_rows`
（可用行按计数切 2/1/3）+ `live_cols`（剔训练期退化站，只用训练期统计）。
这样这里数出来的零，就是模型真正看到的那批零，不是另一套数据。

回答四个问题：
  1. 总体零率有多少？三个 split 分别多少？
  2. 零在**站间**怎么分布？有没有"永远全是零"或"从来不零"的站？
  3. 零在**一天之内**怎么分布？（昼夜规律）
  4. 零是"零星一个"还是"一停一大片"？（连续零段长度）
"""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import prob_baseline as pb                                          # noqa: E402

CITIES = ["LOA", "JHB", "MEL", "AMS", "SPO", "SZH"]

# 连续零段的分档（小时）。1 = 孤立的一格；24+ = 停了一整天以上。
RUN_EDGES = [(1, 1, "1h"), (2, 5, "2-5h"), (6, 23, "6-23h"),
             (24, 71, "24-71h"), (72, 10 ** 9, ">=72h")]


def runs_of_zero(col):
    """一条列的**连续零段**长度数组（非零或 NaN 都算打断）。

    向量化：把「是零」化成 0/1，两侧补 0，看跳变位置就能切出所有段。
    """
    zc = (np.isfinite(col) & (col == 0)).astype(np.int8)
    d = np.diff(np.concatenate(([0], zc, [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return ends - starts                                   # 两端：各段的长度


def main():
    print("=" * 108)
    print("时序中的零：六城直接统计（口径 = 主实验的 prob_baseline）")
    print("=" * 108)
    print(f"\n{'城':<5}{'站数':>6}{'可用行':>8}{'总零率':>9}"
          f"{'train':>8}{'valid':>8}{'test':>8}   停摆丢弃行")
    store = {}
    for c in CITIES:
        y0, t0 = pb.load(c)
        ok, trm, vam, tem, u, span = pb.split_rows(y0, t0)
        cols, smean = pb.live_cols(y0, trm)
        y = y0[:, cols]
        valid = np.isfinite(y)
        z = (y == 0) & valid

        allm = np.ones(len(t0), bool)          # [T] 全行（rate 吃的是**行**掩码）

        def rate(mask):
            m = mask[:, None] & valid          # [T,N]
            return z[m].sum() / max(1, m.sum())

        store[c] = dict(y=y, z=z, valid=valid, trm=trm, vam=vam, tem=tem,
                        t0=t0, n=len(cols))
        print(f"{c:<5}{len(cols):>6}{int(valid.sum()):>8}{rate(allm):>9.3f}"
              f"{rate(trm):>8.3f}{rate(vam):>8.3f}{rate(tem):>8.3f}"
              f"   {span['drop_h']:>5}")

    # ---- 2. 零在站间分布 ----
    print("\n" + "-" * 108)
    print("零率在**站间**的分布：把每站自己的零率算出来，看有多少站落在各档")
    print(f"{'城':<5}{'=0%(从不零)':>12}{'<10%':>8}{'10-50%':>9}"
          f"{'50-90%':>9}{'>90%':>8}{'=100%(恒零)':>13}{'零率中位':>10}")
    for c in CITIES:
        d = store[c]
        m = d['valid'].sum(0) > 0
        sr = np.where(m, d['z'].sum(0) / np.maximum(1, d['valid'].sum(0)), np.nan)[m]
        n = len(sr)
        b = [np.mean(sr == 0), np.mean((sr > 0) & (sr < .1)),
             np.mean((sr >= .1) & (sr < .5)), np.mean((sr >= .5) & (sr < .9)),
             np.mean((sr >= .9) & (sr < 1)), np.mean(sr == 1)]
        print(f"{c:<5}{b[0]:>11.1%}{b[1]:>8.1%}{b[2]:>9.1%}"
              f"{b[3]:>9.1%}{b[4]:>8.1%}{b[5]:>13.1%}{np.median(sr):>10.3f}")
    print("（各档是占该城站数的比例；中位=每站零率的中位数）")

    # ---- 3. 零在一天之内 ----
    print("\n" + "-" * 108)
    print("零率 vs 一天中的小时（0-23），看昼夜规律")
    for c in CITIES:
        d = store[c]
        hod = d['t0'].hour.values
        row = []
        for h in range(24):
            m = (hod == h)[:, None] & d['valid']
            row.append(d['z'][m].sum() / max(1, m.sum()))
        row = np.array(row)
        bar = "".join("▁▂▃▄▅▆▇█"[min(7, int(v * 8 / max(0.001, row.max())))]
                      for v in row)
        print(f"{c:<5}{bar}  峰{row.max():.3f}@{row.argmax()}h "
              f"谷{row.min():.3f}@{row.argmin()}h")

    # ---- 4. 连续零段长度 ----
    print("\n" + "-" * 108)
    print("零的**连续性**：所有连续零段按长度分档（占所有零段的比例 / 占总零格的比例）")
    print(f"{'城':<5}" + "".join(f"{nm:>22}" for _, _, nm in RUN_EDGES))
    for c in CITIES:
        d = store[c]
        # 只在**可用行**上按时间顺序数零段：把停摆行（NaN）当"打断"，
        # 否则停摆造成的缺口会被误当成一次长零段。
        T, N = d['y'].shape
        all_runs = np.concatenate([runs_of_zero(d['y'][:, j]) for j in range(N)])
        tot = all_runs.sum()
        cells = []
        for lo, hi, _ in RUN_EDGES:
            m = (all_runs >= lo) & (all_runs <= hi)
            cells.append((m.sum() / len(all_runs), all_runs[m].sum() / tot))
        line = f"{c:<5}" + "".join(f"{a:>10.1%}/{b:>10.1%}"
                                   for a, b in cells)
        print(line)
    print("（左=零段条数占比，右=零格数占比；如 '>=72h' 右值大 = 绝大多数零集中在长时间停摆式空闲）")


if __name__ == "__main__":
    main()
