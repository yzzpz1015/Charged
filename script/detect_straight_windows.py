# -*- coding: utf-8 -*-
"""全数据集扫描：目标块（72h）是不是「纯直线」（逐小时增量完全相同且非零）或「全平」（增量恒为 0）。

背景：SPO st24 那张图里真值是几乎直线缓步上升，不像是真实充电量。追下去发现是
站点年度总量被摊到 8760 小时生成的伪序列。这里给出它在测试窗口里到底占多少。

判定：
  纯直线 A —— 目标块 71 个逐小时增量两两相等（容差 1e-6，浮点存储误差），且增量非零；
              并且这段等增量在原始序列里连续延伸 >= 72 小时（避免把真实的缓慢斜率误判）。
  全平   B —— 目标块 72 个值完全相同（增量恒为 0）。
输出：
  result/prob_baseline/straight_windows.csv   逐站逐窗标签
  stdout 汇总
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kairos_prob_eval as ke          # noqa: E402
import prob_baseline as pb             # noqa: E402

TOL = 1e-6
L = 72
OUT = os.path.join(pb.BASE, 'result/prob_baseline/straight_windows.csv')


def marks(x):
    """返回 (lin, flat, n_big)：逐小时是否落在 >=72h 的等增量段里（分非零/全零），及这种段的个数。"""
    T = len(x)
    lin = np.zeros(T, bool)
    flat = np.zeros(T, bool)
    if T < 2:
        return lin, flat, 0
    d = np.diff(x)
    fin = np.isfinite(d)
    eq = np.zeros(len(d), bool)
    eq[1:] = fin[1:] & fin[:-1] & (np.abs(d[1:] - d[:-1]) <= TOL)
    if not eq.any():
        return lin, flat, 0
    ix = np.flatnonzero(np.diff(np.r_[False, eq, False]))
    a, e = ix[0::2], ix[1::2]
    h = e - a + 2                       # 覆盖的小时数
    big = h >= L
    for s, hh in zip(a[big], h[big]):
        g = d[s - 1] if s > 0 else d[0]
        (flat if abs(g) <= TOL else lin)[s:s + hh] = True
    return lin, flat, int(big.sum())


def count_full(mask, starts):
    """starts 里有多少个 72h 块被 mask 完全覆盖。"""
    c = np.r_[0, np.cumsum(mask.astype(np.int64))]
    return int((c[starts + L] - c[starts] == L).sum())


rows = []
print(f"{'城':>4} {'站':>5} | {'测试窗':>7} {'A纯直线':>7} {'A%':>6} {'B全平':>6} {'B%':>6} "
      f"| {'有A的站':>8} {'有段站':>7} | {'>=72h直线段':>11}", flush=True)
print('-' * 108, flush=True)

for city in ['LOA', 'JHB', 'MEL', 'SPO', 'SZH', 'AMS']:
    y0, t0 = pb.load(city)
    N = y0.shape[1]
    ds, raw, starts = ke.load_city(city)
    w = np.asarray(starts['te'])
    b = w + ke.SEQ_L                    # 目标块起点
    nW = nA = nB = 0
    stA = stSeg = nseg = 0
    for j in range(N):
        x = y0[:, j]
        lin, flat, nb = marks(x)
        nseg += nb
        stSeg += int(nb > 0)
        ok = (b + L) <= len(x)
        bb = b[ok]
        if len(bb) == 0:
            continue
        cA = count_full(lin, bb)
        cB = count_full(flat, bb)
        nW += len(bb)
        nA += cA
        nB += cB
        stA += int(cA > 0)
        for k, s in enumerate(bb):
            lab = 'A' if lin[s:s + L].all() else ('B' if flat[s:s + L].all() else '')
            if lab:
                rows.append(dict(city=city, station=j, win_start=int(w[ok][k]),
                                 target_start=int(s), kind=lab,
                                 value=float(x[s]), slope=float(np.diff(x[s:s + L])[0])))
    print(f"{city:>4} {N:>5} | {nW:>7} {nA:>7} {100*nA/max(nW,1):>5.2f}% {nB:>6} "
          f"{100*nB/max(nW,1):>5.2f}% | {stA:>8} {stSeg:>7} | {nseg:>11}", flush=True)

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
print(f'\n逐窗明细 -> {OUT}  （{len(df)} 行）', flush=True)
if len(df):
    g = df.groupby(['city', 'kind']).size().unstack(fill_value=0)
    g['站数'] = df.groupby('city')['station'].nunique()
    print(g.to_string())
