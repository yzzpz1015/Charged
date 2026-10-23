# -*- coding: utf-8 -*-
"""删掉纯直线段后，按 2:1:3 重新划分 train/valid/test，并把划分落盘。

规则
----
1. **纯直线小时**：该站在一个"逐小时增量完全相同且非零、连续 >=72 小时"的段里
   （与 ``detect_straight_windows.py`` 同一判定）。
2. **整行判为合成**：该小时有 >=50% 的站在纯直线段里。低于 50% 时多数站是真实数据，
   删整行会丢掉真实样本；这只对"全城同时被摊平"的时段（LOA 9/17~9/30、MEL 6/14~6/19）
   才成立。
3. 可用行 = 非停摆（``pb.load`` 已把停摆行置 NaN）且非合成行。
4. 在可用行上枚举所有合法的 144h 窗口（72h 历史 + 72h 目标，整段连续可用），
   按**窗口起点的时间顺序**切 2:1:3。
   为什么按窗口而不是按行：JHB 按行切时 valid 段正好落在被直线打碎的区间里，
   只剩 5 个窗口（无法选模型）；按窗口切则每段都保足份额。
   两种切法的份额都是 2:1:3，但按窗口切不会因时间碎片化而塌掉某一段。

边界怎么取
----------
``b1 = st[i1] + 72``、``b2 = st[i2] + 72``（st = 合法窗口起点，i1/i2 = 2/6、3/6 分位）。
这样"目标起点 i+72 落在 train 区间"恰好等价于"i 属于前 i1 个合法起点"，
窗口数精确等于 2:1:3，不需要近似。

注意
----
新的 test 段与旧划分不同；**旧模型权重必须重训**，否则新 test 里的行在旧训练段里
出现过（例如 LOA 新 test 从 6/24 开始，旧 train 覆盖到 6/25），属于直接泄漏。

用法::

    python script/clean_split.py [--cities LOA JHB ...]
输出 ``result/prob_baseline/clean_split_<city>.npz``（starts_train/valid/test、
``split_of_row``、``usable``、``synth_row``）并打印汇总表。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prob_baseline as pb          # noqa: E402

TOL = 1e-6
SEQ = PRE = 72
L = SEQ + PRE
MIN_STRETCH = 72        # 等增量段至少这么长才判为合成（与 detect_straight_windows.py 一致）
SYNTH_FRAC = 0.5
OUT = os.path.join(pb.BASE, 'result/prob_baseline')


def straight_hours(x):
    """[T] bool：该小时落在"等增量且非零、连续 >=72h"的段里。"""
    T = len(x)
    m = np.zeros(T, bool)
    if T < 2:
        return m
    d = np.diff(x)
    fin = np.isfinite(d)
    eq = np.zeros(len(d), bool)
    eq[1:] = fin[1:] & fin[:-1] & (np.abs(d[1:] - d[:-1]) <= TOL)
    if not eq.any():
        return m
    ix = np.flatnonzero(np.diff(np.r_[False, eq, False]))
    a, e = ix[0::2], ix[1::2]
    h = e - a + 2
    for s, hh in zip(a[h >= MIN_STRETCH], h[h >= MIN_STRETCH]):
        g = d[s - 1] if s > 0 else d[0]
        if abs(g) > TOL:                     # 增量恒为 0 的段不算（可能是真停摆）
            m[s:s + hh] = True
    return m


def clean_split(y0, synth_frac=SYNTH_FRAC):
    T, N = y0.shape
    sh = np.stack([straight_hours(y0[:, j]) for j in range(N)], 1)
    f = sh.mean(1)
    synth = f >= synth_frac
    base = np.isfinite(y0).all(1)
    ok = base & ~synth

    run = np.zeros(T + 1, np.int32)          # run[i] = 从 i 起连续可用的小时数
    for i in range(T - 1, -1, -1):
        run[i] = run[i + 1] + 1 if ok[i] else 0
    st = np.flatnonzero(run[:max(0, T - L + 1)] >= L)

    P = len(st)
    i1 = int(P * 2 / 6)
    i2 = i1 + int(P * 1 / 6)
    # 行切点取"边界窗口的目标起点"。目标块是 [i+72, i+144)，跨切点的窗口两边都不算，
    # 会被自然丢掉；下面用与 split_window_starts 相同的整块判定报实际窗口数。
    split_of_row = np.full(T, -1, np.int8)
    b1 = int(st[i1] + SEQ) if P else T
    b2 = int(st[i2] + SEQ) if P else T
    rows = np.arange(T)
    split_of_row[ok & (rows < b1)] = 0
    split_of_row[ok & (rows >= b1) & (rows < b2)] = 1
    split_of_row[ok & (rows >= b2)] = 2

    parts = []
    for sid in (0, 1, 2):
        is_id = (split_of_row == sid).astype(np.int64)
        cs = np.r_[0, np.cumsum(is_id)]
        keep = (cs[st + L] - cs[st + SEQ] == PRE)     # 目标块整块属于本段
        parts.append(st[keep])
    return dict(starts=parts, split_of_row=split_of_row, usable=ok,
                synth=synth, frac=f, base=base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cities', nargs='*', default=pb.CITIES)
    args = ap.parse_args()
    print(f"{'城':>4} {'原可用h':>7} {'删合成h':>7} {'剩可用h':>7} | "
          f"{'总窗':>6} {'train':>6} {'valid':>6} {'test':>6} | "
          f"{'train止':>10} {'valid止':>10} {'test止':>10}", flush=True)
    for city in args.cities:
        y0, t0 = pb.load(city)
        r = clean_split(y0)
        parts = r['starts']
        ends = [t0[min(p[-1] + L, len(t0) - 1)] if len(p) else None for p in parts]
        fmt = lambda x: str(x)[:10] if x is not None else '-'
        print(f"{city:>4} {r['base'].sum():>7} {r['synth'].sum():>7} "
              f"{r['usable'].sum():>7} | {len(np.concatenate(parts)):>6} "
              f"{len(parts[0]):>6} {len(parts[1]):>6} {len(parts[2]):>6} | "
              f"{fmt(ends[0]):>10} {fmt(ends[1]):>10} {fmt(ends[2]):>10}", flush=True)
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(
            f"{OUT}/clean_split_{city}.npz",
            starts_train=parts[0], starts_valid=parts[1], starts_test=parts[2],
            split_of_row=r['split_of_row'], usable=r['usable'], synth=r['synth'])
    print(f"\n划分已写入 {OUT}/clean_split_<city>.npz", flush=True)


if __name__ == '__main__':
    main()
