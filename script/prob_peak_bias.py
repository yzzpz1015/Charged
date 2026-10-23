# -*- coding: utf-8 -*-
"""点预测的零/峰偏差诊断：误差到底落在哪一类格子上。

回答三个问题：
  1. 零格（目标=0）上模型给了多少非零量？
  2. 峰格（该站测试期非零值的 top 10%）被压掉了多少（收缩比、回归斜率）？
  3. "多零少峰"的窗口里，峰是不是更压不住？

数据来源：训练脚本落盘的 predict_plain_pre72_fold6.npy（测试期全部 72 个 horizon，
形状 [M*N, 72]），与 --clean 评测用同一张站×窗掩码。

用法：
    python script/prob_peak_bias.py --cities LOA JHB MEL SPO SZH
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'script'))
sys.path.insert(0, BASE)

from kairos_prob_eval import load_city, clean_keep, SEQ_L  # noqa: E402

OUT = os.path.join(BASE, 'result', 'prob_analysis')


def summarize(name, y, p):
    """一组格子的偏差摘要。"""
    n = y.size
    if n == 0:
        return dict(group=name, n=0)
    y, p = y.ravel(), p.ravel()
    nz = y > 0
    # 峰格外的量：# 预测在真值为 0 时给了多少
    return dict(group=name, n=n,
                target_mean=float(y.mean()), pred_mean=float(p.mean()),
                bias=float((p - y).mean()),
                ratio=float(p.mean() / y.mean()) if y.mean() > 1e-9 else np.nan,
                mae=float(np.abs(p - y).mean()),
                corr=float(np.corrcoef(y, p)[0, 1]) if n > 2 and y.std() > 0 and p.std() > 0 else np.nan,
                slope=float(np.cov(p, y)[0, 1] / np.var(p)) if n > 2 and p.std() > 0 else np.nan,
                target_p99=float(np.quantile(y, .99)), pred_p99=float(np.quantile(p, .99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cities', nargs='+', default=['LOA', 'JHB', 'MEL', 'SPO', 'SZH'])
    ap.add_argument('--out_root', default=os.path.join(BASE, 'result', 'kairos_l1_clean'))
    ap.add_argument('--zero_frac', type=float, default=0.7,
                    help='窗口内目标为 0 的小时占比 ≥ 此值，算"多零少峰"窗口')
    ap.add_argument('--peak_q', type=float, default=0.9,
                    help='峰格阈值：该站测试期非零目标值的分位点')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    all_rows = []
    for city in args.cities:
        ds, raw, starts = load_city(city)
        N = raw.shape[1]
        M = len(starts['te'])
        keep = clean_keep(raw, starts['te'])                      # [M,N]
        pnpy = os.path.join(args.out_root, city, f'predict_plain_pre{SEQ_L}_fold6.npy')
        if not os.path.exists(pnpy):
            print(f'[skip] {city}: 缺 {pnpy}')
            continue
        P = np.load(pnpy)
        if P.shape != (M * N, SEQ_L):
            print(f'[skip] {city}: predict 形状 {P.shape} != {(M*N, SEQ_L)}')
            continue
        P = P.reshape(M, N, SEQ_L).astype(np.float32)

        # 目标 [M,N,72]
        Y = np.stack([raw[starts['te'] + SEQ_L + h] for h in range(SEQ_L)], axis=2).astype(np.float32)
        m = keep[..., None]                                        # 只保留干净格子

        # 每站峰阈值：测试期非零目标值的分位点
        thr = np.full(N, np.inf, np.float32)
        for j in range(N):
            v = Y[keep[:, j], j, :]
            v = v[v > 0]
            if v.size:
                thr[j] = np.quantile(v, args.peak_q)

        zf = (Y == 0).mean(2)                                      # [M,N] 窗口零率
        is_peak = (Y >= thr[None, :, None]) & (Y > 0)
        is_zero = Y == 0
        is_mid = (~is_peak) & (~is_zero)

        # 误差贡献：top 目标格的平方误差占比
        se = ((P - Y) ** 2) * m
        yf = Y[keep]
        top = yf >= np.quantile(yf[yf > 0], 0.9) if (yf > 0).any() else np.zeros_like(yf, bool)
        sse_top = float(se[keep][top].sum()) / max(float(se[keep].sum()), 1e-9)

        rows = []
        for label, grp in [('全部干净格', np.ones_like(Y, bool)),
                           ('零格 y=0', is_zero), ('中间格', is_mid), ('峰格', is_peak)]:
            g = grp & m
            rows.append(summarize(label, Y[g], P[g]))
        # 多零少峰窗口
        dense = (zf >= args.zero_frac)                              # [M,N]
        for label, grp in [('多零窗口 · 全部', np.repeat(dense[..., None], SEQ_L, 2)),
                           ('多零窗口 · 峰格', np.repeat(dense[..., None], SEQ_L, 2) & is_peak),
                           ('多零窗口 · 零格', np.repeat(dense[..., None], SEQ_L, 2) & is_zero)]:
            g = grp & m
            rows.append(summarize(label, Y[g], P[g]))
        # 按 horizon
        for h in [0, 5, 11, 23, 47, 71]:
            g = keep
            rows.append(summarize(f'H={h+1} 全部', Y[:, :, h][g], P[:, :, h][g]))
            gp = g & is_peak[:, :, h]
            rows.append(summarize(f'H={h+1} 峰格', Y[:, :, h][gp], P[:, :, h][gp]))

        df = pd.DataFrame(rows)
        df.insert(0, 'city', city)
        path = os.path.join(OUT, f'peak_bias_{city}.csv')
        df.to_csv(path, index=False)
        all_rows.append(df)

        print(f'\n===== {city}  干净格 {keep.mean()*100:.1f}%  '
              f'零格占比 {(Y[keep]==0).mean()*100:.1f}%  '
              f'峰格占比 {is_peak[keep].mean()*100:.1f}%  '
              f'峰格贡献 SSE {sse_top*100:.0f}% =====')
        print(f'{"分组":<18}{"n":>9}{"目标均值":>10}{"预测均值":>10}{"偏差":>9}'
              f'{"比例":>7}{"MAE":>9}{"相关":>7}{"斜率":>7}')
        for r in rows:
            if not r.get('n'):
                continue
            print(f'{r["group"]:<18}{r["n"]:>9d}{r["target_mean"]:>10.2f}'
                  f'{r["pred_mean"]:>10.2f}{r["bias"]:>9.2f}{r["ratio"]:>7.2f}'
                  f'{r["mae"]:>9.2f}{r["corr"]:>7.3f}{r["slope"]:>7.3f}')

    if all_rows:
        pool = pd.concat(all_rows, ignore_index=True)
        pool.to_csv(os.path.join(OUT, 'peak_bias_all.csv'), index=False)
        print(f'\n[saved] {OUT}/peak_bias_all.csv')


if __name__ == '__main__':
    main()
