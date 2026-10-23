# -*- coding: utf-8 -*-
"""Weekday/weekend breakdown of retrieval-vs-L1 explanatory power & gain.

For a given pre_len, loads the SAME fold-fold test windows used by both
train_kairos_l1 (L1 plain) and kairos_retrieval_teacher (kNN teacher), maps each
window's FORECAST-START time to weekday (dayofweek<5) vs weekend (>=5), and
reports pooled R2 + MAE per split for L1 and the retrieval teacher, plus the
gain = retrieval - L1.

Alignment: both scripts build identical flattened samples [M*N, pre_len] with
row r = window (r//N), site (r%N); forecast start of window w = time[te_s + w + seq_l].
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from api.dataset.common import (EVDataset, SPLIT_TRAIN,
                                SPLIT_VALID, SPLIT_TEST)


def build_samples(raw, starts, seq_l, pre_len):
    """按**窗口起点数组**建样本（起点来自 EVDataset.window_starts，新划分的唯一取窗入口）。

    旧版按连续区间 ``end - start - seq_l - pre_len`` 枚举，新口径下是错的（2026-10-04 改）：
    目标只需落在本段、历史可伸进上一段，所以合法窗口**比连续枚举多**；AMS 段内还夹着
    停摆行，连续枚举会把跨停摆缺口的窗也当成合法。行 r = 窗口 (r//N)、站 (r%N)。
    """
    M = len(starts)
    assert M > 0
    N = raw.shape[1]
    X = np.zeros((M, N, seq_l), dtype=np.float32)
    Y = np.zeros((M, N, pre_len), dtype=np.float32)
    for i, base in enumerate(starts):
        base = int(base)
        X[i] = raw[base:base + seq_l, :].T
        Y[i] = raw[base + seq_l:base + seq_l + pre_len, :].T
    return X.reshape(-1, seq_l), Y.reshape(-1, pre_len)


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    ss = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - float(((y_true - y_pred) ** 2).sum()) / ss if ss > 0 else np.nan


def mae(y_true, y_pred):
    return float(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--city', default='MEL')
    ap.add_argument('--pre_len', type=int, required=True)
    ap.add_argument('--seq_l', type=int, default=72)
    ap.add_argument('--fold', type=int, default=6)
    ap.add_argument('--total_fold', type=int, default=6)
    ap.add_argument('--data_path', default='data/')
    ap.add_argument('--data_suffix', default='_remove_const',
                        help='城市目录后缀：_remove_const(默认,已剔恒值站) / _remove_zero')
    ap.add_argument('--l1_dir', default='result/kairos_l1')
    ap.add_argument('--ret_dir', default='result/kairos_retrieval')
    args = ap.parse_args()

    data_dir = f"{args.data_path}{args.city}{args.data_suffix}/"
    ev = EVDataset(feature='volume', auxiliary='None', data_path=data_dir)
    ev.split_cross_validation(fold=args.fold, total_fold=args.total_fold,
                              train_ratio=0.8, valid_ratio=0.1)
    raw = ev.feat
    N = raw.shape[1]
    time = ev.time  # pd.DatetimeIndex aligned to raw rows

    st_te = ev.window_starts(SPLIT_TEST, args.seq_l, args.pre_len)
    _, Y_test = build_samples(raw, st_te, args.seq_l, args.pre_len)
    Me = len(Y_test) // N

    # per-window forecast-start day-of-week -> broadcast to per-row mask
    # 预测原点 = 起点 + seq_l；起点取 st_te[w]（**不是** te_s + w —— 新口径下窗口起点
    # 不再是从 test_start 起的连续整数，历史可以伸进上一段）。
    fcast_start = pd.to_datetime([time[st_te[w] + args.seq_l] for w in range(Me)])
    is_weekend_w = (fcast_start.dayofweek.values >= 5)  # [Me] bool
    is_weekend = np.repeat(is_weekend_w, N)             # [Me*N] bool
    print(f"[split] Me={Me} windows, weekend windows={int(is_weekend_w.sum())}, "
          f"weekday windows={int((~is_weekend_w).sum())}")
    print(f"[split] forecast-start range: {fcast_start[0]} .. {fcast_start[-1]}")

    lab_path = os.path.join(args.ret_dir, f"label_teacher_k10_pre{args.pre_len}_fold{args.fold}.npy")
    if not os.path.exists(lab_path):
        lab_path = os.path.join(args.l1_dir, f"label_plain_pre{args.pre_len}_fold{args.fold}.npy")
    lab = np.load(lab_path)
    l1 = np.load(os.path.join(args.l1_dir, f"predict_plain_pre{args.pre_len}_fold{args.fold}.npy"))
    ret = np.load(os.path.join(args.ret_dir, f"predict_teacher_k10_pre{args.pre_len}_fold{args.fold}.npy"))

    # shape check (predictions must match labels exactly)
    assert lab.shape == l1.shape == ret.shape, (lab.shape, l1.shape, ret.shape)

    def report(mask, name):
        n = int(mask.sum())
        m = {
            'L1 R2': r2(lab[mask], l1[mask]),
            'L1 MAE': mae(lab[mask], l1[mask]),
            'ret R2': r2(lab[mask], ret[mask]),
            'ret MAE': mae(lab[mask], ret[mask]),
        }
        m['gain R2'] = m['ret R2'] - m['L1 R2']
        print(f"[{name}] n_rows={n}")
        for k, v in m.items():
            print(f"    {k:>10}: {v:+.4f}")
        return m

    print(f"\n=== pre_len={args.pre_len} ===")
    report(np.ones_like(is_weekend, dtype=bool), 'overall')
    report(~is_weekend, 'weekday')
    report(is_weekend, 'weekend')


if __name__ == '__main__':
    main()
