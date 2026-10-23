# -*- coding: utf-8 -*-
"""峰格上尾校准 + q50/均值点预测对比（便宜探针，不训练）。

回答两个问题：

1. **上尾到底窄不窄**：按模型自己预测的 q99 分箱，看每箱里真值超过 q99 的
   比例。若最高箱的超过率远大于 1%，说明上尾欠覆盖（真问题）；若各箱都≈1%，
   说明"峰值被压"主要是点预测=条件均值的天然性质，不是分布坏了。
   * 不要直接按"事后真值是峰"筛格看覆盖率：按结果筛会天然压低覆盖率，
     即使模型完美校准也会显示欠覆盖。
2. **点预测能不能靠换分位数改善**：同一模型输出的 q50（中位数）vs 主干点预测
   （条件均值），在零格/正格/峰格上的 MAE 与偏差。

用法::

    python script/prob_tail_calib.py --cities MEL --horizon 24
"""
import argparse
import os
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import kairos_prob_eval as ke                                        # noqa: E402
from api.dataset.common import SPLIT_TRAIN                           # noqa: E402


def station_p95(raw, st, keep):
    """逐站训练期正值 p95（所有 horizon、clean 格）。"""
    off = np.arange(ke.SEQ_L)[None, :]
    Y = np.transpose(raw[st[:, None] + ke.SEQ_L + off], (0, 2, 1))   # [M,N,72]
    m = keep[:, :, None] & (Y > 0)
    N = raw.shape[1]
    thr = np.full(N, np.nan, np.float64)
    for j in range(N):
        v = Y[:, j, :][m[:, j, :]]
        if v.size >= 100:
            thr[j] = np.quantile(v, 0.95)
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["MEL"])
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out_root", default=os.path.join(BASE, "result", "kairos_l1_clean"))
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    j = args.horizon - 1
    off = np.arange(ke.SEQ_L)[None, :]

    for city in args.cities:
        ds, raw, starts = ke.load_city(city)
        N = raw.shape[1]
        st_te = np.asarray(starts["te"])
        keep = ke.clean_keep(raw, st_te)
        st_tr = ds.window_starts(SPLIT_TRAIN, ke.SEQ_L, ke.SEQ_L)
        keep_tr = ke.clean_keep(raw, st_tr)
        thr = station_p95(raw, st_tr, keep_tr)

        qm = ke.build_kairos(city, "quant", args.out_root, dev, 99)
        km = ke.build_kairos(city, "plain", args.out_root, dev, 99)
        if qm is None or km is None:
            print(f"{city}: 缺权重，跳过", flush=True); continue
        Qs, Ps = [], []
        for c in range(0, len(st_te), args.chunk):
            st = st_te[c:c + args.chunk]
            X = raw[st[:, None] + off]
            X = np.transpose(X, (0, 2, 1)).reshape(-1, ke.SEQ_L).astype(np.float32)
            q = ke._forward(qm, X, dev).reshape(len(st), N, ke.SEQ_L, 99)
            p = ke._forward(km, X, dev).reshape(len(st), N, ke.SEQ_L)
            Qs.append(q[:, :, j, :])
            Ps.append(p[:, :, j])
        Q = np.concatenate(Qs, 0)          # [M,N,99]
        P = np.concatenate(Ps, 0)          # [M,N]
        del qm, km
        torch.cuda.empty_cache()

        y = raw[st_te + ke.SEQ_L + j]      # [M,N]
        m = keep & np.isfinite(y)
        yv = y[m]
        q50, q90, q95, q99 = (Q[..., k][m] for k in (49, 89, 94, 98))
        pv = P[m]
        pk = np.zeros_like(y, bool)
        ok = np.isfinite(thr)
        pk[:, ok] = y[:, ok] > thr[ok][None, :]
        pk &= m
        zpk = pk[m]

        print(f"\n===== {city}  H={args.horizon}  测试格 {m.sum()}  "
              f"峰格 {zpk.sum()} ({100 * zpk.mean():.1f}%) =====", flush=True)
        # ---- 1. 按预测 q99 分箱的尾校准（不按结果筛）----
        print("按预测 q99 分箱：箱内 P(y > q99)（理想≈0.01）")
        order = np.argsort(q99)
        bins = np.array_split(order, 5)
        for b, idx in enumerate(bins):
            if len(idx) == 0:
                continue
            exc = np.mean(yv[idx] > q99[idx])
            print(f"  q99 箱{b + 1}/5  n={len(idx):>7d}  q99范围 "
                  f"[{q99[idx].min():8.2f},{q99[idx].max():8.2f}]  "
                  f"超过率 {exc:.4f}", flush=True)
        # ---- 2. 全局名义覆盖率 ----
        print(f"全局覆盖率  q50/90/95/99 = "
              f"{np.mean(yv <= q50):.3f}/{np.mean(yv <= q90):.3f}/"
              f"{np.mean(yv <= q95):.3f}/{np.mean(yv <= q99):.3f}", flush=True)
        # 分位头无单调约束：先查交叉，再做逐格单调重排（排序 99 个分位）。
        print(f"分位交叉率  q50>q90 {np.mean(q50 > q90):.3f}  "
              f"q90>q95 {np.mean(q90 > q95):.3f}  "
              f"q95>q99 {np.mean(q95 > q99):.3f}  "
              f"q99最小值 {q99.min():.2f}", flush=True)
        Q.sort(axis=-1)
        s50, s90, s95, s99 = (Q[..., k][m] for k in (49, 89, 94, 98))
        print(f"单调重排后覆盖率 q50/90/95/99 = "
              f"{np.mean(yv <= s50):.3f}/{np.mean(yv <= s90):.3f}/"
              f"{np.mean(yv <= s95):.3f}/{np.mean(yv <= s99):.3f}", flush=True)
        # ---- 3. 峰格上的 q99 覆盖（会因按结果筛而偏低，仅作描述）----
        if zpk.any():
            yp, qp = yv[zpk], q99[zpk]
            print(f"峰格：P(y<=q99)={np.mean(yp <= qp):.3f}  "
                  f"q99/真值 中位={np.median(qp / np.maximum(yp, 1e-6)):.3f}  "
                  f"真值>q99 的比例={np.mean(yp > qp):.3f}", flush=True)
        # ---- 4. q50 vs 主干点预测（条件均值）----
        print(f"{'口径':<10}{'n':>8}{'MAE(均值)':>11}{'MAE(q50)':>11}"
              f"{'bias(均值)':>12}{'bias(q50)':>11}")
        masks = {
            "全部": np.ones_like(yv, bool),
            "真零": yv == 0,
            "真正": yv > 0,
            "峰格": zpk,
        }
        for nm, mm in masks.items():
            if not mm.any():
                continue
            print(f"{nm:<10}{mm.sum():>8d}"
                  f"{np.mean(np.abs(pv[mm] - yv[mm])):>11.3f}"
                  f"{np.mean(np.abs(q50[mm] - yv[mm])):>11.3f}"
                  f"{np.mean(pv[mm] - yv[mm]):>12.3f}"
                  f"{np.mean(q50[mm] - yv[mm]):>11.3f}", flush=True)


if __name__ == "__main__":
    main()
