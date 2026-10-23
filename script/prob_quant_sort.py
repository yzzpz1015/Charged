# -*- coding: utf-8 -*-
"""分位单调重排的免费后处理：排序 99 个分位前后，CRPS / 峰格 CRPS 对比。

分位头没有单调约束，SPO 有 26% 的格子 q90>q95，逆 CDF 插值直接失效。
逐格排序（monotone rearrangement）不改变每个 τ 的边际含义，只是把交叉拉直。

用法::

    python script/prob_quant_sort.py --cities SPO SZH MEL --horizon 24 --device 1
"""
import argparse
import os
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import prob_baseline as pb                                            # noqa: E402
import kairos_prob_eval as ke                                        # noqa: E402
from api.dataset.common import SPLIT_TRAIN                           # noqa: E402

U = 200
PEAK_MIN = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["SPO"])
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--out_root", default=os.path.join(BASE, "result", "kairos_l1_clean"))
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    j = args.horizon - 1

    for city in args.cities:
        ds, raw, starts = ke.load_city(city)
        N = raw.shape[1]
        st_te = np.asarray(starts["te"])
        keep = ke.clean_keep(raw, st_te)
        st_tr = ds.window_starts(SPLIT_TRAIN, ke.SEQ_L, ke.SEQ_L)
        keep_tr = ke.clean_keep(raw, st_tr)
        off = np.arange(ke.SEQ_L)[None, :]
        Ytr = np.transpose(raw[st_tr[:, None] + ke.SEQ_L + off], (0, 2, 1))
        thr = np.full(N, np.nan)
        for s in range(N):
            v = Ytr[:, s, :][keep_tr[:, s][:, None] & (Ytr[:, s, :] > 0)]
            if v.size >= PEAK_MIN:
                thr[s] = np.quantile(v, 0.95)

        qm = ke.build_kairos(city, "quant", args.out_root, dev, 99)
        if qm is None:
            print(f"{city}: 缺 quant 权重，跳过", flush=True); continue
        qs = np.arange(1, 100) / 100.0
        u2d = ke.strat_u(np.random.default_rng(0), U, N, shared=False)
        acc = {k: np.zeros(4) for k in ("raw", "sort")}   # n, crps, peak_n, peak_crps
        cross = 0.0
        ncell = 0
        for c0 in range(0, len(st_te), args.chunk):
            st = st_te[c0:c0 + args.chunk]
            X = raw[st[:, None] + off]
            X = np.transpose(X, (0, 2, 1)).reshape(-1, ke.SEQ_L).astype(np.float32)
            Q = ke._forward(qm, X, dev).reshape(len(st), N, ke.SEQ_L, 99)
            Q = Q[:, :, j, :]
            for m in range(len(st)):
                mk = keep[c0 + m]
                if not mk.any():
                    continue
                y = raw[st[m] + ke.SEQ_L + j]
                ok = np.isfinite(thr)
                pk = np.zeros(N, bool)
                pk[ok] = y[ok] > thr[ok]
                cross += np.mean(Q[m][:, 89] > Q[m][:, 94])
                ncell += 1
                for nm, qv in (("raw", Q[m]), ("sort", np.sort(Q[m], axis=-1))):
                    smp = ke.quant_to_samples_ind(qv, qs, u2d)
                    c = pb.crps_fair(smp[:, mk], y[mk])
                    acc[nm][0] += mk.sum(); acc[nm][1] += c.sum()
                    pm = mk & pk
                    if pm.any():
                        cp = pb.crps_fair(smp[:, pm], y[pm])
                        acc[nm][2] += pm.sum(); acc[nm][3] += cp.sum()
        del qm
        torch.cuda.empty_cache()
        print(f"\n===== {city} H={args.horizon} =====", flush=True)
        print(f"  q90>q95 交叉率（窗口平均） {cross / max(ncell, 1):.3f}", flush=True)
        for nm in ("raw", "sort"):
            n, c, pn, pc = acc[nm]
            print(f"  {nm:<5} 平均CRPS {c / max(n, 1):.4f}  峰格CRPS {pc / max(pn, 1):.4f}  "
                  f"(峰格 {int(pn)}/{int(n)} = {100 * pn / max(n, 1):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
