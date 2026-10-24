# -*- coding: utf-8 -*-
"""#18 Chronos-Bolt / #19 Moirai：零样本概率预测（不训练，确定性）。

口径：每站独立喂 72h 历史，直接预测未来 72 步；与训练方法共用同一批测试窗、
同一 clean mask、同一 ``StreamingMetrics``。

* Chronos-Bolt 原生只有 9 个分位（0.1…0.9），0.01/0.99 是管线插值/夹断；
  PICP90 只作诊断，表注写明。
* Moirai 用 ``uni2ts`` 的 ``MoiraiForecast``，100 条样本 → 经验分位（99 个），
  与分位类方法走同一条 ``update_quantiles`` 路径。

用法::

    python3 script/baselines/run_zeroshot.py --method chronos_bolt --city MEL --device 0
    python3 script/baselines/run_zeroshot.py --method moirai --city MEL --device 0
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402

WEIGHTS = os.path.join(BASE, "external", "weights")


def run_chronos(args, raw, starts, keep, smean, qs_np, dev):
    from chronos import ChronosBoltPipeline
    path = args.ckpt or os.path.join(WEIGHTS, "chronos-bolt-small")
    pipe = ChronosBoltPipeline.from_pretrained(path, device_map=str(dev),
                                               torch_dtype=torch.float32)
    met = C.make_metrics(args.city, raw, starts, keep, smean, qs_np,
                         with_diag=not args.no_diag)
    M, N = len(starts), raw.shape[1]
    off = np.arange(C.SEQ_L)
    chunk = max(1, args.batch_series // N)
    t0 = time.time()
    for c0 in range(0, M, chunk):
        c1 = min(c0 + chunk, M)
        ctx = raw[starts[c0:c1, None] + off]                  # [w,72,N]
        ctx = ctx.transpose(0, 2, 1).reshape(-1, C.SEQ_L)     # [w*N,72]
        q, _ = pipe.predict_quantiles(
            torch.tensor(ctx, dtype=torch.float32), prediction_length=C.PRE_LEN,
            quantile_levels=list(map(float, qs_np)))
        q = q.float().cpu().numpy().reshape(c1 - c0, N, C.PRE_LEN, len(qs_np))
        met.update_quantiles(c0, q)
        if (c0 // chunk) % 10 == 0:
            print(f"  {c1}/{M} 窗  {time.time()-t0:.0f}s", flush=True)
    return met


def run_moirai(args, ds, raw, starts, keep, smean, qs_np, dev):
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    path = args.ckpt or os.path.join(WEIGHTS, "moirai-1.0-R-small")
    module = MoiraiModule.from_pretrained(path).to(dev)
    model = MoiraiForecast(
        prediction_length=C.PRE_LEN, target_dim=1, feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0, context_length=C.SEQ_L,
        module=module, patch_size="auto", num_samples=C.M_SAMPLE)
    # 不传 device 时 PyTorchPredictor 默认落 cuda:0，--device 会被忽略
    predictor = model.create_predictor(batch_size=args.batch_series,
                                       device=str(dev))

    met = C.make_metrics(args.city, raw, starts, keep, smean, qs_np,
                         with_diag=not args.no_diag)
    M, N = len(starts), raw.shape[1]
    off = np.arange(C.SEQ_L)
    chunk = max(1, args.batch_windows)
    t0 = time.time()
    for c0 in range(0, M, chunk):
        c1 = min(c0 + chunk, M)
        entries, meta = [], []
        for m in range(c0, c1):
            start = pd.Period(ds.time[int(starts[m])], freq="h")
            for n in range(N):
                entries.append(dict(start=start,
                                    target=raw[starts[m] + off, n].astype(np.float32)))
                meta.append((m, n))
        fc = list(predictor.predict(ListDataset(entries, freq="h")))
        Q = np.empty((c1 - c0, N, C.PRE_LEN, len(qs_np)), np.float32)
        for (m, n), f in zip(meta, fc):
            sim = np.asarray(f.samples, dtype=np.float32)     # [S,72]
            Q[m - c0, n] = np.quantile(sim, qs_np, axis=0).T
        met.update_quantiles(c0, Q)
        if (c0 // chunk) % 5 == 0:
            print(f"  {c1}/{M} 窗  {time.time()-t0:.0f}s", flush=True)
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["chronos_bolt", "moirai"])
    ap.add_argument("--city", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0, help="零样本确定性，仅用于文件名")
    ap.add_argument("--n_quant", type=int, default=99)
    ap.add_argument("--batch_series", type=int, default=4096)
    ap.add_argument("--batch_windows", type=int, default=32,
                    help="Moirai 每块窗口数（条目数 = 窗口×站）")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--limit_test", type=int, default=0)
    ap.add_argument("--no_diag", action="store_true")
    args = ap.parse_args()
    dev = C.torch_device(args.device)

    ds, raw, starts, keep, smean = C.load_city(args.city)
    C.check_protocol(args.city, ds, raw, starts)
    qs_np = np.arange(1, args.n_quant + 1) / (args.n_quant + 1)
    te = starts["te"][:args.limit_test] if args.limit_test else starts["te"]
    te_keep = keep["te"][:len(te)]

    t0 = time.time()
    if args.method == "chronos_bolt":
        met = run_chronos(args, raw, te, te_keep, smean, qs_np, dev)
    else:
        met = run_moirai(args, ds, raw, te, te_keep, smean, qs_np, dev)
    res = met.result()
    extra = dict(elapsed_s=round(time.time() - t0, 1), n_test_windows=len(te),
                 zero_shot=True)
    C.save_metrics(args.method, args.city, 0, res, extra)
    if not args.no_diag:
        C.save_diag(args.method, args.city, 0, met)


if __name__ == "__main__":
    main()
