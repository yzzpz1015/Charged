# -*- coding: utf-8 -*-
"""Zero-shot TimesFM 2.0 evaluation on EV charging demand (pure time-series input).

Feeds each site's raw `seq_l`-step history directly to a pretrained TimesFM
foundation model and compares the `pre_len`-step forecast against raw labels,
using the exact same split + windowing + metric as the trained baselines
(R²≈0.79 PatchTST / 0.787 EV-VLM), so results are directly comparable.

No training, no fine-tuning, no satellite/text modalities.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from api.dataset.common import EVDataset, SPLIT_TEST
from api.utils import calculate_regression_metrics, convert_numpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default='MEL')
    parser.add_argument('--seq_l', type=int, default=72)
    parser.add_argument('--pre_len', type=int, default=6)
    parser.add_argument('--fold', type=int, default=6)
    parser.add_argument('--total_fold', type=int, default=6)
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--data_suffix', default='_remove_const',
                        help='城市目录后缀：_remove_const(默认,已剔恒值站) / _remove_zero')
    parser.add_argument('--checkpoint', default='google/timesfm-2.0-500m-pytorch')
    parser.add_argument('--chunk_size', type=int, default=1024)
    # None -> model pads to its full context (2048). Smaller value truncates the
    # decoder context (faster) but may shift positional encoding; verify before use.
    parser.add_argument('--forecast_context_len', type=int, default=None)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--out', default='result/zero_shot/MEL/timesfm-2.0-500m')
    parser.add_argument('--limit_windows', type=int, default=0,
                        help='0 = all test windows; >0 = smoke test on first N windows')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    data_dir = f"{args.data_path}{args.city}{args.data_suffix}/"

    # --- 1. Same dataset + same split as trained baselines ---
    ev_dataset = EVDataset(feature='volume', auxiliary='None', data_path=data_dir)
    ev_dataset.split_cross_validation(
        fold=args.fold,
        total_fold=args.total_fold,
        train_ratio=0.8,
        valid_ratio=0.1,
    )
    raw = ev_dataset.feat  # [T, N] raw volume (NOT standardized)
    N = raw.shape[1]
    # 窗口起点必须来自 window_starts（新划分的唯一取窗入口）。旧写法
    # ``test_end - test_start - seq_l - pre_len`` 假设起点是连续区间，新口径下是错的：
    # 目标只需落在 test 段、历史可伸进上一段，所以合法窗口**比连续枚举多**；AMS 的段内
    # 还夹着停摆行，连续枚举会把跨缺口的窗也当成合法。要么崩、要么静默错位。
    starts = ev_dataset.window_starts(SPLIT_TEST, args.seq_l, args.pre_len)
    if args.limit_windows > 0:
        starts = starts[:args.limit_windows]
    num_windows = len(starts)
    print(f"[data] city={args.city} raw_feat={raw.shape} N_sites={N} "
          f"test={ev_dataset.test_start}~{ev_dataset.test_end} "
          f"num_windows={num_windows} total_series={num_windows * N}")

    # --- 2. Load TimesFM (frozen, zero-shot) ---
    from transformers import TimesFmModelForPrediction
    model = TimesFmModelForPrediction.from_pretrained(args.checkpoint).to(device).eval()
    print(f"[model] loaded {args.checkpoint} on {device}")

    # --- 3. Build raw history series + raw labels ---
    # window i, site n: history = raw[starts[i] : starts[i]+seq_l, n]
    #                  label   = raw[starts[i]+seq_l : ... +pre_len, n]
    series = []
    labels = np.zeros((num_windows, N, args.pre_len), dtype=np.float32)
    for i in range(num_windows):
        base = starts[i]
        hist = raw[base:base + args.seq_l, :].astype(np.float32)        # [seq_l, N]
        lab = raw[base + args.seq_l: base + args.seq_l + args.pre_len, :].astype(np.float32)  # [pre_len, N]
        labels[i] = lab.T                                                # [N, pre_len]
        for n in range(N):
            series.append(torch.from_numpy(hist[:, n]))                  # 1-D tensor

    # --- 4. Zero-shot inference in chunks ---
    preds = []
    total = len(series)
    freq = 0  # hourly -> high-frequency class (0=sub-daily, 1=weekly/monthly, 2=quarterly/yearly)
    with torch.no_grad():
        for s in range(0, total, args.chunk_size):
            chunk = [t.to(device) for t in series[s:s + args.chunk_size]]
            out = model(
                past_values=chunk,
                freq=[freq] * len(chunk),
                forecast_context_len=args.forecast_context_len,
            )
            preds.append(out.mean_predictions[:, :args.pre_len].cpu().numpy())
            if (s // args.chunk_size + 1) % 5 == 0 or s + args.chunk_size >= total:
                print(f"[infer] {min(s + args.chunk_size, total)}/{total} series done")

    pred_array = np.concatenate(preds, axis=0).reshape(num_windows, N, args.pre_len)

    # --- 5. Metrics in raw space (identical to baseline) ---
    pred_flat = pred_array.reshape(-1, args.pre_len)
    label_flat = labels.reshape(-1, args.pre_len)
    metrics = calculate_regression_metrics(y_true=label_flat, y_pred=pred_flat)
    metrics = convert_numpy(metrics)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'predict.npy'), pred_array)
    np.save(os.path.join(args.out, 'label.npy'), labels)
    with open(os.path.join(args.out, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print("[metrics]")
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
