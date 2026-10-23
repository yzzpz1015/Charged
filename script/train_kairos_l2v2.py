# -*- coding: utf-8 -*-
"""Train Kairos L2 v2: three-stream layer-by-layer injection + pluggable fusion.

Corrected architecture (api/model/kairos_l2v2.py): three decoder streams through
the frozen Kairos trunk — ts (native only), poi (ts + POI injected every layer),
sat (ts + satellite injected every layer). Per-stream heads give a clean
within-model net-contribution reading (y_poi - y_ts, y_sat - y_ts); a fusion
module (--fusion: static | gate | mlp) produces y_collab, the deployed forecast.

All outputs are trained jointly (sum of normalised-space MSE, robust scale).
All-zero-history windows are dropped in train AND eval (see data-zero analysis).
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from api.dataset.common import (EVDataset, SPLIT_TRAIN,
                                SPLIT_VALID, SPLIT_TEST)
from api.utils import calculate_regression_metrics, convert_numpy


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
    parser.add_argument('--checkpoint', default='mldi-lab/Kairos_23m')
    parser.add_argument('--kairos_repo', default='/tmp/Kairos_repo')
    parser.add_argument('--use_poi', action='store_true', default=True)
    parser.add_argument('--use_sat', action='store_true', default=True)
    parser.add_argument('--fusion', default='gate', choices=['static', 'gate', 'mlp'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--out', default='result/kairos_l2v2')
    parser.add_argument('--limit_train', type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.kairos_repo)
    import tsfm.model.kairos  # noqa: F401
    from transformers import AutoModel

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    data_dir = f"{args.data_path}{args.city}{args.data_suffix}/"

    # --- 1. dataset + split ---
    ev_dataset = EVDataset(feature='volume', auxiliary='None', data_path=data_dir)
    ev_dataset.split_cross_validation(fold=args.fold, total_fold=args.total_fold,
                                      train_ratio=0.8, valid_ratio=0.1)
    raw = ev_dataset.feat  # [T, N]
    N = raw.shape[1]
    feat_cols = ev_dataset.feat_columns  # string-ordered site ids
    st_tr = ev_dataset.window_starts(SPLIT_TRAIN, args.seq_l, args.pre_len)
    st_va = ev_dataset.window_starts(SPLIT_VALID, args.seq_l, args.pre_len)
    st_te = ev_dataset.window_starts(SPLIT_TEST, args.seq_l, args.pre_len)

    X_train, Y_train = build_samples(raw, st_tr, args.seq_l, args.pre_len)
    X_valid, Y_valid = build_samples(raw, st_va, args.seq_l, args.pre_len)
    X_test, Y_test = build_samples(raw, st_te, args.seq_l, args.pre_len)

    # Drop all-zero-history windows (train AND eval). site_idx computed before
    # filtering so per-site external features stay aligned.
    def filter_zero_history(X, Y):
        keep = X.sum(axis=1) > 0
        site_idx = (np.arange(X.shape[0]) % N)[keep]
        return X[keep], Y[keep], site_idx, int((~keep).sum())

    X_train, Y_train, tr_idx, tr_dropped = filter_zero_history(X_train, Y_train)
    X_valid, Y_valid, va_idx, va_dropped = filter_zero_history(X_valid, Y_valid)
    X_test, Y_test, te_idx, te_dropped = filter_zero_history(X_test, Y_test)
    if args.limit_train > 0:
        X_train, Y_train, tr_idx = X_train[:args.limit_train], Y_train[:args.limit_train], tr_idx[:args.limit_train]
    print(f"[data] train={X_train.shape} (dropped {tr_dropped}) "
          f"valid={X_valid.shape} (dropped {va_dropped}) "
          f"test={X_test.shape} (dropped {te_dropped})")

    # --- 2. external static features (per site), aligned to feat_cols order ---
    poi_feat = None
    sat_feat = None

    if args.use_poi:
        sites_info = pd.read_csv(f'{data_dir}sites.csv', header=0)
        site_col = 'site' if 'site' in sites_info.columns else 'site_id'
        sites_info = sites_info.set_index(site_col)
        sites_info.index = sites_info.index.astype(str)
        poi_extra = ev_dataset._load_poi_extra(
            site_ids=[str(c) for c in feat_cols], sites_info=sites_info)
        if poi_extra is not None:
            poi_feat = poi_extra[0].astype(np.float32)  # [N, C_poi]
            print(f"[feat] POI shape={poi_feat.shape}")
        else:
            print("[feat] POI unavailable, disabling poi path")
            args.use_poi = False

    if args.use_sat:
        sat_path = f'tiles_dataset/{args.city}/embedding/z16/sat_embed_dinov2_768.npy'
        if os.path.exists(sat_path):
            sat_dinov2 = np.load(sat_path)  # [N, 768], numeric-ordered
            digital_order = sorted(feat_cols, key=int)
            idx = [digital_order.index(sid) for sid in feat_cols]
            sat_feat = sat_dinov2[idx].astype(np.float32)  # remap to feat_cols order
            sat_feat = sat_feat / (np.linalg.norm(sat_feat, axis=1, keepdims=True) + 1e-8)
            print(f"[feat] sat semantic shape={sat_feat.shape} (L2-normalized)")
        else:
            print(f"[feat] sat embedding missing ({sat_path}), disabling sat path")
            args.use_sat = False

    poi_dim = poi_feat.shape[1] if poi_feat is not None else 1
    sat_dim = sat_feat.shape[1] if sat_feat is not None else 1

    # per-sample site feature (use the pre-filtered site index)
    def site_feats(idx, feat):
        return None if feat is None else feat[idx]

    poi_train = site_feats(tr_idx, poi_feat)
    poi_valid = site_feats(va_idx, poi_feat)
    poi_test = site_feats(te_idx, poi_feat)
    sat_train = site_feats(tr_idx, sat_feat)
    sat_valid = site_feats(va_idx, sat_feat)
    sat_test = site_feats(te_idx, sat_feat)

    # --- 3. model ---
    backbone = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True).to(device).eval()
    from api.model.kairos_l2v2 import KairosL2
    model = KairosL2(backbone, pre_len=args.pre_len, poi_dim=poi_dim, sat_dim=sat_dim,
                     use_poi=args.use_poi, use_sat=args.use_sat, fusion=args.fusion).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] streams={model.streams} fusion={args.fusion} "
          f"trainable_params={sum(p.numel() for p in trainable):,}")

    # --- 4. train ---
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    n_train = X_train.shape[0]
    n_batches = (n_train + args.batch_size - 1) // args.batch_size
    best_val, best_state, bad = float('inf'), None, 0

    def to_t(x):
        return None if x is None else torch.from_numpy(x).to(device)

    def forward_eval(X, Y, poi, sat):
        model.eval()
        n = X.shape[0]
        out_keys = list(model.streams) + ['collab']
        preds = {k: [] for k in out_keys}
        total = 0.0
        with torch.no_grad():
            for b in range(0, n, args.batch_size):
                sl = slice(b, b + args.batch_size)
                xb = to_t(X[sl]); yb = to_t(Y[sl])
                pb = None if poi is None else to_t(poi[sl])
                sb = None if sat is None else to_t(sat[sl])
                outs, (loc, scale) = model(xb, poi_feat=pb, sat_feat=sb)
                scale_robust = scale.clamp(min=1.0)
                y_norm_label = (yb - loc) / scale_robust
                for k, y_norm in outs.items():
                    if y_norm is None:
                        continue
                    total += F.mse_loss(y_norm * (scale / scale_robust), y_norm_label).item() * xb.shape[0]
                    preds[k].append((y_norm * scale + loc).cpu().numpy())
        return total / n, {k: np.concatenate(v, axis=0) for k, v in preds.items() if v}

    for epoch in range(args.epochs):
        perm = np.random.permutation(n_train)
        X_train, Y_train = X_train[perm], Y_train[perm]
        if poi_train is not None:
            poi_train = poi_train[perm]
        if sat_train is not None:
            sat_train = sat_train[perm]

        model.train()
        total_loss = 0.0
        for b in range(n_batches):
            sl = slice(b * args.batch_size, (b + 1) * args.batch_size)
            xb = to_t(X_train[sl]); yb = to_t(Y_train[sl])
            pb = None if poi_train is None else to_t(poi_train[sl])
            sb = None if sat_train is None else to_t(sat_train[sl])
            outs, (loc, scale) = model(xb, poi_feat=pb, sat_feat=sb)
            scale_robust = scale.clamp(min=1.0)
            y_norm_label = (yb - loc) / scale_robust
            loss = sum(
                F.mse_loss(y_norm * (scale / scale_robust), y_norm_label)
                for y_norm in outs.values() if y_norm is not None
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_train[sl].shape[0]
        train_loss = total_loss / n_train

        val_loss, _ = forward_eval(X_valid, Y_valid, poi_valid, sat_valid)
        print(f"[epoch {epoch+1}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val, best_state, bad = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early stop] no improvement for {args.patience} epochs")
                break

    # --- 5. eval per stream + collab + gate diagnostics ---
    model.load_state_dict(best_state)
    _, test_preds = forward_eval(X_test, Y_test, poi_test, sat_test)
    test_label = Y_test.reshape(-1, args.pre_len)

    os.makedirs(args.out, exist_ok=True)
    tag = f"pre{args.pre_len}_fold{args.fold}"
    metrics_all = {}
    for name, pred in test_preds.items():
        m = convert_numpy(calculate_regression_metrics(
            y_true=test_label, y_pred=pred.reshape(-1, args.pre_len)))
        metrics_all[name] = m
        np.save(os.path.join(args.out, f'predict_{name}_{tag}.npy'), pred)

    # gate diagnostics: mean over layers per modality (not a comparison instrument)
    gates = {}
    for m in ('poi', 'sat'):
        if m in model.gates:
            gates[f'gate_{m}_mean'] = float(model.gates[m].detach().mean().cpu().item())
            gates[f'gate_{m}_per_layer'] = model.gates[m].detach().cpu().tolist()
    metrics_all['gates'] = gates

    torch.save(best_state, os.path.join(args.out, f'{tag}.pth'))
    with open(os.path.join(args.out, f'metrics_{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=4)

    print(f"[gates] {gates}")
    print("[metrics] R2 per stream / collab:")
    for name, m in metrics_all.items():
        if name == 'gates':
            continue
        print(f"  {name}: R2={m.get('R²', m.get('R2')):.4f}")


if __name__ == '__main__':
    main()
