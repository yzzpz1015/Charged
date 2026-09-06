# -*- coding: utf-8 -*-
"""Train Kairos L2 v3: three EXTERNAL streams (POI / satellite image / satellite
description) + collaboration (--fusion: gate | urban | align).

Corrected path composition (per user): no pure-time-series stream. Frozen Kairos
trunk keeps native TS cross-attn; the three external modalities are injected
layer-by-layer, each its own stream/head. Per-modal readings are same-model clean;
the deployed forecast is the collaboration output (headline vs separately-trained L1).
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

from api.dataset.common import EVDataset
from api.utils import calculate_regression_metrics, convert_numpy


def build_samples(raw, start, end, seq_l, pre_len):
    num_windows = end - start - seq_l - pre_len
    assert num_windows > 0
    N = raw.shape[1]
    X = np.zeros((num_windows, N, seq_l), dtype=np.float32)
    Y = np.zeros((num_windows, N, pre_len), dtype=np.float32)
    for i in range(num_windows):
        base = start + i
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
    parser.add_argument('--checkpoint', default='mldi-lab/Kairos_23m')
    parser.add_argument('--kairos_repo', default='/tmp/Kairos_repo')
    parser.add_argument('--gpt2_path', default='gpt2')
    parser.add_argument('--fusion', default='urban', choices=['gate', 'urban', 'align'])
    parser.add_argument('--gate_init', type=float, default=0.0,
                        help='injection gate init; 1.0 + O zero-init (LoRA-style) lets injection learn')
    parser.add_argument('--align_lambda', type=float, default=1.0,
                        help='weight of img<->desc alignment loss (fusion=align only)')
    parser.add_argument('--align_loss', default='cos', choices=['cos', 'mse'],
                        help='alignment objective on stream hiddens')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--out', default='result/kairos_l2v3')
    parser.add_argument('--limit_train', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--randomize', default='none', choices=['none', 'desc', 'all'],
                        help='random-signal injection control: desc=random caption tokens, '
                             'all=random POI/img/desc. Tests whether the external-path gain is '
                             'regularization (structure) vs real signal (content).')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sys.path.insert(0, args.kairos_repo)
    import tsfm.model.kairos  # noqa: F401
    from transformers import AutoModel, GPT2Tokenizer

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    data_dir = f"{args.data_path}{args.city}_remove_zero/"

    # --- 1. dataset + split ---
    ev_dataset = EVDataset(feature='volume', auxiliary='None', data_path=data_dir)
    ev_dataset.split_cross_validation(fold=args.fold, total_fold=args.total_fold,
                                      train_ratio=0.8, valid_ratio=0.1)
    raw = ev_dataset.feat  # [T, N]
    N = raw.shape[1]
    feat_cols = ev_dataset.feat_columns  # string-ordered site ids
    tr_s, tr_e = ev_dataset.train_start, ev_dataset.train_end
    va_s, va_e = ev_dataset.valid_start, ev_dataset.valid_end
    te_s, te_e = ev_dataset.test_start, ev_dataset.test_end

    X_train, Y_train = build_samples(raw, tr_s, tr_e, args.seq_l, args.pre_len)
    X_valid, Y_valid = build_samples(raw, va_s, va_e, args.seq_l, args.pre_len)
    X_test, Y_test = build_samples(raw, te_s, te_e, args.seq_l, args.pre_len)

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

    # --- 2. external features (per site, aligned to feat_cols order) ---
    # POI density
    sites_info = pd.read_csv(f'{data_dir}sites.csv', header=0)
    site_col = 'site' if 'site' in sites_info.columns else 'site_id'
    sites_info = sites_info.set_index(site_col)
    sites_info.index = sites_info.index.astype(str)
    poi_extra = ev_dataset._load_poi_extra(
        site_ids=[str(c) for c in feat_cols], sites_info=sites_info)
    poi_feat = poi_extra[0].astype(np.float32)  # [N, C_poi]
    print(f"[feat] POI shape={poi_feat.shape}")

    # satellite image embedding (DINOv2 CLS)
    sat_path = f'tiles_dataset/{args.city}/embedding/z16/sat_embed_dinov2_768.npy'
    sat_dinov2 = np.load(sat_path)  # [N, 768], numeric-ordered
    digital_order = sorted(feat_cols, key=int)
    idx = [digital_order.index(sid) for sid in feat_cols]
    img_feat = sat_dinov2[idx].astype(np.float32)  # remap to feat_cols order
    img_feat = img_feat / (np.linalg.norm(img_feat, axis=1, keepdims=True) + 1e-8)
    print(f"[feat] satellite image (DINOv2 CLS) shape={img_feat.shape} (L2-normalized)")

    # satellite description (VLM captions) -> GPT-2 token ids
    cap_path = f'tiles_dataset/{args.city}/embedding/z16/sat_captions.json'
    with open(cap_path) as f:
        captions = json.load(f)
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_path)
    tokenizer.pad_token = tokenizer.eos_token
    cap_ids = [captions[str(sid)] for sid in feat_cols]
    enc = tokenizer(cap_ids, padding='max_length', truncation=True, max_length=48, return_tensors='np')
    desc_ids = enc['input_ids'].astype(np.int64)  # [N, S_desc]
    print(f"[feat] satellite description shape={desc_ids.shape} "
          f"sample[0]={captions[str(feat_cols[0])][:60]}...")

    # --- 2b. random-signal injection control (regularization probe) ---
    # Randomize the per-SITE static features (consistent across train/valid/test,
    # since the same site keeps the same random feature). Uses a dedicated RNG so
    # the training seed / global np state is untouched.
    if args.randomize in ('desc', 'all'):
        rng = np.random.default_rng(args.seed + 12345)
        desc_ids = rng.integers(0, tokenizer.vocab_size, size=desc_ids.shape).astype(np.int64)
    if args.randomize == 'all':
        rng = np.random.default_rng(args.seed + 54321)
        poi_feat = rng.standard_normal(poi_feat.shape).astype(np.float32)
        img_feat = rng.standard_normal(img_feat.shape).astype(np.float32)
        img_feat = img_feat / (np.linalg.norm(img_feat, axis=1, keepdims=True) + 1e-8)
    print(f"[randomize] mode={args.randomize} (random-signal control)")

    def site_feats(idx, feat):
        return None if feat is None else feat[idx]

    poi_train, poi_valid, poi_test = (site_feats(i, poi_feat) for i in (tr_idx, va_idx, te_idx))
    img_train, img_valid, img_test = (site_feats(i, img_feat) for i in (tr_idx, va_idx, te_idx))
    de_train, de_valid, de_test = (site_feats(i, desc_ids) for i in (tr_idx, va_idx, te_idx))

    # --- 3. model ---
    backbone = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True).to(device).eval()
    gpt2 = AutoModel.from_pretrained(args.gpt2_path).to(device).eval()
    gpt2_wte = gpt2.get_input_embeddings()
    from api.model.kairos_l2v3 import KairosL2V3
    model = KairosL2V3(backbone, gpt2_wte, pre_len=args.pre_len, poi_dim=poi_feat.shape[1],
                       img_dim=img_feat.shape[1], fusion=args.fusion,
                       gate_init=args.gate_init).to(device)
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

    def forward_eval(X, Y, poi, img, de):
        model.eval()
        n = X.shape[0]
        out_keys = list(model.streams) + ['collab']
        preds = {k: [] for k in out_keys}
        total = 0.0
        with torch.no_grad():
            for b in range(0, n, args.batch_size):
                sl = slice(b, b + args.batch_size)
                xb = to_t(X[sl]); yb = to_t(Y[sl])
                pb = to_t(poi[sl]); ib = to_t(img[sl]); db = to_t(de[sl])
                outs, (loc, scale), _ = model(xb, poi_feat=pb, img_feat=ib, desc_ids=db)
                scale_robust = scale.clamp(min=1.0)
                y_norm_label = (yb - loc) / scale_robust
                for k, y_norm in outs.items():
                    total += F.mse_loss(y_norm * (scale / scale_robust), y_norm_label).item() * xb.shape[0]
                    preds[k].append((y_norm * scale + loc).cpu().numpy())
        return total / n, {k: np.concatenate(v, axis=0) for k, v in preds.items() if v}

    for epoch in range(args.epochs):
        perm = np.random.permutation(n_train)
        X_train, Y_train = X_train[perm], Y_train[perm]
        poi_train, img_train, de_train = poi_train[perm], img_train[perm], de_train[perm]

        model.train()
        total_loss = 0.0
        for b in range(n_batches):
            sl = slice(b * args.batch_size, (b + 1) * args.batch_size)
            xb = to_t(X_train[sl]); yb = to_t(Y_train[sl])
            pb = to_t(poi_train[sl]); ib = to_t(img_train[sl]); db = to_t(de_train[sl])
            outs, (loc, scale), hiddens = model(xb, poi_feat=pb, img_feat=ib, desc_ids=db)
            scale_robust = scale.clamp(min=1.0)
            y_norm_label = (yb - loc) / scale_robust
            loss = sum(
                F.mse_loss(y_norm * (scale / scale_robust), y_norm_label)
                for y_norm in outs.values()
            )
            # alignment objective: pull img<->desc stream hiddens together;
            # gradient flows into inj_ca['img'] / inj_ca['desc'] injection params
            if args.fusion == 'align' and args.align_lambda > 0:
                h_img, h_desc = hiddens['img'], hiddens['desc']
                if args.align_loss == 'cos':
                    loss = loss + args.align_lambda * (
                        1 - F.cosine_similarity(h_img, h_desc, dim=-1)).mean()
                else:
                    loss = loss + args.align_lambda * F.mse_loss(h_img, h_desc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_train[sl].shape[0]
        train_loss = total_loss / n_train

        val_loss, _ = forward_eval(X_valid, Y_valid, poi_valid, img_valid, de_valid)
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
    _, test_preds = forward_eval(X_test, Y_test, poi_test, img_test, de_test)
    test_label = Y_test.reshape(-1, args.pre_len)

    os.makedirs(args.out, exist_ok=True)
    tag = f"pre{args.pre_len}_fold{args.fold}"
    metrics_all = {}
    for name, pred in test_preds.items():
        m = convert_numpy(calculate_regression_metrics(
            y_true=test_label, y_pred=pred.reshape(-1, args.pre_len)))
        metrics_all[name] = m
        np.save(os.path.join(args.out, f'predict_{name}_{tag}.npy'), pred)

    gates = {}
    for m in ('poi', 'img', 'desc'):
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
