# -*- coding: utf-8 -*-
"""Train Kairos L1 variants (L1-plain / L1-lora / L1b) on EV charging demand.

Frozen Kairos backbone + trainable point head, per-site (univariate). The three
variants share this script; --variant selects them:

  plain : frozen trunk + point head
  lora  : frozen trunk + point head + LoRA on attention q/k/v/o
  l1b   : frozen trunk + point head + frozen-distilgpt2 semantics (gated)

Loss is MSE in Kairos's instance-normalized space (stable across sites of very
different volume scales); metrics (R2 etc.) are computed in raw space, identical
to the zero-shot baseline for direct comparison.
"""
import os
import sys
import json
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from api.dataset.common import EVDataset
from api.utils import calculate_regression_metrics, convert_numpy


def build_samples(raw: np.ndarray, start: int, end: int, seq_l: int, pre_len: int):
    """Sliding windows over [start, end): return X [M*N, seq_l], Y [M*N, pre_len].

    window i, site n: x = raw[start+i : start+i+seq_l, n],
                      y = raw[start+i+seq_l : start+i+seq_l+pre_len, n]
    """
    num_windows = end - start - seq_l - pre_len
    assert num_windows > 0, f"not enough history in [{start},{end}) for seq_l={seq_l} pre_len={pre_len}"
    N = raw.shape[1]
    X = np.zeros((num_windows, N, seq_l), dtype=np.float32)
    Y = np.zeros((num_windows, N, pre_len), dtype=np.float32)
    for i in range(num_windows):
        base = start + i
        X[i] = raw[base:base + seq_l, :].T          # [N, seq_l]
        Y[i] = raw[base + seq_l:base + seq_l + pre_len, :].T  # [N, pre_len]
    return X.reshape(-1, seq_l), Y.reshape(-1, pre_len)


def build_stat_texts(raw: np.ndarray, start: int, end: int, seq_l: int, pre_len: int):
    """Rule-generated statistical description per window+site (NOT LLM-generated).

    Returns list of strings length M*N, matching build_samples ordering.
    """
    num_windows = end - start - seq_l - pre_len
    N = raw.shape[1]
    texts = []
    for i in range(num_windows):
        base = start + i
        hist = raw[base:base + seq_l, :]  # [seq_l, N]
        for n in range(N):
            x = hist[:, n]
            lo, hi = float(x.min()), float(x.max())
            mean, std = float(x.mean()), float(x.std())
            trend = "up" if x[-1] > x[0] else ("down" if x[-1] < x[0] else "flat")
            texts.append(
                f"range {lo:.1f} to {hi:.1f}, mean {mean:.1f}, std {std:.1f}, trend {trend}."
            )
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default='MEL')
    parser.add_argument('--variant', default='plain', choices=['plain', 'lora', 'l1b'])
    parser.add_argument('--seq_l', type=int, default=72)
    parser.add_argument('--pre_len', type=int, default=6)
    parser.add_argument('--fold', type=int, default=6)
    parser.add_argument('--total_fold', type=int, default=6)
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--checkpoint', default='mldi-lab/Kairos_23m')
    parser.add_argument('--kairos_repo', default='/tmp/Kairos_repo')
    parser.add_argument('--text_model_path', default='api/model/weights/distilgpt2_model')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--out', default='result/kairos_l1')
    parser.add_argument('--limit_train', type=int, default=0,
                        help='>0 = cap train samples for a smoke run')
    args = parser.parse_args()

    # Register Kairos into transformers Auto* (must precede AutoModel import).
    sys.path.insert(0, args.kairos_repo)
    import tsfm.model.kairos  # noqa: F401
    from transformers import AutoModel

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    data_dir = f"{args.data_path}{args.city}_remove_zero/"

    # --- 1. dataset + split (identical to zero-shot baseline) ---
    ev_dataset = EVDataset(feature='volume', auxiliary='None', data_path=data_dir)
    ev_dataset.split_cross_validation(fold=args.fold, total_fold=args.total_fold,
                                      train_ratio=0.8, valid_ratio=0.1)
    raw = ev_dataset.feat  # [T, N] raw volume
    N = raw.shape[1]
    tr_s, tr_e = ev_dataset.train_start, ev_dataset.train_end
    va_s, va_e = ev_dataset.valid_start, ev_dataset.valid_end
    te_s, te_e = ev_dataset.test_start, ev_dataset.test_end

    X_train, Y_train = build_samples(raw, tr_s, tr_e, args.seq_l, args.pre_len)
    X_valid, Y_valid = build_samples(raw, va_s, va_e, args.seq_l, args.pre_len)
    X_test, Y_test = build_samples(raw, te_s, te_e, args.seq_l, args.pre_len)
    if args.limit_train > 0:
        X_train, Y_train = X_train[:args.limit_train], Y_train[:args.limit_train]
    print(f"[data] train={X_train.shape} valid={X_valid.shape} test={X_test.shape}")

    # --- 2. model: frozen Kairos + L1 head ---
    backbone = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True).to(device).eval()
    from api.model.kairos_l1 import KairosL1
    model = KairosL1(
        backbone,
        pre_len=args.pre_len,
        use_lora=(args.variant == 'lora'),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_llm_semantics=(args.variant == 'l1b'),
        text_model_path=args.text_model_path,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"[model] variant={args.variant} trainable_params={n_params:,}")

    # --- 3. L1b: tokenize statistical texts once per split ---
    text_cache = {}
    if args.variant == 'l1b':
        tok = model.text_tokenizer
        for split, (s, e) in [('train', (tr_s, tr_e)), ('valid', (va_s, va_e)), ('test', (te_s, te_e))]:
            texts = build_stat_texts(raw, s, e, args.seq_l, args.pre_len)
            enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
            ids = enc['input_ids']
            mask = enc['attention_mask']
            if args.limit_train > 0 and split == 'train':
                ids, mask = ids[:args.limit_train], mask[:args.limit_train]
            text_cache[split] = (ids, mask)
        print(f"[text] L1b stat texts tokenized, max_len={text_cache['train'][0].shape[1]}")

    # --- 4. train ---
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    n_train = X_train.shape[0]
    n_batches = (n_train + args.batch_size - 1) // args.batch_size

    best_val, best_state, bad = float('inf'), None, 0

    def forward_batch(X, Y, split):
        """Batched eval forward; returns (mean normalized-space loss, raw-space preds)."""
        model.eval()
        n = X.shape[0]
        preds, total = [], 0.0
        with torch.no_grad():
            for b in range(0, n, args.batch_size):
                sl = slice(b, b + args.batch_size)
                xb = torch.from_numpy(X[sl]).to(device)
                yb = torch.from_numpy(Y[sl]).to(device)
                kwargs = {}
                if args.variant == 'l1b':
                    ids, mask = text_cache[split]
                    kwargs['text_ids'] = ids[sl].to(device)
                    kwargs['text_mask'] = mask[sl].to(device)
                y_norm, (loc, scale) = model(xb, **kwargs)
                scale_robust = scale.clamp(min=1.0)
                y_norm_label = (yb - loc) / scale_robust
                total += F.mse_loss(y_norm * (scale / scale_robust), y_norm_label).item() * xb.shape[0]
                preds.append((y_norm * scale + loc).cpu().numpy())
        return total / n, np.concatenate(preds, axis=0)

    for epoch in range(args.epochs):
        # shuffle
        perm = np.random.permutation(n_train)
        X_train, Y_train = X_train[perm], Y_train[perm]
        if args.variant == 'l1b':
            ids, mask = text_cache['train']
            text_cache['train'] = (ids[perm], mask[perm])

        model.train()
        total_loss = 0.0
        for b in range(n_batches):
            sl = slice(b * args.batch_size, (b + 1) * args.batch_size)
            Xb, Yb = X_train[sl], Y_train[sl]
            kwargs = {}
            if args.variant == 'l1b':
                ids, mask = text_cache['train']
                kwargs['text_ids'] = ids[sl].to(device)
                kwargs['text_mask'] = mask[sl].to(device)
            xb = torch.from_numpy(Xb).to(device)
            yb = torch.from_numpy(Yb).to(device)
            y_norm, (loc, scale) = model(xb, **kwargs)
            scale_robust = scale.clamp(min=1.0)
            y_norm_label = (yb - loc) / scale_robust
            loss = F.mse_loss(y_norm * (scale / scale_robust), y_norm_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.shape[0]
        train_loss = total_loss / n_train

        # valid (normalized-space loss, same scale as train)
        val_loss, _ = forward_batch(X_valid, Y_valid, 'valid')
        print(f"[epoch {epoch+1}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val, best_state, bad = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early stop] no improvement for {args.patience} epochs")
                break

    # --- 5. eval on test (raw space, identical metric to zero-shot baseline) ---
    model.load_state_dict(best_state)
    with torch.no_grad():
        _, test_pred = forward_batch(X_test, Y_test, 'test')
    test_pred = test_pred.reshape(-1, args.pre_len)
    test_label = Y_test.reshape(-1, args.pre_len)
    metrics = convert_numpy(calculate_regression_metrics(y_true=test_label, y_pred=test_pred))

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.variant}_pre{args.pre_len}_fold{args.fold}"
    np.save(os.path.join(args.out, f'predict_{tag}.npy'), test_pred)
    np.save(os.path.join(args.out, f'label_{tag}.npy'), test_label)
    torch.save(best_state, os.path.join(args.out, f'{tag}.pth'))
    with open(os.path.join(args.out, f'metrics_{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"[metrics] {tag}")
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
