# -*- coding: utf-8 -*-
"""Stage 1 decisive gate: does retrieval (kNN on a station's own history) beat
the fine-tuned TSFM head (L1)?

TS-Memory's core claim is that retrieval-augmented targets beat a plain
fine-tuned head.  The retrieval teacher is PlugMem's ceiling, so if the teacher
does not beat L1 (R2=0.7957), retrieval loses and the full PlugMem machinery is
unnecessary.  This script builds that teacher with NO training:

  * per-site retrieval DB = Kairos encoder embeddings of every 72-step train
    window (instance-norm'd inside encode(), so embeddings match SHAPE not level)
  * for each test window, retrieve top-K similar train windows of the SAME site
  * teacher = softmax(-dist/tau)-weighted mean of the retrieved futures, each
    future level-aligned via instance-norm transfer (neighbor context loc/scale
    -> query loc/scale)
  * compare teacher R2 vs L1 R2 on the identical fold-6 test windows.
"""
import os
import sys
import argparse
import numpy as np
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from api.dataset.common import EVDataset


def build_samples(raw, start, end, seq_l, pre_len):
    """Identical to train_kairos_l1.build_samples.  Row r = window (r//N), site (r%N)."""
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


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    ss = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - float(((y_true - y_pred) ** 2).sum()) / ss if ss > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--city', default='MEL')
    ap.add_argument('--seq_l', type=int, default=72)
    ap.add_argument('--pre_len', type=int, default=6)
    ap.add_argument('--fold', type=int, default=6)
    ap.add_argument('--total_fold', type=int, default=6)
    ap.add_argument('--data_path', default='data/')
    ap.add_argument('--checkpoint', default='mldi-lab/Kairos_23m')
    ap.add_argument('--kairos_repo', default='/tmp/Kairos_repo')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_scale', type=float, default=1.0,
                    help='softmax temperature = tau_scale * mean(top-k dist)')
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--pool', default='mean', choices=['mean', 'last'],
                    help='embedding pooling over encoder sequence')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--l1_dir', default='result/kairos_l1')
    args = ap.parse_args()

    sys.path.insert(0, args.kairos_repo)
    import tsfm.model.kairos  # noqa: F401
    from transformers import AutoModel

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    data_dir = f"{args.data_path}{args.city}_remove_zero/"

    # --- data + split (identical to L1) ---
    ev = EVDataset(feature='volume', auxiliary='None', data_path=data_dir)
    ev.split_cross_validation(fold=args.fold, total_fold=args.total_fold,
                              train_ratio=0.8, valid_ratio=0.1)
    raw = ev.feat  # [T, N]
    N = raw.shape[1]
    tr_s, tr_e = ev.train_start, ev.train_end
    te_s, te_e = ev.test_start, ev.test_end

    X_train, Y_train = build_samples(raw, tr_s, tr_e, args.seq_l, args.pre_len)
    X_test, Y_test = build_samples(raw, te_s, te_e, args.seq_l, args.pre_len)
    Mt = len(X_train) // N
    Me = len(X_test) // N
    print(f"[data] N={N} sites, train={len(X_train)} ({Mt} win/site), test={len(X_test)} ({Me} win/site)")

    # --- verify test labels match L1's saved labels (alignment check) ---
    lab_path = os.path.join(args.l1_dir, f"label_plain_pre{args.pre_len}_fold{args.fold}.npy")
    pred_path = os.path.join(args.l1_dir, f"predict_plain_pre{args.pre_len}_fold{args.fold}.npy")
    if os.path.exists(lab_path) and os.path.exists(pred_path):
        lab = np.load(lab_path)
        pred = np.load(pred_path)
        if lab.shape == Y_test.shape and np.allclose(lab, Y_test, atol=1e-4):
            print(f"[align] Y_test == label_plain, L1 R2 (recomputed) = {r2(lab, pred):+.4f}")
        else:
            print(f"[warn] Y_test {Y_test.shape} vs label_plain {lab.shape} mismatch; will recompute L1 on own windows")
            pred = None
    else:
        print("[warn] L1 npy not found; skipping L1 comparison")
        pred = None

    # --- frozen Kairos backbone ---
    backbone = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def encode(X):
        E, L, S = [], [], []
        for b in range(0, len(X), args.batch_size):
            xb = torch.from_numpy(X[b:b + args.batch_size]).to(device)
            hidden, (loc, scale), *_ = backbone.encode(xb)
            if args.pool == 'mean':
                emb = hidden.mean(dim=1)
            else:
                emb = hidden[:, -1, :]
            E.append(emb.cpu().numpy())
            L.append(loc.cpu().numpy())
            S.append(scale.cpu().numpy())
        return (np.concatenate(E), np.concatenate(L).reshape(-1, 1),
                np.concatenate(S).reshape(-1, 1))

    print("[encode] train windows ...")
    E_train, loc_tr, scale_tr = encode(X_train)
    print("[encode] test windows ...")
    E_test, loc_te, scale_te = encode(X_test)
    d = E_train.shape[1]
    print(f"[encode] done, embedding dim={d}")

    # --- retrieval + teacher, per site ---
    tau = args.tau_scale
    teacher = np.zeros_like(Y_test)

    for n in range(N):
        tr_idx = np.arange(Mt) * N + n
        te_idx = np.arange(Me) * N + n
        Etr, Ete = E_train[tr_idx], E_test[te_idx]
        # squared L2 via broadcast (avoid scipy dep)
        d2 = (Ete ** 2).sum(1, keepdims=True) + (Etr ** 2).sum(1)[None, :] - 2 * Ete @ Etr.T
        dist = np.sqrt(np.maximum(d2, 0.0))  # [Me, Mt]
        K = min(args.top_k, Mt)
        order = np.argpartition(dist, K - 1, axis=1)[:, :K]
        dist_k = np.take_along_axis(dist, order, axis=1)
        w = np.exp(-dist_k / (tau * dist_k.mean(axis=1, keepdims=True) + 1e-8))
        w = w / w.sum(axis=1, keepdims=True)

        f_k = Y_train[tr_idx][order]                       # [Me, K, pre]
        l_k = loc_tr[tr_idx][order]                        # [Me, K, 1]
        s_k = np.maximum(scale_tr[tr_idx][order], 1e-3)    # [Me, K, 1]
        l_q = loc_te[te_idx][:, None, :]                   # [Me, 1, 1]
        s_q = scale_te[te_idx][:, None, :]                 # [Me, 1, 1]
        f_mapped = (f_k - l_k) / s_k * s_q + l_q           # instance-norm transfer
        teacher[te_idx] = (w[:, :, None] * f_mapped).sum(1)

    # --- compare ---
    r2_teacher = r2(Y_test, teacher)
    print(f"\n{'method':<28}{'R2':>9}")
    if pred is not None:
        print(f"{'L1 (fine-tuned head)':<28}{r2(Y_test, pred):>+9.4f}")
    print(f"{'retrieval teacher (kNN)':<28}{r2_teacher:>+9.4f}")

    out_dir = 'result/kairos_retrieval'
    os.makedirs(out_dir, exist_ok=True)
    tag = f"teacher_k{args.top_k}_pre{args.pre_len}_fold{args.fold}"
    np.save(os.path.join(out_dir, f'predict_{tag}.npy'), teacher)
    np.save(os.path.join(out_dir, f'label_{tag}.npy'), Y_test)
    print(f"[saved] {out_dir}/predict_{tag}.npy")


if __name__ == '__main__':
    main()
