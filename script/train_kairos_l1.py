# -*- coding: utf-8 -*-
"""Train Kairos L1 variants (L1-plain / L1-lora / L1b) on EV charging demand.

Frozen Kairos backbone + trainable point head, per-site (univariate). The three
variants share this script; --variant selects them:

  plain : frozen trunk + point head
  lora  : frozen trunk + point head + LoRA on attention q/k/v/o
  l1b   : frozen trunk + point head + frozen-distilgpt2 semantics (gated)

Loss is MSE in **raw space** (see batch_loss for why); metrics (R2 etc.) are also
raw space, identical to the zero-shot baseline for direct comparison.
"""
import os
import sys
import json
import argparse
import importlib.util
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from api.dataset.common import EVDataset, SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST
from api.utils import (calculate_regression_metrics, convert_numpy,
                       straight_window_keep)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 基线的划分/站集权威表（8 元组：tr起/tr止/va起/va止/te起/te止/停摆行数/站数）。
# 评测口径只允许有一份定义，这里只读不写。
pa = _load("prob_anchor_swap",
           os.path.join(parent_dir, "script", "prob_anchor_swap.py"))


def build_windows(raw: np.ndarray, starts: np.ndarray, seq_l: int, pre_len: int):
    """按**全局窗口起点**取窗：返回 X [M*N, seq_l], Y [M*N, pre_len]。

    ``starts`` 必须来自 ``EVDataset.window_starts()``（即
    :func:`api.utils.split_window_starts`）——**取窗语义的唯一来源**。旧实现按
    ``[start, end)`` 连续区间数窗口，在「停摆丢弃 + 可用行计数 2/1/3」的新划分下
    会错：段内仍夹着停摆行（AMS 尤甚），且段首尾不再是窗口的自然边界。

    排布：window-major / site-minor（``index = i * N + n``），
    :func:`build_stat_texts` 必须用同一排布。
    """
    M = len(starts)
    N = raw.shape[1]
    if M == 0:
        return (np.zeros((0, seq_l), np.float32), np.zeros((0, pre_len), np.float32))
    off_in = np.arange(seq_l, dtype=np.int64)[None, :]
    off_out = np.arange(pre_len, dtype=np.int64)[None, :]
    X = raw[starts[:, None] + off_in]                  # [M, seq_l, N]
    Y = raw[starts[:, None] + seq_l + off_out]         # [M, pre_len, N]
    X = np.transpose(X, (0, 2, 1)).reshape(M * N, seq_l)
    Y = np.transpose(Y, (0, 2, 1)).reshape(M * N, pre_len)
    return X.astype(np.float32), Y.astype(np.float32)


def build_stat_texts(raw: np.ndarray, starts: np.ndarray, seq_l: int, pre_len: int):
    """Rule-generated statistical description per window+site (NOT LLM-generated).

    Returns list of strings length M*N, matching build_windows ordering.
    """
    hist = raw[starts[:, None] + np.arange(seq_l, dtype=np.int64)[None, :]]  # [M, seq_l, N]
    # 压成 [M*N]，与 build_windows 的 window-major / site-minor 排布一致
    lo, hi = hist.min(1).reshape(-1), hist.max(1).reshape(-1)
    mean, std = hist.mean(1).reshape(-1), hist.std(1).reshape(-1)
    up, dn = (hist[:, -1, :] > hist[:, 0, :]).reshape(-1), (hist[:, -1, :] < hist[:, 0, :]).reshape(-1)
    texts = []
    for i in range(lo.shape[0]):
        trend = "up" if up[i] else ("down" if dn[i] else "flat")
        texts.append(f"range {lo[i]:.1f} to {hi[i]:.1f}, mean {mean[i]:.1f}, "
                     f"std {std[i]:.1f}, trend {trend}.")
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default='MEL')
    parser.add_argument('--variant', default='plain', choices=['plain', 'lora', 'l1b', 'quant'],
                        help='quant = 分位数头 + pinball 损失（做法③），直接最小化 CRPS')
    parser.add_argument('--backbone', default='kairos', choices=['kairos', 'patchtst'],
                        help='patchtst 只支持 --variant quant（分位头消融）')
    parser.add_argument('--tag_prefix', default='',
                        help='输出文件名前缀；空则用 variant。PatchTST 消融用 patch_quant 避免覆盖 Kairos 产物')
    parser.add_argument('--n_quant', type=int, default=99,
                        help='quant 变体的分位点数，均匀铺在 (0,1)：i/(n+1), i=1..n')
    parser.add_argument('--seq_l', type=int, default=72)
    parser.add_argument('--pre_len', type=int, default=6)
    parser.add_argument('--fold', type=int, default=6)
    parser.add_argument('--total_fold', type=int, default=6)
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--data_suffix', default='_remove_const',
                        help='城市目录后缀：_remove_const(默认,已剔恒值站) / _remove_zero')
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
    parser.add_argument('--out', default='result/kairos_l1',
                        help='输出根目录，实际写入 <out>/<city>/（多城并行不会互相覆盖）')
    parser.add_argument('--limit_train', type=int, default=0,
                        help='>0 = cap train samples for a smoke run')
    args = parser.parse_args()

    # Register Kairos into transformers Auto* (must precede AutoModel import).
    sys.path.insert(0, args.kairos_repo)
    import tsfm.model.kairos  # noqa: F401
    from transformers import AutoModel

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    data_dir = f"{args.data_path}{args.city}{args.data_suffix}/"
    # 每城一个子目录。文件名（tag）里不含城市，多城并行跑同一个 --out 会**互相覆盖**
    # （2026-10-06 踩过：6 城写同一批 .npy/.pth，最后只剩最后完成那城的）。
    out_dir = os.path.join(args.out, args.city)

    # --- 1. dataset + split (identical to zero-shot baseline) ---
    ev_dataset = EVDataset(feature='volume', auxiliary='None', data_path=data_dir)
    ev_dataset.split_cross_validation(fold=args.fold, total_fold=args.total_fold,
                                      train_ratio=0.8, valid_ratio=0.1)
    raw = ev_dataset.feat  # [T, N] raw volume（剔站已在 split_cross_validation 里做完）
    N = raw.shape[1]

    # 取窗走 window_starts —— 与模型侧（CreateDataset）**同一实现**，且与基线
    # script/prob_baseline.py 的划分同口径。旧脚本手算 end-start-seq_l-pre_len，
    # 在有停摆缺口的 AMS 上会错。
    starts = {k: ev_dataset.window_starts(sid, args.seq_l, args.pre_len)
              for k, sid in (('train', SPLIT_TRAIN), ('valid', SPLIT_VALID), ('test', SPLIT_TEST))}
    for k, v in starts.items():
        assert len(v) > 0, f"{args.city} 的 {k} 段窗口数为 0，划分/剔站有问题"

    # 口径自检：站数与划分区间必须与基线逐项一致（check_split_alignment 的轻量版）
    _exp = pa.EXPECT.get(args.city)
    if _exp is None:
        print(f"[warn] {args.city} 不在基线 EXPECT 表里，跳过口径自检")
    else:
        _T = ev_dataset.time
        _got_tr = _T[ev_dataset.train_start].strftime('%m-%d')
        _got_te = _T[ev_dataset.test_start].strftime('%m-%d')
        assert N == _exp[7], f"{args.city} 剔站后站数 {N} != 基线 {_exp[7]}"
        assert (_got_tr, _got_te) == (_exp[0], _exp[4]), \
            f"{args.city} 划分与基线不一致：train起 {_got_tr}/test起 {_got_te} " \
            f"vs 基线 {_exp[0]}/{_exp[4]}"
    print(f"[data] 站 {N}  窗口数 train={len(starts['train'])} "
          f"valid={len(starts['valid'])} test={len(starts['test'])}", flush=True)

    X_train, Y_train = build_windows(raw, starts['train'], args.seq_l, args.pre_len)
    X_valid, Y_valid = build_windows(raw, starts['valid'], args.seq_l, args.pre_len)
    X_test, Y_test = build_windows(raw, starts['test'], args.seq_l, args.pre_len)

    # --- 1b. 逐站剔除合成直线段（模型是单站预测，所以按 站×窗 剔除，不删整行）---
    # 样本排布是 window-major / site-minor（build_windows 的 index = i*N + n），
    # keep.reshape(-1) 与之一一对应。test 也留着全量 —— 落盘的 predict_*.npy 要与
    # kairos_prob_eval 的独立推理路径对账；评估指标按同一个掩码算。
    keep = {k: straight_window_keep(raw, starts[k], args.seq_l, args.pre_len)
            for k in ('train', 'valid', 'test')}
    flat = {k: keep[k].reshape(-1) for k in keep}
    for k in ('train', 'valid', 'test'):
        pct = 100 * (1 - keep[k].mean()) if keep[k].size else 0.0
        print(f"[clean] {k}: 剔除 {pct:.1f}% 的站×窗样本"
              f"（整段 {args.seq_l + args.pre_len}h 无合成直线）", flush=True)
    X_train, Y_train = X_train[flat['train']], Y_train[flat['train']]
    X_valid, Y_valid = X_valid[flat['valid']], Y_valid[flat['valid']]
    if args.limit_train > 0:
        X_train, Y_train = X_train[:args.limit_train], Y_train[:args.limit_train]
    print(f"[data] train={X_train.shape} valid={X_valid.shape} "
          f"test={X_test.shape}（test 全量推理，指标按 clean 掩码）", flush=True)

    # --- 2. model: frozen Kairos + L1 head, or PatchTST + quantile head (ablation) ---
    qs = np.arange(1, args.n_quant + 1) / (args.n_quant + 1)
    if args.backbone == 'patchtst':
        if args.variant != 'quant':
            raise ValueError('--backbone patchtst 只支持 --variant quant')
        from api.model.patchtst_quant import PatchTSTQuant
        model = PatchTSTQuant(
            seq_len=args.seq_l,
            pre_len=args.pre_len,
            quantiles=qs,
        ).to(device)
    else:
        backbone = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True).to(device).eval()
        from api.model.kairos_l1 import KairosL1
        model = KairosL1(
            backbone,
            pre_len=args.pre_len,
            quantiles=(qs if args.variant == 'quant' else None),
            use_lora=(args.variant == 'lora'),
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            use_llm_semantics=(args.variant == 'l1b'),
            text_model_path=args.text_model_path,
        ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"[model] variant={args.variant} trainable_params={n_params:,}", flush=True)

    # --- 3. L1b: tokenize statistical texts once per split ---
    text_cache = {}
    if args.variant == 'l1b':
        tok = model.text_tokenizer
        for split in ('train', 'valid', 'test'):
            texts = build_stat_texts(raw, starts[split], args.seq_l, args.pre_len)
            if split in ('train', 'valid'):          # 与过滤后的 X/Y 同序
                texts = [t for t, m in zip(texts, flat[split]) if m]
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

    def batch_loss(y_norm, yb, loc, scale):
        """按变体选损失。

        quant : pinball，**回报 2×pinball 使其就是 CRPS 本身**，这样 train/val 曲线
                与评测指标同量纲（CRPS(F,y)=2∫ρ_τ(F⁻¹(τ),y)dτ）。
        点预测头：**原始空间 MSE**。

        为什么不在 Kairos 的归一化空间算：归一化空间里每个 (窗,站) 等权，而站点量级
        跨站差几个数量级（JHB 活跃站 smean 低到 0.38，AMS 是另一个量级）。评测口径
        R² 与 crpsΣ（Σ_j CRPS_j/Σ_j smean_j）都是**按站点量级加权**的，在归一化空间
        训练等于换了一个与指标不符的目标函数。

        实测（同划分同超参，只换损失口径）：
          JHB 归一化 −1.0052 / 原始 0.4783   —— 归一化直接崩到 R²<0
          MEL 原始 0.7000（PatchTST 0.7135）
          SPO 原始 0.5147（PatchTST 0.4997）
        旧实现（"scale>=1 用归一化、scale<1 换回原始单位"）是第三种、量纲不连续的口径，
        已删（git 历史里有）：它使小 scale 样本与其余样本处在不同量纲下相加。
        """
        if args.variant == 'quant':
            return 2.0 * model.quantile_loss(y_norm, yb, loc, scale)
        return F.mse_loss(model.denorm(y_norm, loc, scale), yb)

    def forward_batch(X, Y, split):
        """Batched eval forward; returns (mean loss, raw-space preds).

        preds 形状：[n, pre_len]（点）或 [n, pre_len, Q]（分位数）。
        """
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
                total += batch_loss(y_norm, yb, loc, scale).item() * xb.shape[0]
                preds.append(model.denorm(y_norm, loc, scale).cpu().numpy())
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
            loss = batch_loss(y_norm, yb, loc, scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.shape[0]
        train_loss = total_loss / n_train

        # valid (normalized-space loss, same scale as train)
        val_loss, _ = forward_batch(X_valid, Y_valid, 'valid')
        print(f"[epoch {epoch+1}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
              flush=True)

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
    test_label = Y_test.reshape(-1, args.pre_len)
    # 指标只算 clean 的站×窗；落盘的 predict 仍是全量（与评测脚本对账用）。
    tm = flat['test']

    # --- 6. 落盘 ---
    # **只存 predict / starts / pth**。不存 label：它等于 raw[starts + seq_l + j, n]，
    # 读一次数据集就能原样重建（见 kairos_prob_eval.load_city），存一份是白占
    # 1.6 GB —— /home/data 是共享盘且长期接近满（2026-10-06 撞过 ENOSPC，
    # 多城并行写盘把 JHB/SPO 的 torch.save 直接写崩）。
    # 也**不存分位数张量**：AMS 上 [M*N, 72, 99] 是 66 TB 量级；分位数一律在
    # kairos_prob_eval.py 里边算边评，不落地。
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.tag_prefix or args.variant}_pre{args.pre_len}_fold{args.fold}"
    if args.variant == 'quant':
        med = int(np.argmin(np.abs(qs - 0.5)))
        test_med = test_pred[:, :, med]
        np.save(os.path.join(out_dir, f'qlevels_{tag}.npy'), qs)
        metrics = convert_numpy(calculate_regression_metrics(
            y_true=test_label[tm], y_pred=test_med[tm]))
        metrics['quantile_levels'] = args.n_quant
        metrics['point_src'] = f'median quantile (q={qs[med]:.2f})'
        metrics['test_clean_frac'] = float(tm.mean())
    else:
        test_pred = test_pred.reshape(-1, args.pre_len)
        np.save(os.path.join(out_dir, f'predict_{tag}.npy'), test_pred.astype(np.float32))
        metrics = convert_numpy(calculate_regression_metrics(
            y_true=test_label[tm], y_pred=test_pred[tm]))
        metrics['test_clean_frac'] = float(tm.mean())
    np.save(os.path.join(out_dir, f'starts_{tag}.npy'), starts['test'])
    np.save(os.path.join(out_dir, f'keep_test_{tag}.npy'), keep['test'])
    torch.save(best_state, os.path.join(out_dir, f'{tag}.pth'))
    with open(os.path.join(out_dir, f'metrics_{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"[metrics] {tag} -> {out_dir}")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
