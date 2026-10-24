# -*- coding: utf-8 -*-
"""#7–#10：点预测主干 + 共用分位头（99 分位，raw 空间 pinball）。

用法::

    python3 script/baselines/train_quant_backbone.py \
        --method itransformer --city MEL --seed 0 --device 0

方法：``itransformer`` / ``stgcn`` / ``stgcn_nograph`` / ``gwnet`` / ``mqcnn``。

协议（所有方法共用）：
* 窗 ``seq_l=72, pre_len=72``，训练/验证均按 clean mask 加权；
* 每 epoch 固定抽 ``steps_per_epoch`` 个窗口批（各城训练预算相同，避免 SZH 800+
  站把 epoch 拉成几小时、小城欠训练）；
* 早停看验证集 masked pinball；测试集边推理边累计指标，只落 metrics + 诊断子集；
* 邻接：distance.csv → **reindex** → kNN(k=8, exp(-d/median)) 对称化 + 自环；
  ``stgcn_nograph`` 用 A=I，参数量与带图版完全相同。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402
from api.model.quant_backbones import (  # noqa: E402
    GWNetQuant, STGCNQuant, iTransformerQuant, knn_adjacency,
)
from api.model.quant_head import pinball_loss  # noqa: E402
from api.model.mqcnn_quant import MQCNNQuant  # noqa: E402

METHODS = ("itransformer", "stgcn", "stgcn_nograph", "gwnet", "mqcnn")


def build_model(args, N, n_quant, adj):
    if adj is not None:
        adj = torch.from_numpy(adj)
    if args.method == "itransformer":
        return iTransformerQuant(N, C.SEQ_L, C.PRE_LEN, n_quant,
                                 d_model=args.d_model, nhead=args.nhead,
                                 n_layers=args.layers, d_ff=args.d_ff,
                                 dropout=args.dropout)
    if args.method in ("stgcn", "stgcn_nograph"):
        return STGCNQuant(N, C.SEQ_L, C.PRE_LEN, n_quant,
                          d_model=args.d_model, n_layers=args.layers,
                          dropout=args.dropout, adj_matrix=adj,
                          graph=args.method == "stgcn")
    if args.method == "gwnet":
        return GWNetQuant(N, C.SEQ_L, C.PRE_LEN, n_quant,
                          d_model=args.d_model, n_layers=args.layers,
                          dropout=args.dropout, adj_matrix=adj)
    if args.method == "mqcnn":
        return MQCNNQuant(N, C.SEQ_L, C.PRE_LEN, n_quant,
                          d_model=args.d_model, dropout=args.dropout)
    raise ValueError(args.method)


def batch_windows(N, budget):
    """一个 batch 放几个窗口：按站数折算，限制显存。"""
    return int(max(1, min(32, budget // max(N, 1))))


@torch.no_grad()
def eval_loss(model, raw, starts, keep, qs, device, w, bs=1):
    model.eval()
    tot, n = 0.0, 0
    off = np.arange(C.SEQ_L)[None, :]
    for c0 in range(0, len(starts), w):
        st = starts[c0:c0 + w]
        X = raw[st[:, None] + off].astype(np.float32)                  # [w,T,N]
        Y = raw[st[:, None] + C.SEQ_L + np.arange(C.PRE_LEN)[None, :]]
        Y = Y.transpose(0, 2, 1).astype(np.float32)
        xb = torch.from_numpy(X).to(device)
        yb = torch.from_numpy(Y).to(device)
        kb = torch.from_numpy(keep[c0:c0 + w]).to(device)
        yn, (loc, scale) = model(xb)
        pred = model.denorm(yn, loc, scale)
        tot += float(pinball_loss(pred, yb, qs, kb)) * len(st)
        n += len(st)
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def run_test(model, raw, starts, keep, smean, qs, device, w, with_diag=True):
    model.eval()
    met = C.make_metrics(None, raw, starts, keep, smean, qs, with_diag=with_diag)
    off = np.arange(C.SEQ_L)[None, :]
    for c0 in range(0, len(starts), w):
        st = starts[c0:c0 + w]
        X = raw[st[:, None] + off].astype(np.float32)                  # [w,T,N]
        xb = torch.from_numpy(X).to(device)
        yn, (loc, scale) = model(xb)
        pred = model.denorm(yn, loc, scale).float().cpu().numpy()   # [w,N,72,Q]
        met.update_quantiles(c0, pred)
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--city", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps_per_epoch", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_series", type=int, default=4096,
                    help="一个 batch 的站序列数上限（决定窗口数 w=budget/N）")
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--n_quant", type=int, default=99)
    ap.add_argument("--limit_test", type=int, default=0, help=">0 只跑前 N 个测试窗（冒烟）")
    ap.add_argument("--no_diag", action="store_true")
    ap.add_argument("--eval_only", action="store_true",
                    help="跳过训练，直接载入 <ckpt>/seed{s}.pth 重评")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dev = C.torch_device(args.device)

    ds, raw, starts, keep, smean = C.load_city(args.city)
    N, M = C.check_protocol(args.city, ds, raw, starts)
    qs_np = np.arange(1, args.n_quant + 1) / (args.n_quant + 1)
    qs = torch.tensor(qs_np, dtype=torch.float32, device=dev)

    adj = None
    if args.method in ("stgcn", "stgcn_nograph", "gwnet"):
        dist = C.load_distance(args.city, ds)
        adj = knn_adjacency(torch.from_numpy(dist), k=args.knn).numpy()
        if args.method == "stgcn_nograph":
            adj = None

    model = build_model(args, N, args.n_quant, adj).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    w = batch_windows(N, args.batch_series)
    print(f"[{args.method}/{args.city}/seed{args.seed}] 站 {N} 窗 tr/va/te "
          f"{len(starts['tr'])}/{len(starts['va'])}/{len(starts['te'])} "
          f"batch_windows {w} params {n_par/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(args.seed)
    off = np.arange(C.SEQ_L)[None, :]

    out_m, out_d, out_c = C.out_dirs(args.method, args.city)
    ckpt = os.path.join(out_c, f"seed{args.seed}.pth")
    best, bad, t0 = float("inf"), 0, time.time()

    for ep in range(1, (0 if args.eval_only else args.epochs) + 1):
        model.train()
        perm = rng.permutation(len(starts["tr"]))
        run, nstep = 0.0, 0
        for b0 in range(0, min(len(perm), args.steps_per_epoch * w), w):
            idx = perm[b0:b0 + w]
            st = starts["tr"][idx]
            X = raw[st[:, None] + off].astype(np.float32)                  # [w,T,N]
            Y = raw[st[:, None] + C.SEQ_L + np.arange(C.PRE_LEN)[None, :]]
            Y = Y.transpose(0, 2, 1).astype(np.float32)
            xb, yb = torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev)
            kb = torch.from_numpy(keep["tr"][idx]).to(dev)
            yn, (loc, scale) = model(xb)
            pred = model.denorm(yn, loc, scale)
            loss = pinball_loss(pred, yb, qs, kb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss)
            nstep += 1
        sched.step()
        vl = eval_loss(model, raw, starts["va"], keep["va"], qs, dev, w)
        flag = ""
        if vl < best - 1e-5:
            best, bad = vl, 0
            torch.save(model.state_dict(), ckpt)
            flag = " *"
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1 or flag:
            print(f"  ep {ep:3d}  train {run/max(nstep,1):.4f}  valid {vl:.4f}  "
                  f"best {best:.4f}  {time.time()-t0:.0f}s{flag}", flush=True)
        if bad >= args.patience:
            print(f"  early stop @ ep {ep}", flush=True)
            break

    if args.eval_only:
        ep = 0
    elif not os.path.exists(ckpt):
        raise SystemExit(f"没有 checkpoint：{ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    te_starts = starts["te"]
    if args.limit_test:
        te_starts = te_starts[:args.limit_test]
    met = run_test(model, raw, te_starts, keep["te"][:len(te_starts)], smean,
                   qs_np, dev, w, with_diag=not args.no_diag)
    res = met.result()
    extra = dict(params=n_par, best_valid=best, epochs_run=ep,
                 elapsed_s=round(time.time() - t0, 1),
                 d_model=args.d_model, layers=args.layers, lr=args.lr,
                 knn=args.knn, n_test_windows=len(te_starts))
    C.save_metrics(args.method, args.city, args.seed, res, extra)
    if not args.no_diag:
        C.save_diag(args.method, args.city, args.seed, met)


if __name__ == "__main__":
    main()
