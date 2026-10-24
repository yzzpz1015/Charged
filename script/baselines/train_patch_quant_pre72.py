# -*- coding: utf-8 -*-
"""#6 的 pre72 版本：PatchTST + 99 分位头（每站单变量，raw 空间 pinball）。

现有 ``result/patchtst_quant_clean`` 只有 ``pre_len=24``，H=48/72 无法对齐。
这里用**同一模型结构**重训 pre_len=72，指标与 #7–#10 共用 ``StreamingMetrics``。

与旧训练脚本的两点差别（都是为了让口径与 #7–#10 一致）：
* clean mask 用**样本权重**（按 站×窗 置零），不是把 unclean 样本整条丢掉
  （JHB/SPO 的 clean 率只有 35%/58%，丢掉太伤）；
* 测试窗沿用 ``window_starts(test,72,72)``，H=24/48/72 取第 23/47/71 步。

用法::

    python3 script/baselines/train_patch_quant_pre72.py --city MEL --device 0
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402
from api.model.patchtst_quant import PatchTSTQuant  # noqa: E402

METHOD = "patchtst_quant_pre72"


def flat_xy(raw, starts):
    X, Y = C.flatten_windows(*C.build_windows(raw, starts))
    return X, Y


@torch.no_grad()
def eval_loss(model, X, Y, keep, qs, dev, bs=512):
    model.eval()
    tot, n = 0.0, 0
    q = torch.tensor(qs, dtype=torch.float32, device=dev)
    for b0 in range(0, len(X), bs):
        xb = torch.from_numpy(X[b0:b0 + bs]).to(dev)
        yb = torch.from_numpy(Y[b0:b0 + bs]).to(dev)
        wb = torch.from_numpy(keep[b0:b0 + bs]).to(dev).float()
        yn, (loc, scale) = model(xb)
        pred = model.denorm(yn, loc, scale)
        err = yb[:, :, None] - pred
        rho = torch.maximum(q * err, (q - 1.0) * err)          # [b,L,Q]
        w = wb[:, None, None]
        tot += float((rho * w).sum() / (w.sum() * rho.shape[1] * rho.shape[2]).clamp_min(1)) * len(xb)
        n += len(xb)
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps_per_epoch", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--n_quant", type=int, default=99)
    ap.add_argument("--limit_test", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--no_diag", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dev = C.torch_device(args.device)

    ds, raw, starts, keep, smean = C.load_city(args.city)
    N, _ = C.check_protocol(args.city, ds, raw, starts)
    qs = np.arange(1, args.n_quant + 1) / (args.n_quant + 1)

    Xtr, Ytr = flat_xy(raw, starts["tr"])
    Xva, Yva = flat_xy(raw, starts["va"])
    ktr = keep["tr"].reshape(-1).astype(np.float32)
    kva = keep["va"].reshape(-1).astype(np.float32)
    print(f"[{METHOD}/{args.city}/seed{args.seed}] 站 {N} "
          f"train {Xtr.shape} valid {Xva.shape} test 窗 {len(starts['te'])}",
          flush=True)

    model = PatchTSTQuant(seq_len=C.SEQ_L, pre_len=C.PRE_LEN, quantiles=qs).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    out_m, out_d, out_c = C.out_dirs(METHOD, args.city)
    ckpt = os.path.join(out_c, f"seed{args.seed}.pth")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    q = torch.tensor(qs, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(args.seed)
    best, bad, t0 = float("inf"), 0, time.time()

    for ep in range(1, (0 if args.eval_only else args.epochs) + 1):
        model.train()
        perm = rng.permutation(len(Xtr))
        run, nstep = 0.0, 0
        for b0 in range(0, min(len(perm), args.steps_per_epoch * args.batch_size),
                       args.batch_size):
            idx = perm[b0:b0 + args.batch_size]
            xb = torch.from_numpy(Xtr[idx]).to(dev)
            yb = torch.from_numpy(Ytr[idx]).to(dev)
            wb = torch.from_numpy(ktr[idx]).to(dev).float()
            yn, (loc, scale) = model(xb)
            pred = model.denorm(yn, loc, scale)
            err = yb[:, :, None] - pred
            rho = torch.maximum(q * err, (q - 1.0) * err)
            w = wb[:, None, None]
            loss = (rho * w).sum() / (w.sum() * rho.shape[1] * rho.shape[2]).clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss)
            nstep += 1
        sched.step()
        vl = eval_loss(model, Xva, Yva, kva, qs, dev, args.batch_size)
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
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    te = starts["te"]
    if args.limit_test:
        te = te[:args.limit_test]
    met = C.make_metrics(args.city, raw, te, keep["te"][:len(te)], smean, qs,
                         with_diag=not args.no_diag)
    Xte, _ = flat_xy(raw, te)
    w = max(1, args.batch_size // N)
    batch = w * N
    with torch.no_grad():
        for c0 in range(0, len(te), w):
            c1 = min(c0 + w, len(te))
            xb = torch.from_numpy(Xte[c0 * N:c1 * N]).to(dev)
            yn, (loc, scale) = model(xb)
            pred = model.denorm(yn, loc, scale).float().cpu().numpy()
            pred = pred.reshape(c1 - c0, N, C.PRE_LEN, args.n_quant)
            met.update_quantiles(c0, pred)

    res = met.result()
    extra = dict(params=n_par, best_valid=best,
                 elapsed_s=round(time.time() - t0, 1), n_test_windows=len(te),
                 epochs_run=ep if not args.eval_only else 0, pre_len=C.PRE_LEN,
                 note="same arch as pre24 #6, retrained at pre_len=72")
    C.save_metrics(METHOD, args.city, args.seed, res, extra)
    if not args.no_diag:
        C.save_diag(METHOD, args.city, args.seed, met)


if __name__ == "__main__":
    main()
