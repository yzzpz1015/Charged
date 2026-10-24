# -*- coding: utf-8 -*-
"""#16 TSDiff：整段序列扩散 + 历史硬引导（实现见 api/model/tsdiff.py）。

统一协议同其他 baseline；测试每窗采样 200 条轨迹（DDIM 50 步），
``StreamingMetrics.update_samples``。

差异（表注）：hard self-guidance（替代论文的软梯度引导）、DDIM 50 步、
不做 unconditional 预训练。

用法::

    python3 script/baselines/train_tsdiff.py --method tsdiff --city MEL --device 0
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
from api.model.tsdiff import TSDiffModel  # noqa: E402

OFF = np.arange(C.SEQ_L)[None, :]
OFF_Y = np.arange(C.PRE_LEN)[None, :]


def batch_windows(N, budget):
    return int(max(1, min(32, budget // max(N, 1))))


def get_xy(raw, starts, idx=None):
    st = starts if idx is None else starts[idx]
    X = raw[st[:, None] + OFF].astype(np.float32)
    Y = raw[st[:, None] + C.SEQ_L + OFF_Y].astype(np.float32)
    return X, Y


def prep(model, X, Y, keep, dev):
    """-> z0 [R,144] 归一化整段, weight [R], loc/scale [R,1]"""
    w, T, N = X.shape
    xb = torch.from_numpy(X).to(dev)
    yb = torch.from_numpy(Y).to(dev)
    loc, scale = model.revin.stats(xb)                       # [w,1,N]
    xn = (xb - loc) / scale
    yn = (yb - loc) / scale
    z = torch.cat([xn, yn], dim=1)                           # [w,T+L,N]
    z0 = z.permute(0, 2, 1).reshape(w * N, T + C.PRE_LEN)
    wgt = torch.from_numpy(keep.reshape(-1).astype(np.float32)).to(dev)
    lr = loc.permute(0, 2, 1).reshape(w * N, 1)
    sr = scale.permute(0, 2, 1).reshape(w * N, 1)
    return z0, wgt, lr, sr


@torch.no_grad()
def eval_loss(model, raw, starts, keep, dev, n_win, w):
    model.eval()
    idx = np.unique(np.linspace(0, len(starts) - 1,
                               min(n_win, len(starts))).round().astype(int))
    tot, nw = 0.0, 0
    for b0 in range(0, len(idx), w):
        j = idx[b0:b0 + w]
        X, Y = get_xy(raw, starts, j)
        z0, wgt, _, _ = prep(model, X, Y, keep[j], dev)
        tot += float(model.p_loss(z0, wgt)) * len(j)
        nw += len(j)
    model.train()
    return tot / max(nw, 1)


@torch.no_grad()
def run_test(model, raw, starts, keep, smean, dev, w, n_sample,
             ddim_steps, row_chunk, eta, with_diag=True):
    model.eval()
    qs = np.arange(1, 100) / 100.0
    met = C.make_metrics(None, raw, starts, keep, smean, qs,
                         with_diag=with_diag)
    t0 = time.time()
    for c0 in range(0, len(starts), w):
        c1 = min(c0 + w, len(starts))
        X, _ = get_xy(raw, starts[c0:c1])
        ww, T, N = X.shape
        xb = torch.from_numpy(X).to(dev)
        loc, scale = model.revin.stats(xb)
        xn = ((xb - loc) / scale).permute(0, 2, 1).reshape(ww * N, T)
        lr = loc.permute(0, 2, 1).reshape(ww * N, 1)
        sr = scale.permute(0, 2, 1).reshape(ww * N, 1)
        S = model.sample(xn, n_samples=n_sample, ddim_steps=ddim_steps,
                         row_chunk=row_chunk, eta=eta)        # [S, ww*N, L]
        S = model.denorm(S, lr, sr).clamp_min(0.0)
        S = S.view(n_sample, ww, N, C.PRE_LEN).permute(1, 2, 3, 0)
        met.update_samples(c0, S.float().cpu().numpy())
        if (c0 // max(w, 1)) % 10 == 0:
            el = time.time() - t0
            print(f"    采样 {c1}/{len(starts)} 窗  {el:.0f}s  "
                  f"ETA {el/max(c1,1)*(len(starts)-c1)/60:.0f}min", flush=True)
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="tsdiff", choices=["tsdiff"])
    ap.add_argument("--city", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps_per_epoch", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_series", type=int, default=4096)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--diffusion_steps", type=int, default=1000)
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=1.0,
                    help="DDIM eta：1=补齐跳步随机性（推荐），0=确定性")
    ap.add_argument("--n_sample", type=int, default=C.M_SAMPLE)
    ap.add_argument("--sample_windows", type=int, default=0)
    ap.add_argument("--row_chunk", type=int, default=32768)
    ap.add_argument("--val_windows", type=int, default=256)
    ap.add_argument("--limit_test", type=int, default=0)
    ap.add_argument("--no_diag", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dev = C.torch_device(args.device)

    ds, raw, starts, keep, smean = C.load_city(args.city)
    N, M = C.check_protocol(args.city, ds, raw, starts)
    model = TSDiffModel(N, C.SEQ_L, C.PRE_LEN, d_model=args.d_model,
                        n_layers=args.layers, dropout=args.dropout,
                        diffusion_steps=args.diffusion_steps).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    w = batch_windows(N, args.batch_series)
    sw = args.sample_windows or max(1, min(w, args.batch_series // (N * 8)))
    print(f"[{args.method}/{args.city}/seed{args.seed}] 站 {N} 窗 "
          f"{len(starts['tr'])}/{len(starts['va'])}/{len(starts['te'])} "
          f"batch_windows {w} 采样块 {sw} params {n_par/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(args.seed)
    _, _, out_c = C.out_dirs(args.method, args.city)
    ckpt = os.path.join(out_c, f"seed{args.seed}.pth")
    best, bad, t0 = float("inf"), 0, time.time()

    for ep in range(1, (0 if args.eval_only else args.epochs) + 1):
        model.train()
        perm = rng.permutation(len(starts["tr"]))
        run, nstep = 0.0, 0
        for b0 in range(0, min(len(perm), args.steps_per_epoch * w), w):
            idx = perm[b0:b0 + w]
            X, Y = get_xy(raw, starts["tr"], idx)
            z0, wgt, _, _ = prep(model, X, Y, keep["tr"][idx], dev)
            loss = model.p_loss(z0, wgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss)
            nstep += 1
        sched.step()
        vl = eval_loss(model, raw, starts["va"], keep["va"], dev,
                       args.val_windows, w)
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

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    te_all = starts["te"]
    if args.limit_test:
        # 与 csdi_official 一致：等间隔抽窗，而不是取前 N 个（避免只覆盖测试期前段）
        sel = np.unique(np.linspace(0, len(te_all) - 1,
                                   min(args.limit_test, len(te_all))
                                   ).round().astype(int))
        te, keep_te = te_all[sel], keep["te"][sel]
    else:
        te, keep_te = te_all, keep["te"]
    met = run_test(model, raw, te, keep_te, smean, dev, sw,
                   args.n_sample, args.ddim_steps, args.row_chunk, eta=args.eta,
                   with_diag=not args.no_diag)
    res = met.result()
    extra = dict(params=n_par, best_valid=best,
                 elapsed_s=round(time.time() - t0, 1),
                 d_model=args.d_model, layers=args.layers,
                 diffusion_steps=args.diffusion_steps,
                 ddim_steps=args.ddim_steps, eta=args.eta, n_sample=args.n_sample,
                 n_test_windows=len(te), guidance="hard_history_projection")
    C.save_metrics(args.method, args.city, args.seed, res, extra)
    if not args.no_diag:
        C.save_diag(args.method, args.city, args.seed, met)


if __name__ == "__main__":
    main()
