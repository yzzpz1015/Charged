# -*- coding: utf-8 -*-
"""#21 OpenCity-Plus：公开权重时空 FM + 逐站分位头（72→72）。

官方模型是 288→288 点预测；本协议 72→72，只替换末层（见
``api/model/opencity_quant.py``）。默认 ``--ft head`` 沿用官方 ``eval`` 口径：
冻结主干、只训分位头；``--ft full`` 为主干低学习率微调。

用法::

    python3 script/baselines/train_opencity.py --city MEL --seed 0 --device 1
    python3 script/baselines/train_opencity.py --city MEL --seed 0 --device 1 --eval_only

产物：``result/baselines/{ckpt,metrics}/opencity/<city>/...``；加 ``BASELINE_TAG``
时只改评测产物目录，不覆盖旧 JSON。
"""
import argparse
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402
from api.model.opencity_quant import (  # noqa: E402
    OpenCityQuant, build_graph, load_pretrained,
)
from api.model.quant_head import pinball_loss  # noqa: E402

METHOD = "opencity"
PRETRAIN = os.path.join(BASE, "external", "OpenCity", "model_weights",
                        "OpenCity", "OpenCity-plus.pth")
OFF_IN = np.arange(C.SEQ_L, dtype=np.int64)[None, :]


def make_args(city, A, A_mx, lap, n_quant):
    return types.SimpleNamespace(
        adj_mx_dict={city: torch.from_numpy(A_mx)},
        sh_mx_dict={city: torch.from_numpy(A)},
        lap_mx_dict={city: torch.from_numpy(lap)},
        embed_dim=512, skip_dim=512, lape_dim=8,
        geo_num_heads=0, sem_num_heads=0, tc_num_heads=16, t_num_heads=16,
        mlp_ratio=2, qkv_bias=True, drop=0.1, attn_drop=0.3, drop_path=0.0,
        s_attn_size=3, t_attn_size=1, enc_depth=6, type_ln="pre",
        input_window=C.SEQ_L, output_window=C.PRE_LEN, far_mask_delta=5,
        n_quant=n_quant,
    )


def time_arrays(ds):
    t = pd.DatetimeIndex(ds.time)
    # 官方 time_add：hour 0 -> 60, hour 23 -> 1440；ISO 周一=1
    day = ((t.hour.values + 1) * 60).astype(np.int64)
    week = (t.weekday.values + 1).astype(np.int64)   # 周一=1
    return day, week


def make_batch(raw, day, week, starts, device):
    """starts [w] -> input/lbls [w,72,N,3], y [w,N,72]（raw 空间）。"""
    s = np.asarray(starts)[:, None] + OFF_IN              # [w,72]
    w, T = s.shape
    N = raw.shape[1]
    inp = np.empty((w, T, N, 3), np.float32)
    inp[..., 0] = raw[s]
    inp[..., 1] = day[s][:, :, None]
    inp[..., 2] = week[s][:, :, None]

    sf = s + C.SEQ_L
    lbl = np.zeros((w, T, N, 3), np.float32)
    lbl[..., 1] = day[sf][:, :, None]
    lbl[..., 2] = week[sf][:, :, None]

    y = raw[sf].transpose(0, 2, 1).astype(np.float32)     # [w,N,72]
    return (torch.from_numpy(inp).to(device),
            torch.from_numpy(lbl).to(device),
            torch.from_numpy(y).to(device))


def masked_pinball(model, raw, day, week, starts, keep, qs, city, device, w):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b0 in range(0, len(starts), w):
            st = starts[b0:b0 + w]
            inp, lbl, y = make_batch(raw, day, week, st, device)
            pred = model(inp, lbl, city)
            kb = torch.from_numpy(keep[b0:b0 + len(st)]).to(device)
            loss = pinball_loss(pred, y, qs, kb)
            tot += float(loss) * len(st)
            n += len(st)
    return tot / max(n, 1)


def auto_windows(N):
    if N <= 80:
        return 8
    if N <= 300:
        return 4
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--ft", choices=("head", "full"), default="head")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="head=1e-3；full 建议 1e-4")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_windows", type=int, default=0,
                    help="0 = 按站数自动（<=80:8, <=300:4, >300:1）")
    ap.add_argument("--steps_per_epoch", type=int, default=0,
                    help="0 = 全训练集；>0 = 每 epoch 固定窗口数")
    ap.add_argument("--n_quant", type=int, default=99)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--limit_test", type=int, default=0, help=">0 只跑前 N 个测试窗")
    ap.add_argument("--no_diag", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = C.torch_device(args.device)

    ds, raw, starts, keep, smean = C.load_city(args.city)
    N, _ = C.check_protocol(args.city, ds, raw, starts)
    qs_np = np.arange(1, args.n_quant + 1) / (args.n_quant + 1)
    qs = torch.tensor(qs_np, dtype=torch.float32, device=device)
    day, week = time_arrays(ds)

    dist = C.load_distance(args.city, ds)
    A, A_mx, lap = build_graph(dist, k=args.knn)
    model_args = make_args(args.city, A, A_mx, lap, args.n_quant)
    model = OpenCityQuant(model_args, [args.city], device, dim_in=1,
                          n_quant=args.n_quant, hidden=args.hidden,
                          dropout=args.dropout).to(device)
    n_loaded = load_pretrained(model, PRETRAIN)

    trainable = list(model.qhead.parameters())
    if args.ft == "head":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.qhead.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True
        trainable = [p for p in model.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in trainable)

    w = args.batch_windows or auto_windows(N)
    out_m, out_d, out_c = C.out_dirs(METHOD, args.city)
    ckpt = os.path.join(out_c, f"seed{args.seed}.pth")
    print(f"[{METHOD}/{args.city}/seed{args.seed}] 站 {N} 窗 tr/va/te "
          f"{len(starts['tr'])}/{len(starts['va'])}/{len(starts['te'])} "
          f"batch_windows {w} ft {args.ft} params {n_par/1e6:.2f}M "
          f"trainable {n_tr/1e6:.2f}M pretrained {n_loaded} tensors", flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    ckpt_loaded = args.eval_only
    best, bad, ep = float("inf"), 0, 0

    if not args.eval_only:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        rng = np.random.default_rng(args.seed)
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            if args.ft == "head":
                model.eval()          # 冻结主干；Dropout/DropPath 全关
                model.qhead.train()
            else:
                model.train()
            perm = rng.permutation(len(starts["tr"]))
            if args.steps_per_epoch:
                perm = perm[:args.steps_per_epoch * w]
            run, nstep = 0.0, 0
            for b0 in range(0, len(perm), w):
                idx = perm[b0:b0 + w]
                inp, lbl, y = make_batch(raw, day, week, starts["tr"][idx], device)
                pred = model(inp, lbl, args.city)
                kb = torch.from_numpy(keep["tr"][idx]).to(device)
                loss = pinball_loss(pred, y, qs, kb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                opt.step()
                run += float(loss)
                nstep += 1
            sched.step()
            vl = masked_pinball(model, raw, day, week, starts["va"],
                                keep["va"], qs, args.city, device, w)
            flag = ""
            if vl < best - 1e-6:
                best, bad = vl, 0
                torch.save(model.state_dict(), ckpt)
                flag = " *"
            else:
                bad += 1
            if ep % 2 == 0 or ep == 1 or flag:
                print(f"  ep {ep:3d}  train {run/max(nstep,1):.2f}  valid {vl:.2f}  "
                      f"best {best:.2f}  {time.time()-t0:.0f}s{flag}", flush=True)
            if bad >= args.patience:
                print(f"  early stop @ ep {ep}", flush=True)
                break

    if not os.path.exists(ckpt):
        raise SystemExit(f"没有 checkpoint：{ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    te_starts = starts["te"][:args.limit_test] if args.limit_test else starts["te"]
    te_keep = keep["te"][:len(te_starts)]
    met = C.make_metrics(None, raw, te_starts, te_keep, smean, qs_np,
                         with_diag=not args.no_diag)
    t1 = time.time()
    with torch.no_grad():
        for c0 in range(0, len(te_starts), w):
            st = te_starts[c0:c0 + w]
            inp, lbl, _ = make_batch(raw, day, week, st, device)
            pred = model(inp, lbl, args.city).float().cpu().numpy()
            met.update_quantiles(c0, pred)
    res = met.result()
    extra = dict(params=n_par, trainable=n_tr, best_valid=best, epochs_run=ep,
                 elapsed_s=round(time.time() - t1, 1), ft=args.ft, lr=args.lr,
                 n_quant=args.n_quant, hidden=args.hidden, knn=args.knn,
                 batch_windows=w, n_test_windows=len(te_starts),
                 pretrained=os.path.relpath(PRETRAIN, BASE),
                 pretrained_tensors=n_loaded,
                 note="OpenCity-Plus head-only (official eval recipe)" if args.ft == "head"
                      else "OpenCity-Plus full fine-tune")
    C.save_metrics(METHOD, args.city, args.seed, res, extra)
    if not args.no_diag:
        C.save_diag(METHOD, args.city, args.seed, met)


if __name__ == "__main__":
    main()
