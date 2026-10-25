# -*- coding: utf-8 -*-
"""UniTS-FT + 残差池（参照 #1/#5 的池，不是官方 UniST）。

官方 UniTS 是“多变量通用时序模型”，不是图模型；这里把它当预训练主干：
* 加载官方 ``units_x64_supervised_checkpoint.pth``（d_model=64）；
* 为新城市新建一个 dataset prompt（enc_in=N，任意站数），其余权重全部迁移；
* 用我们的 pre72 窗口与 clean mask 做少样本微调（归一化 MSE，逐窗只用手历史统计）；
* 概率输出走 **残差池**（``kairos_prob_eval.bootstrap_variants`` 的 roll28），
  与 #1/#5 同一条池、同一组测试窗，因此表里应标为 ``UniTS-FT + pool``。

用法::

    python3 script/baselines/train_unist.py --city MEL --device 0
    python3 script/baselines/train_unist.py --city MEL --device 0 --epochs 1 --steps_per_epoch 2
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
sys.path.insert(0, os.path.join(BASE, "external", "UniTS"))

import common as C  # noqa: E402
import kairos_prob_eval as ke  # noqa: E402
import prob_baseline as pb  # noqa: E402
from models.UniTS import Model as UniTSModel  # noqa: E402

SEQ_L, PRE_LEN = C.SEQ_L, C.PRE_LEN
JS = (23, 47, 71)


def build_model(city, n_station, ckpt):
    import argparse as _ap
    args = _ap.Namespace(prompt_num=10, stride=16, patch_len=16,
                         d_model=64, e_layers=3, n_heads=8, dropout=0.1)
    tag = f"Charged_{city}_fc"
    cfg = {  # configs_list 里每个任务是一个 (task_data_name, config)
        "task_name": "long_term_forecast", "dataset": f"Charged_{city}",
        "data": "custom", "seq_len": SEQ_L, "pred_len": PRE_LEN,
        "enc_in": n_station, "label_len": 0, "c_out": None, "embed": "timeF",
    }
    model = UniTSModel(args, [(tag, cfg)], pretrain=False)
    sd = model.state_dict()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ck = {k.replace("module.", "", 1): v for k, v in ck.items()}
    matched = {k: v for k, v in ck.items()
               if k in sd and tuple(sd[k].shape) == tuple(v.shape)}
    missing = [k for k in sd if k not in matched]
    model.load_state_dict(matched, strict=False)
    print(f"[unist] 迁移 {len(matched)}/{len(sd)} 个参数；"
          f"新初始化 {missing}（只应是本城 prompt/mask）", flush=True)
    if len(missing) > 2:
        raise AssertionError(f"意外缺失参数：{missing[:10]}")
    return model


def get_xy(raw, starts):
    off = np.arange(SEQ_L)[None, :]
    off_y = np.arange(PRE_LEN)[None, :]
    X = raw[starts[:, None] + off].astype(np.float32)
    Y = raw[starts[:, None] + SEQ_L + off_y].astype(np.float32)
    return X, Y


def batch_loss(model, X, Y, keep, sig, dev):
    """加权的 raw-space MSE：逐站除以**训练期**标准差（下限 1 kWh/h）。

    不用逐窗历史 std 作分母：零历史窗的 std→0，会让归一化误差爆到 1e10
    （冒烟时实测 train loss 6.6e11），训练直接失稳。训练期逐站 std 只来自
    train 段，无泄漏，且对恒零站给出 1 kWh/h 的绝对容差。
    """
    xb = torch.from_numpy(X).to(dev)
    yb = torch.from_numpy(Y).to(dev)
    w = torch.from_numpy(keep.astype(np.float32)).to(dev)[:, None, :]
    pred = model.forecast(xb, None, 0)                     # raw 空间 [B,72,N]
    err = ((pred - yb) / sig) ** 2
    return (err * w).sum() / w.expand_as(err).sum().clamp_min(1.0)


@torch.no_grad()
def predict(model, X, dev, bs):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(dev)
        out.append(model.forecast(xb, None, 0).cpu().numpy())
    return np.concatenate(out, 0)                          # [M,72,N]


def eval_windows(model, raw, starts, keep, sig, dev, n_win, bs):
    model.eval()
    idx = np.unique(np.linspace(0, len(starts) - 1,
                               min(n_win, len(starts))).round().astype(int))
    X, Y = get_xy(raw, starts[idx])
    tot, n = 0.0, 0
    for i in range(0, len(X), bs):
        tot += float(batch_loss(model, X[i:i + bs], Y[i:i + bs],
                               keep[idx][i:i + bs], sig, dev)) * len(X[i:i + bs])
        n += len(X[i:i + bs])
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--steps_per_epoch", type=int, default=80)
    ap.add_argument("--batch_windows", type=int, default=16)
    ap.add_argument("--val_windows", type=int, default=128)
    ap.add_argument("--pred_batch", type=int, default=16)
    ap.add_argument("--ckpt", default=os.path.join(
        BASE, "external", "weights", "units_x64_supervised_checkpoint.pth"))
    ap.add_argument("--out", default=os.path.join(BASE, "result", "baselines"))
    ap.add_argument("--method", default="unist_pool")
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dev = C.torch_device(args.device)

    ds, raw, starts, keep, smean = C.load_city(args.city)
    N = raw.shape[1]
    C.check_protocol(args.city, ds, raw, starts)
    # 逐站归一化尺度：只取 train 段；下限 1 kWh/h（恒零站的绝对容差）
    sstd = np.nanstd(raw[ds.split_of_row == 0], axis=0).astype(np.float32)
    sig = torch.from_numpy(np.maximum(np.nan_to_num(sstd), 1.0)).to(C.torch_device(args.device))[None, None, :]
    # 残差池用的时间/可用行数组，与 kairos_prob_eval.run_city 完全同源
    y0, t0 = pb.load(args.city)
    ok, trm, vam, tem, _, _ = pb.split_rows(y0, t0)
    cols, _ = pb.live_cols(y0, trm)
    if len(cols) != N:
        raise AssertionError(f"{args.city} 站数不一致：common {N} vs pb {len(cols)}")
    y = y0[:, cols]
    hod = t0.hour.values
    wkd = (t0.dayofweek.values >= 5).astype(np.int8)
    for nm in ("va", "te"):
        a = raw[starts[nm][:, None] + SEQ_L + np.array(JS)[None, :]]
        b = np.stack([y[starts[nm] + SEQ_L + j] for j in JS], 1)
        if not np.allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True):
            raise AssertionError(f"{args.city} {nm} 站序/取值与 pb 不一致")

    out_c = os.path.join(args.out, "ckpt", args.method, args.city)
    os.makedirs(out_c, exist_ok=True)
    ckpt = os.path.join(out_c, f"seed{args.seed}.pth")
    model = build_model(args.city, N, args.ckpt).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    w = args.batch_windows
    print(f"[{args.method}/{args.city}/seed{args.seed}] 站 {N} 窗 "
          f"{len(starts['tr'])}/{len(starts['va'])}/{len(starts['te'])} "
          f"batch {w} params {n_par/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(args.seed)
    best, bad, t0s, ep_run = float("inf"), 0, time.time(), 0

    for ep in range(1, (0 if args.eval_only else args.epochs) + 1):
        ep_run = ep
        model.train()
        perm = rng.permutation(len(starts["tr"]))
        run, nstep = 0.0, 0
        for b0 in range(0, min(len(perm), args.steps_per_epoch * w), w):
            idx = perm[b0:b0 + w]
            X, Y = get_xy(raw, starts["tr"][idx])
            loss = batch_loss(model, X, Y, keep["tr"][idx], sig, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss)
            nstep += 1
        sched.step()
        vl = eval_windows(model, raw, starts["va"], keep["va"], sig, dev,
                          args.val_windows, w)
        flag = ""
        if vl < best - 1e-6:
            best, bad = vl, 0
            torch.save(model.state_dict(), ckpt)
            flag = " *"
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1 or flag:
            print(f"  ep {ep:3d}  train {run/max(nstep,1):.4f}  valid {vl:.4f}  "
                  f"best {best:.4f}  {time.time()-t0s:.0f}s{flag}", flush=True)
        if bad >= args.patience:
            print(f"  early stop @ ep {ep}", flush=True)
            break

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    Xva, _ = get_xy(raw, starts["va"])
    Xte, _ = get_xy(raw, starts["te"])
    pva = predict(model, Xva, dev, args.pred_batch)        # [M,72,N]
    pte = predict(model, Xte, dev, args.pred_batch)
    yte = raw[starts["te"][:, None] + SEQ_L + np.array(JS)[None, :]]  # [M,3,N]
    prov = {"va": {j: pva[:, j, :] for j in JS},
            "te": {j: pte[:, j, :] for j in JS}}
    rng2 = np.random.default_rng(args.seed)
    res = ke.bootstrap_variants(prov, y, starts, ok, hod, wkd, JS, smean, N,
                                rng2, keep_te=keep["te"])

    tag = os.environ.get("BASELINE_TAG", "").strip()
    mdir = os.path.join(args.out, "metrics" + (f"_{tag}" if tag else ""),
                        args.method, args.city)
    os.makedirs(mdir, exist_ok=True)
    common = dict(method=args.method, city=args.city, seed=args.seed,
                  seq_l=SEQ_L, pre_len=PRE_LEN, n_sample=ke.M_SAMPLE,
                  stations=N, params=n_par, best_valid=best,
                  epochs_run=ep_run, elapsed_s=round(time.time() - t0s, 1),
                  d_model=64, layers=3, lr=args.lr, knn=0,
                  n_test_windows=len(starts["te"]),
                  unist_ckpt=os.path.basename(args.ckpt),
                  prob_head="roll28_residual_pool")
    for j, d in res["roll28"].items():
        pc, yc, mc = pte[:, j, :], yte[:, JS.index(j), :], keep["te"]
        e = pc[mc] - yc[mc]
        npt = max(int(mc.sum()), 1)
        ss_tot = float(((yc[mc] - yc[mc].mean()) ** 2).sum())
        rec = dict(common, horizon=j + 1,
                   crps_norm_sum=d["crps_norm_sum"],
                   crps_sum_norm=d["crps_sum_norm"],
                   crps_nz=d["crps_nz"], nz_rate=d["nz_rate"],
                   n_scored_windows=d["n_used"],
                   picp50=d["picp50"], picp80=d["picp80"], picp90=d["picp90"],
                   mpiw50=d["mpiw50"], mpiw80=d["mpiw80"], mpiw90=d["mpiw90"],
                   # 点预测 = UniTS 主干输出（residual pool 只加概率层）
                   mae=float(np.abs(e).mean()),
                   rmse=float(np.sqrt((e ** 2).mean())),
                   r2=float(1.0 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
                   n_point_cells=npt)
        with open(os.path.join(mdir, f"H{j+1}_seed{args.seed}.json"), "w") as f:
            import json
            json.dump(rec, f, ensure_ascii=False, indent=1)
        print(f"[{args.method}/{args.city}/seed{args.seed}] H{j+1}  "
              f"crpsΣ {d['crps_norm_sum']:.4f}  crpsΣsum {d['crps_sum_norm']:.4f}  "
              f"crps_nz {d['crps_nz']:.4f}  PICP50 {d['picp50']:.3f}  "
              f"PICP90 {d['picp90']:.3f}", flush=True)
    for j in JS:  # 参照：不用滚动池的 honest 版本
        d = res["honest"][j]
        print(f"    [honest] H{j+1} crpsΣ {d['crps_norm_sum']:.4f}", flush=True)


if __name__ == "__main__":
    main()
