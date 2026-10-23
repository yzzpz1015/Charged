# -*- coding: utf-8 -*-
"""TimeDiff baseline（逐站单变量版本），MEL / SPO。

为什么是逐站
------------
做法④⑤ 是「整块 [站 × H] 扩散」，站间关联由模型自己学；本脚本是**单站条件扩散**
（TimeDiff, Shen & Kwok ICML 2023 的非自回归条件扩散 + future mixup），
所以它没有跨站耦合自由度，和做法③一样只能产出边际。用来回答的是：
「把画师从 CSDI/zidiff 换成更强的 TimeDiff，边际 crpsΣ 能不能追上分位头？」
联合口径（crpsΣsum）按独立耦合拼，仅作对照。

条件/归一化口径与其它方法对齐
------------------------------
* 窗口起点来自 ``EVDataset.window_starts``，与基线/③/④/⑤同一实现；
* 合成直线段按 **站×窗** 剔除（``straight_window_keep``）；
* 逐站 mean/std 只用**干净训练窗**估，避免验证/测试泄漏；
* 训练在全局标准化空间，TimeDiff 内部再减最近 label_len 的均值；
* 评测分母 smean 与 ``prob_baseline.live_cols`` 同口径。

源码
----
官方代码未公开；使用第三方实现 https://github.com/dlgudwn1219/TimeDiff
（DLinear pretrain + 条件 DDPM / CNN U-Net）。这里只保留模型与前向，
数据/训练/评测全部按本项目的口径重写。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))
sys.path.insert(0, "/tmp/TimeDiff_repo")

import kairos_prob_eval as ke                                        # noqa: E402
import prob_baseline as pb                                           # noqa: E402
from api.dataset.common import (EVDataset, SPLIT_TRAIN, SPLIT_VALID,  # noqa: E402
                                SPLIT_TEST)
from api.utils import straight_window_keep                           # noqa: E402
from models_diffusion.DDPM import Model as TimeDiffModel             # noqa: E402

SEQ_L = 72
PRE_LEN = 24


class Args:
    """TimeDiff 模型需要的字段（默认值取自仓库 README 的推荐命令）。"""

    features = "S"                 # 逐站单变量；f_dim=0 走全通道
    seq_len = SEQ_L
    label_len = SEQ_L
    pred_len = PRE_LEN
    num_vars = 1
    diff_steps = 100
    UNet_Type = "CNN"
    type_sampler = "none"          # 用完整 p_sample 链，保留随机性
    parameterization = "x_start"
    use_window_normalization = True
    our_ddpm_clip = 100.0
    ablation_study_case = "none"
    ablation_study_F_type = "Linear"
    ablation_study_masking_type = "none"
    ablation_study_masking_tau = 0.9
    weight_pred_loss = 0.0
    ddpm_inp_embed = 256
    ddpm_dim_diff_steps = 256
    ddpm_channels_conv = 256
    ddpm_channels_fusion_I = 256
    ddpm_layers_inp = 5
    ddpm_layers_I = 5
    ddpm_layers_II = 10
    cond_ddpm_num_layers = 30
    cond_ddpm_channels_conv = 32
    sample_times = 24
    # 第三方实现默认在训练时做 element-wise future mixup（把真值混进 pred_out，
    # 再送进去噪网络）。实测它会让网络学会"抄真值"，推理时真值缺席 -> 网络
    # 几乎不看 x_t，24 条样本互差 0.02，采样塌成点质量（SPO crpsΣ 0.815）。
    # 关掉 mixup 后退化成标准条件 DDPM，仍是 DLinear 先验 + 非自回归条件扩散。
    no_mixup = True
    vis_ar_part = 0
    out_figures = 0
    vis_MTS_analysis = 0
    use_amp = False


def build_data(city):
    ds = EVDataset(feature="volume", auxiliary="None",
                   data_path=f"{BASE}/data/{city}_remove_const/")
    ds.split_cross_validation(fold=6, total_fold=6, train_ratio=0.8, valid_ratio=0.1)
    raw = ds.feat.astype(np.float64)                       # [T, N]
    N = raw.shape[1]
    starts = {k: ds.window_starts(sid, SEQ_L, PRE_LEN)
              for k, sid in (("train", SPLIT_TRAIN), ("valid", SPLIT_VALID),
                             ("test", SPLIT_TEST))}
    keep = {k: straight_window_keep(raw, starts[k], SEQ_L, PRE_LEN) for k in starts}

    off_in = np.arange(SEQ_L)[None, :]
    off_out = np.arange(PRE_LEN)[None, :]

    def blocks(st):
        X = raw[st[:, None] + off_in]                      # [M, 72, N]
        Y = raw[st[:, None] + SEQ_L + off_out]             # [M, 24, N]
        return X.transpose(0, 2, 1), Y.transpose(0, 2, 1)  # [M, N, T]

    data = {"raw": raw, "starts": starts, "keep": keep, "N": N}
    for k in starts:
        X, Y = blocks(starts[k])
        data[f"X_{k}"] = X.astype(np.float32)
        data[f"Y_{k}"] = Y.astype(np.float32)

    # 逐站尺度：只用干净训练窗（合成直线段是伪数据，不能进统计量）。
    # 这个尺度只作**下限**；真正用于归一化的是逐窗历史 72h 的 std（见下），
    # 与其它模型的 RevIN 同思路。全局尺度在 MEL 站 45 上会爆炸：该站训练期
    # 近乎恒定（clean std 0.31），验证期却出现 130 的平台，z 可达 284，
    # 几个格子就能独占整条验证损失。
    ktr = keep["train"]
    ncell = ktr.sum(0).astype(np.float64) * SEQ_L
    mean = (data["X_train"] * ktr[:, :, None]).sum((0, 2)) / np.maximum(ncell, 1)
    var = (((data["X_train"] - mean[None, :, None]) * ktr[:, :, None]) ** 2).sum((0, 2))
    var /= np.maximum(ncell, 1)
    sd = np.sqrt(var)
    pos = sd[np.isfinite(sd) & (sd > 0)]
    floor = max(1e-2, 0.02 * float(np.median(pos))) if pos.size else 1e-2
    sd = np.maximum(sd, floor)
    mean = np.where(np.isfinite(mean), mean, 0.0)

    # 逐窗尺度只用**历史 72h**（因果）：std 的 10% 逐站下限防常数窗。
    scales = {}
    for k in starts:
        hs = data[f"X_{k}"].std(axis=2)                       # [M,N]
        scales[k] = np.maximum(hs, 0.1 * sd[None, :]).astype(np.float32)
    data["scales"] = scales

    # 评测分母：与 prob_baseline.live_cols 同一口径（训练段可用行的逐站均值）
    tr_rows = ds.split_of_row == SPLIT_TRAIN
    smean = np.nanmean(raw[tr_rows], axis=0)

    data.update(mean=mean, sd=sd, smean=smean)
    print(f"[{city}] 站 {N}  窗口 train={len(starts['train'])} "
          f"valid={len(starts['valid'])} test={len(starts['test'])}  "
          f"clean 保留 train={100 * ktr.mean():.1f}% "
          f"test={100 * keep['test'].mean():.1f}%", flush=True)
    print(f"[{city}] 逐站 std 中位 {np.median(sd):.3f}（floor {floor:.3f}），"
          f"smean 中位 {np.median(smean):.3f}", flush=True)
    return data


def flat_xy(data, split, clean=True):
    sc = data["scales"][split][:, :, None]
    X = np.clip(data[f"X_{split}"] / sc, -20, 20)
    Y = np.clip(data[f"Y_{split}"] / sc, -20, 20)
    X = X.reshape(-1, SEQ_L)
    Y = Y.reshape(-1, PRE_LEN)
    if clean:
        m = data["keep"][split].reshape(-1)
        X, Y = X[m], Y[m]
    return X.astype(np.float32), Y.astype(np.float32)


def build_model(device):
    args = Args()
    args.device = device
    model = TimeDiffModel(args).to(device)
    return model, args


def run_epoch(model, loader, opt, device):
    model.train()
    tot = n = 0
    for xb, yb in loader:
        x = xb.to(device, non_blocking=True).unsqueeze(-1)          # [B,72,1]
        y = yb.to(device, non_blocking=True).unsqueeze(-1)          # [B,24,1]
        dec = torch.cat([x, y], dim=1)                              # [B,96,1]
        opt.zero_grad(set_to_none=True)
        loss = model.train_forward(x, None, dec, None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot += float(loss) * len(xb)
        n += len(xb)
    return tot / max(n, 1)


@torch.no_grad()
def val_loss(model, loader, device):
    """验证损失。**不能直接调 train_forward**：它内部会把 diffusion_worker
    设回 train()，验证时 BatchNorm 会用 batch 统计并污染 running stats。
    这里复刻它的前向，但保持 eval。"""
    model.eval()
    tot = n = 0
    for xb, yb in loader:
        x = xb.to(device).unsqueeze(-1)
        y = yb.to(device).unsqueeze(-1)
        dec = torch.cat([x, y], dim=1)                              # [B,96,1]
        lin = model.dlinear_model(
            x.permute(0, 2, 1)[:, :, -model.short_term_range:]).permute(0, 2, 1)
        mean_ = x[:, -model.norm_len:, :].mean(1, keepdim=True)
        std_ = torch.ones_like(x.std(1).unsqueeze(1))
        xi = (x - mean_) / (std_ + 1e-5)
        li = (lin - mean_) / (std_ + 1e-5)
        yf = (dec[:, -model.pred_len:, :] - mean_) / (std_ + 1e-5)
        past = torch.cat([xi.permute(0, 2, 1), li.permute(0, 2, 1)], dim=-1)
        loss = model.diffusion_worker(yf.permute(0, 2, 1), past)
        tot += float(loss) * len(xb)
        n += len(xb)
    return tot / max(n, 1)


@torch.no_grad()
def sample_windows(model, data, split, device, n_samples, batch_windows, seed=0):
    """按窗口采样：返回 [M, S, N, 24] 的原始空间样本 + 真值 + keep。"""
    N = data["N"]
    sc = data["scales"][split][:, :, None]
    X = np.clip(data[f"X_{split}"] / sc, -20, 20)
    Y = data[f"Y_{split}"]
    M = X.shape[0]
    out = np.empty((M, n_samples, N, PRE_LEN), np.float32)
    torch.manual_seed(seed)
    t0 = time.time()
    for c0 in range(0, M, batch_windows):
        c1 = min(c0 + batch_windows, M)
        xb = torch.from_numpy(X[c0:c1].reshape(-1, SEQ_L)).float().to(device)
        xb = xb.unsqueeze(-1)                                       # [B*N,72,1]
        dec = torch.cat([xb, torch.zeros_like(xb[:, :PRE_LEN])], dim=1)
        outs, _, _, _, _ = model(xb, None, dec, None, sample_times=n_samples)
        outs = outs[:, :, :, 0].cpu().numpy()                       # [B*N,S,24]
        outs = outs.reshape(c1 - c0, N, n_samples, PRE_LEN).transpose(0, 2, 1, 3)
        outs = outs * data["scales"][split][c0:c1][:, None, :, None]
        out[c0:c1] = outs
        if (c0 // batch_windows) % 5 == 0:
            print(f"  [sample] {split} {c1}/{M} 窗  {time.time() - t0:.0f}s", flush=True)
    return out, Y.astype(np.float32), data["keep"][split]


def evaluate(smp, Y, keep, smean):
    """逐 horizon 指标，口径同 kairos_prob_eval 的 csdi_pass / run_city。"""
    M, S, N, H = smp.shape
    rows = {}
    for j in range(H):
        per_win = np.full((M, N), np.nan, np.float32)
        cs_sum = 0.0
        nz_sum = nz_win = nz_cells = nz_den = 0
        pit_acc = {}
        used = 0
        for i in range(M):
            mk = keep[i]
            if not mk.any():
                continue
            truth = Y[i, :, j]
            ss = smp[i, :, :, j]                                   # [S,N]
            per_win[i, mk] = pb.crps_fair(ss[:, mk], truth[mk])
            cs_sum += float(pb.crps_fair(ss[:, mk].sum(1), truth[mk].sum()))
            m = (truth > 0) & mk
            if m.any():
                nz_sum += float(pb.crps_fair(ss[:, m], truth[m]).mean())
                nz_win += 1
                nz_cells += int(m.sum())
            nz_den += int(mk.sum())
            for k, v in pb.pit_metrics(ss[:, mk], truth[mk]).items():
                pit_acc[k] = pit_acc.get(k, 0.0) + v
            used += 1
        per_st = np.nansum(per_win, 0)
        cnt = np.maximum(np.sum(np.isfinite(per_win), 0), 1)
        rows[j + 1] = dict(
            crps_norm_sum=float(np.mean(per_st / cnt) / np.nanmean(smean)),
            crps_sum_norm=float(cs_sum / max(used, 1) / np.nansum(smean)),
            crps_nz=float(nz_sum / max(nz_win, 1) / np.nanmean(smean)),
            nz_rate=float(nz_cells / max(nz_den, 1)),
            **{k: v / max(used, 1) for k, v in pit_acc.items()},
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--device", type=int, default=2)
    ap.add_argument("--pretrain_epochs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_samples", type=int, default=24)
    ap.add_argument("--sample_batch_windows", type=int, default=32)
    ap.add_argument("--max_test_windows", type=int, default=0,
                    help=">0 只评测前 N 个测试窗（冒烟用，正式跑不要给）")
    ap.add_argument("--limit_train", type=int, default=0, help=">0 只用前 N 个训练样本（冒烟）")
    ap.add_argument("--load_ckpt", action="store_true",
                    help="跳过训练，直接读 {out}/{city}_timediff.pth 做采样评测")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--out", default=f"{BASE}/result/timediff_clean")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)

    data = build_data(args.city)
    Xtr, Ytr = flat_xy(data, "train", clean=True)
    Xva, Yva = flat_xy(data, "valid", clean=True)
    if args.limit_train > 0:
        Xtr, Ytr = Xtr[:args.limit_train], Ytr[:args.limit_train]
    print(f"[{args.city}] 训练样本 {len(Xtr)} / 验证样本 {len(Xva)}", flush=True)

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                           batch_size=args.batch_size, shuffle=True, drop_last=True,
                           num_workers=0, pin_memory=True)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva[:8192]), torch.from_numpy(Yva[:8192])),
                           batch_size=args.batch_size, shuffle=False, num_workers=0,
                           pin_memory=True)

    model, margs = build_model(device)
    npar = sum(p.numel() for p in model.parameters())
    print(f"[{args.city}] TimeDiff 参数量 {npar / 1e6:.2f}M  device={device}", flush=True)

    ckpt = f"{args.out}/{args.city}_timediff.pth"
    if args.load_ckpt:
        ck = torch.load(ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        best = float(ck.get("best_val", float("nan")))
        print(f"[{args.city}] 跳过训练，载入 {ckpt}（best val {best:.5f}）", flush=True)
    else:
        # ---- 1) DLinear 条件先验预训练 ----
        opt = torch.optim.Adam(model.dlinear_model.parameters(), lr=1e-3)
        for ep in range(args.pretrain_epochs):
            model.train()
            tot = n = 0
            for xb, yb in tr_loader:
                x = xb.to(device).unsqueeze(-1)
                y = yb.to(device).unsqueeze(-1)
                dec = torch.cat([x, y], dim=1)
                opt.zero_grad(set_to_none=True)
                loss = model.pretrain_forward(x, None, dec, None)
                loss.backward(); opt.step()
                tot += float(loss) * len(xb); n += len(xb)
            print(f"  [pretrain] ep{ep + 1:3d} loss {tot / n:.5f}", flush=True)

        # ---- 2) 条件扩散 ----
        opt = torch.optim.Adam(model.diffusion_worker.parameters(), lr=args.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        best, best_state, bad = 1e9, None, 0
        t0 = time.time()
        for ep in range(args.epochs):
            tr = run_epoch(model, tr_loader, opt, device)
            va = val_loss(model, va_loader, device)
            sch.step()
            mark = ""
            if va < best:
                best, bad = va, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                mark = " *"
            else:
                bad += 1
            print(f"  [diff] ep{ep + 1:3d} train {tr:.5f} valid {va:.5f} "
                  f"best {best:.5f} ({time.time() - t0:.0f}s){mark}", flush=True)
            if bad >= args.patience:
                print(f"  早停：{args.patience} 个 epoch 无改善", flush=True)
                break
        if best_state is not None:
            model.load_state_dict(best_state)

        torch.save(dict(model=model.state_dict(), mean=data["mean"], sd=data["sd"],
                        smean=data["smean"], city=args.city, n_station=data["N"],
                        pre_len=PRE_LEN, seq_len=SEQ_L, best_val=best,
                        args=vars(args), class_args=vars(margs)),
                   ckpt)
        print(f"[{args.city}] 已存 {ckpt}", flush=True)

    if args.skip_eval:
        return

    # ---- 3) 采样 + 评测 ----
    if args.max_test_windows > 0:
        for k in ("X_test", "Y_test"):
            data[k] = data[k][:args.max_test_windows]
        data["keep"]["test"] = data["keep"]["test"][:args.max_test_windows]
        data["scales"]["test"] = data["scales"]["test"][:args.max_test_windows]
    t0 = time.time()
    smp, Y, keep = sample_windows(model, data, "test", device,
                                  n_samples=args.n_samples,
                                  batch_windows=args.sample_batch_windows)
    print(f"[{args.city}] 采样完成 {smp.shape}  {time.time() - t0:.0f}s", flush=True)
    rows = evaluate(smp, Y, keep, data["smean"])
    meta = dict(city=args.city, n_station=data["N"], n_test=len(Y),
                n_samples=args.n_samples, best_val=best,
                rows={str(k): v for k, v in rows.items()}, args=vars(args))
    with open(f"{args.out}/{args.city}_timediff_metrics.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    H = PRE_LEN
    r = rows[H]
    print(f"\n[{args.city}] TimeDiff H={H}  crpsΣ {r['crps_norm_sum']:.4f}  "
          f"crpsΣsum(独立) {r['crps_sum_norm']:.4f}  "
          f"crps_nz {r['crps_nz']:.4f}  nz_rate {r['nz_rate']:.3f}  "
          f"PICP50 {r.get('picp50', float('nan')):.3f}  "
          f"MPIW50 {r.get('mpiw50', float('nan')):.3f}", flush=True)
    print(f"[{args.city}] 指标已存 {args.out}/{args.city}_timediff_metrics.json", flush=True)


if __name__ == "__main__":
    main()
