# -*- coding: utf-8 -*-
"""#14 vanilla CSDI：**官方原版**（ermongroup/CSDI）的预测式扩散。

与 #3 的区别只在模型：这里直接调用 ``external/CSDI`` 的 ``CSDI_Forecasting``、
``diff_CSDI`` 和官方损失/采样过程，不加 on/off 门控、不做 Hilbert 空间序。
只替换数据适配层：

* 每个窗口 = ``[history 72 | future 72]`` 共 144 步，``gt_mask`` 前 72 为 1、
  后 72 为 0（官方 forecasting 的 test pattern）；
* 归一化 = **训练期逐站 mean/std**（对应官方 ``meanstd.pkl``），std 下限夹住；
* 训练沿用官方 ``num_sample_features=64`` 的随机特征抽样；
* 采样用官方 `impute` 的同一套公式，仅把样本维并进 batch 以节省时间
  （每个样本独立加噪，数学上等价）。

指标不走官方 ``utils.evaluate``（它算的是官方 quantile CRPS / MSE），而是把
采样得到的未来 72 步反归一化后喂给 ``common.StreamingMetrics``。

用法::

    python3 script/baselines/csdi_official.py --city MEL --device 6 --smoke
    python3 script/baselines/csdi_official.py --city LOA --device 6
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSDI_DIR = os.path.join(BASE, "external", "CSDI")
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CSDI_DIR)

import common as C  # noqa: E402
from main_model import CSDI_Forecasting  # noqa: E402

METHOD = "csdi_official"


def load_config(args, N):
    with open(os.path.join(CSDI_DIR, "config", "base_forecasting.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["epochs"] = args.epochs
    cfg["train"]["batch_size"] = args.batch_windows
    cfg["train"]["lr"] = args.lr
    cfg["train"]["itr_per_epoch"] = 10 ** 8
    cfg["model"]["num_sample_features"] = min(64, N)
    cfg["model"]["is_unconditional"] = 0
    return cfg


class ChargedWindows(torch.utils.data.Dataset):
    """窗口 → 官方 Forecasting_Dataset 同名字段。

    ``keep`` [M,N] 为 clean mask：置 0 的 (窗,站) **只在未来目标块**不参与损失，
    历史仍作条件。这一点不能含糊：官方 ``calc_loss`` 里 ``target_mask =
    observed_mask - cond_mask``，而验证时 ``cond_mask = gt_mask``，若把历史也
    置 0，历史会变成 -1 的"目标"，sum 变小甚至为负，valid loss 会爆到 1e4
    （JHB/SPO 实测）。
    """

    def __init__(self, raw, starts, mean, std, keep=None):
        M, N = len(starts), raw.shape[1]
        L = C.SEQ_L + C.PRE_LEN
        off = np.arange(L)[None, :]
        data = raw[starts[:, None] + off].astype(np.float32)          # [M,L,N]
        self.data = (data - mean[None, None, :]) / std[None, None, :]
        self.gt_mask = np.ones((M, L, N), np.float32)
        self.gt_mask[:, C.SEQ_L:] = 0.0
        self.obs_mask = np.ones((M, L, N), np.float32)
        if keep is not None:
            k = np.asarray(keep, bool)
            self.obs_mask[:, C.SEQ_L:] *= k[:, None, :].astype(np.float32)
        self.tp = np.arange(L, dtype=np.float32)
        self.fid = np.arange(N, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return dict(observed_data=self.data[i],
                    observed_mask=self.obs_mask[i],
                    gt_mask=self.gt_mask[i],
                    timepoints=self.tp,
                    hist_mask=self.obs_mask[i],
                    cut_length=0,
                    feature_id=self.fid)


def collate(items):
    out = {}
    for k in ("observed_data", "observed_mask", "gt_mask", "hist_mask"):
        out[k] = torch.from_numpy(np.stack([it[k] for it in items]))
    out["timepoints"] = torch.from_numpy(np.stack([it["timepoints"] for it in items]))
    out["feature_id"] = torch.from_numpy(np.stack([it["feature_id"] for it in items]))
    out["cut_length"] = torch.zeros(len(items), dtype=torch.long)
    return out


@torch.no_grad()
def impute_batched(model, observed_data, cond_mask, side_info, n_samples,
                   chunk, device):
    """官方 ``impute`` 的批量化版本（样本独立，数学等价）。

    observed_data [B,K,L]；cond_mask [B,K,L]；side_info [B,*,K,L]。
    返回 [B,n_samples,K,L]。
    """
    B, K, L = observed_data.shape
    steps = model.num_steps
    outs = []
    for s0 in range(0, n_samples, chunk):
        c = min(chunk, n_samples - s0)
        obs = observed_data.repeat_interleave(c, dim=0)
        cm = cond_mask.repeat_interleave(c, dim=0)
        si = side_info.repeat_interleave(c, dim=0)
        cur = torch.randn_like(obs)
        cond_obs = (cm * obs).unsqueeze(1)
        for t in range(steps - 1, -1, -1):
            noisy = ((1 - cm) * cur).unsqueeze(1)
            inp = torch.cat([cond_obs, noisy], dim=1)
            pred = model.diffmodel(inp, si, torch.tensor([t], device=device))
            coeff1 = 1 / model.alpha_hat[t] ** 0.5
            coeff2 = (1 - model.alpha_hat[t]) / (1 - model.alpha[t]) ** 0.5
            cur = coeff1 * (cur - coeff2 * pred)
            if t > 0:
                sigma = ((1 - model.alpha[t - 1]) / (1 - model.alpha[t])
                         * model.beta[t]) ** 0.5
                cur = cur + sigma * torch.randn_like(cur)
        outs.append(cur.view(B, c, K, L))
    return torch.cat(outs, dim=1)


@torch.no_grad()
def valid_loss(model, loader, device, max_batches=5):
    tot, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        tot += float(model(batch, is_train=0))
        n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="6")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--steps_per_epoch", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_windows", type=int, default=8)
    ap.add_argument("--n_sample", type=int, default=50)
    ap.add_argument("--sample_chunk", type=int, default=0,
                    help="每次并行采样的样本数；0=按站数自动")
    ap.add_argument("--eval_batch", type=int, default=0,
                    help="每次评几个窗；0=按站数自动")
    ap.add_argument("--n_eval_windows", type=int, default=256,
                    help="均匀抽取的测试窗数；0=全部。官方 50 步采样在大城市"
                         "全量跑不动，固定子集并在 JSON 记 n_test_windows")
    ap.add_argument("--limit_test", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="MEL 小样本冒烟")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--no_diag", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.steps_per_epoch = min(args.steps_per_epoch, 10)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dev = C.torch_device(args.device)

    ds, raw, starts, keep, smean = C.load_city(args.city)
    N, _ = C.check_protocol(args.city, ds, raw, starts)
    tr_rows = ds.split_of_row == 0
    mean = np.nanmean(raw[tr_rows], axis=0)
    std = np.nanstd(raw[tr_rows], axis=0)
    std = np.maximum(std, np.maximum(1e-3, 0.01 * np.abs(mean)))    # 防常数站爆分母

    cfg = load_config(args, N)
    model = CSDI_Forecasting(cfg, str(dev), N).to(dev)
    n_par = sum(p.numel() for p in model.parameters())

    tr_ds = ChargedWindows(raw, starts["tr"], mean, std, keep["tr"])
    va_ds = ChargedWindows(raw, starts["va"], mean, std, keep["va"])
    tr_loader = torch.utils.data.DataLoader(
        tr_ds, batch_size=args.batch_windows, shuffle=True,
        collate_fn=collate, drop_last=True)
    va_loader = torch.utils.data.DataLoader(
        va_ds, batch_size=args.batch_windows, shuffle=False, collate_fn=collate)

    out_m, out_d, out_c = C.out_dirs(METHOD, args.city)
    ckpt = os.path.join(out_c, f"seed{args.seed}.pth")
    print(f"[{METHOD}/{args.city}/seed{args.seed}] 站 {N} 窗 tr/va/te "
          f"{len(starts['tr'])}/{len(starts['va'])}/{len(starts['te'])} "
          f"params {n_par/1e6:.1f}M  sample_features "
          f"{cfg['model']['num_sample_features']}", flush=True)

    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    best, bad = float("inf"), 0
    if not args.eval_only:
        for ep in range(1, args.epochs + 1):
            model.train()
            run, nstep = 0.0, 0
            for batch in tr_loader:
                loss = model(batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                run += float(loss)
                nstep += 1
                if nstep >= args.steps_per_epoch:
                    break
            if ep % 5 == 0 or ep == 1:
                model.eval()
                vl = valid_loss(model, va_loader, dev)
                flag = ""
                if vl < best - 1e-5:
                    best, bad = vl, 0
                    torch.save(model.state_dict(), ckpt)
                    flag = " *"
                else:
                    bad += 5
                print(f"  ep {ep:3d}  train {run/max(nstep,1):.4f}  valid "
                      f"{vl:.4f}  best {best:.4f}  {time.time()-t0:.0f}s{flag}",
                      flush=True)
                if bad >= args.patience:
                    print(f"  early stop @ ep {ep}", flush=True)
                    break
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    model.target_dim = N                      # 官方 evaluate 前同样要重置

    te_idx = np.arange(len(starts["te"]))
    want = 8 if args.smoke else args.n_eval_windows
    if want and want < len(te_idx):
        te_idx = np.unique(np.linspace(0, len(te_idx) - 1,
                                       want).round().astype(int))
    if args.limit_test:
        te_idx = te_idx[:args.limit_test]
    te = starts["te"][te_idx]
    te_keep = keep["te"][te_idx]
    te_ds = ChargedWindows(raw, te, mean, std, te_keep)
    eval_batch = args.eval_batch or max(1, min(8, 2048 // N))
    te_loader = torch.utils.data.DataLoader(
        te_ds, batch_size=eval_batch, shuffle=False, collate_fn=collate)

    chunk = args.sample_chunk or max(1, min(32, 2048 // N))
    if args.smoke:
        chunk = min(chunk, 4)
        args.n_sample = min(args.n_sample, 10)
    n_sample = args.n_sample
    print(f"  eval: {len(te)} 窗  batch {eval_batch}  nsample {n_sample}  "
          f"chunk {chunk}", flush=True)

    met = C.make_metrics(args.city, raw, te, te_keep, smean,
                         np.arange(1, 100) / 100,
                         with_diag=not args.no_diag)

    t1 = time.time()
    for b0, batch in enumerate(te_loader):
        od, om, tp, gm, _, _, fid = model.process_data(batch)
        cond = gm
        target_mask = om * (1 - gm)
        side = model.get_side_info(tp, cond)
        samples = impute_batched(model, od, cond, side, n_sample, chunk, dev)
        # [B,S,K,L] -> 未来 72 步 -> 反归一化 -> [B,N,72,S]
        fut = samples[:, :, :, C.SEQ_L:]                       # [B,S,K,72]
        fut = fut.permute(0, 2, 3, 1).cpu().numpy()            # [B,K,72,S]
        fut = fut * std[None, :, None, None] + mean[None, :, None, None]
        met.update_samples(b0 * eval_batch, fut)
        done = b0 * eval_batch + len(od)
        if b0 % 5 == 0 or done >= len(te):
            el = time.time() - t1
            eta = el / done * (len(te) - done)
            print(f"    {done}/{len(te)} 窗  {el:.0f}s  ETA {eta/60:.0f}min",
                  flush=True)

    res = met.result()
    extra = dict(params=n_par, best_valid=best,
                 elapsed_s=round(time.time() - t0, 1), n_sample=n_sample,
                 n_test_windows=len(te), n_test_windows_total=len(starts["te"]),
                 epochs_run=ep if not args.eval_only else 0,
                 config="official ermongroup/CSDI CSDI_Forecasting")
    C.save_metrics(METHOD, args.city, args.seed, res, extra)
    if not args.no_diag:
        C.save_diag(METHOD, args.city, args.seed, met)


if __name__ == "__main__":
    main()
