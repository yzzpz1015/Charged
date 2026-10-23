# -*- coding: utf-8 -*-
"""评测 PatchTST + 分位头（消融），口径与做法③ 的 quant_pass 完全一致。"""
import argparse
import json
import os
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import kairos_prob_eval as ke                                        # noqa: E402
import prob_baseline as pb                                           # noqa: E402
from api.dataset.common import (EVDataset, SPLIT_TRAIN, SPLIT_TEST)   # noqa: E402
from api.model.patchtst_quant import PatchTSTQuant                   # noqa: E402
from api.utils import straight_window_keep                           # noqa: E402

SEQ_L, PRE_LEN, FOLD = 72, 24, 6


@torch.no_grad()
def quant_pass(model, raw, starts, qs, device, keep, U=200, batch_rows=512):
    N = raw.shape[1]
    M = len(starts)
    Q = len(qs)
    off = np.arange(SEQ_L)[None, :]
    rng = np.random.default_rng(0)
    u_ind = ke.strat_u(rng, U, N, shared=False)
    rows = {j: dict(per_win=np.full((M, N), np.nan, np.float32),
                    cs_sum=0.0, nz_sum=0.0, nz_win=0, nz_cells=0, nz_den=0,
                    picp={}) for j in range(PRE_LEN)}
    w = max(1, batch_rows // N)
    for c0 in range(0, M, w):
        c1 = min(c0 + w, M)
        X = raw[starts[c0:c1, None] + off]                     # [w,72,N]
        X = X.transpose(0, 2, 1).reshape(-1, SEQ_L).astype(np.float32)
        xb = torch.from_numpy(X).to(device)
        yn, (loc, scale) = model(xb)
        pred = model.denorm(yn, loc, scale).cpu().numpy()       # [w*N,24,Q]
        pred = pred.reshape(c1 - c0, N, PRE_LEN, Q)
        pred = np.sort(pred, axis=-1)                           # 分位排序：免费增益
        for m in range(c1 - c0):
            mk = keep[c0 + m]
            if not mk.any():
                continue
            for j in range(PRE_LEN):
                truth = raw[starts[c0 + m] + SEQ_L + j]
                smp = ke.quant_to_samples_ind(pred[m, :, j], qs, u_ind)  # [U,N]
                rows[j]["per_win"][c0 + m, mk] = pb.crps_fair(smp[:, mk], truth[mk])
                rows[j]["cs_sum"] += float(pb.crps_fair(smp[:, mk].sum(1), truth[mk].sum()))
                nz = (truth > 0) & mk
                if nz.any():
                    rows[j]["nz_sum"] += float(pb.crps_fair(smp[:, nz], truth[nz]).mean())
                    rows[j]["nz_win"] += 1
                    rows[j]["nz_cells"] += int(nz.sum())
                rows[j]["nz_den"] += int(mk.sum())
                for k, v in pb.pit_metrics(smp[:, mk], truth[mk]).items():
                    rows[j]["picp"][k] = rows[j]["picp"].get(k, 0.0) + v
                rows[j]["used"] = rows[j].get("used", 0) + 1
    out = {}
    for j, d in rows.items():
        per_st = np.nansum(d["per_win"], 0)
        cnt = np.maximum(np.sum(np.isfinite(d["per_win"]), 0), 1)
        used = max(d.get("used", 0), 1)
        out[j + 1] = dict(
            crps_norm_sum=float(np.mean(per_st / cnt) / np.nanmean(smean_global)),
            crps_sum_norm=float(d["cs_sum"] / used / np.nansum(smean_global)),
            crps_nz=float(d["nz_sum"] / max(d["nz_win"], 1) / np.nanmean(smean_global)),
            nz_rate=float(d["nz_cells"] / max(d["nz_den"], 1)),
            **{k: v / used for k, v in d["picp"].items()},
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--device", type=int, default=2)
    ap.add_argument("--out", default=f"{BASE}/result/patchtst_quant_clean")
    args = ap.parse_args()

    global smean_global
    ds = EVDataset(feature="volume", auxiliary="None",
                   data_path=f"{BASE}/data/{args.city}_remove_const/")
    ds.split_cross_validation(fold=FOLD, total_fold=FOLD, train_ratio=0.8, valid_ratio=0.1)
    raw = ds.feat
    smean_global = np.nanmean(raw[ds.split_of_row == 0], axis=0)
    starts = ds.window_starts(SPLIT_TEST, SEQ_L, PRE_LEN)
    keep = straight_window_keep(raw, starts, SEQ_L, PRE_LEN)
    qs = np.load(f"{args.out}/{args.city}/qlevels_patch_quant_pre24_fold6.npy")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = PatchTSTQuant(seq_len=SEQ_L, pre_len=PRE_LEN, quantiles=list(qs)).to(device)
    ck = torch.load(f"{args.out}/{args.city}/patch_quant_pre24_fold6.pth",
                    map_location=device)
    model.load_state_dict(ck)
    model.eval()
    print(f"[{args.city}] 站 {raw.shape[1]} 测试窗 {len(starts)}  Q={len(qs)}", flush=True)

    rows = quant_pass(model, raw, starts, qs, device, keep)
    for H in (24,):
        r = rows[H]
        print(f"[{args.city}] PatchTST+quant H={H}  crpsΣ {r['crps_norm_sum']:.4f}  "
              f"crpsΣsum {r['crps_sum_norm']:.4f}  crps_nz {r['crps_nz']:.4f}  "
              f"nz_rate {r['nz_rate']:.3f}  PICP50 {r.get('picp50', np.nan):.3f}  "
              f"MPIW50 {r.get('mpiw50', np.nan):.3f}", flush=True)
    with open(f"{args.out}/{args.city}/patch_quant_metrics_full.json", "w") as f:
        json.dump({str(k): v for k, v in rows.items()}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
