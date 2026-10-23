# -*- coding: utf-8 -*-
"""两段模型：零门 + 只在正值格训练的幅度头 + 正值残差池，算 CRPS。

与 ``prob_hurdle.py`` 的区别：幅度不再用「主干预测 + 残差均值」，而是
**只在训练期 y>0 的格子上训一个 GBM**；残差池取自**验证期**（out-of-sample），
避免训练残差过窄。最终仍按 U=200 采样算 crpsΣ / 条件 CRPS。

用法::

    python script/hurdle_pos.py --cities MEL LOA --horizon 24 --device 1
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import prob_baseline as pb                                            # noqa: E402
import prob_hurdle as ph                                              # noqa: E402
import kairos_prob_eval as ke                                         # noqa: E402
from api.dataset.common import SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST   # noqa: E402

U = 200
CAP_ST = 2000
CAP_G = 10000
FIT_CELLS = 300_000
POS_CELLS = 300_000
PEAK_MIN = 100


def gate_mat(P, hod_j, on72, on24, train_on):
    M, N = P.shape
    X = np.stack([
        np.log1p(np.maximum(P, 0)),
        (P > 0).astype(np.float32),
        on72, on24,
        np.broadcast_to(np.sin(2 * np.pi * hod_j / 24)[:, None], (M, N)),
        np.broadcast_to(np.cos(2 * np.pi * hod_j / 24)[:, None], (M, N)),
        np.broadcast_to(train_on[None, :], (M, N)),
    ], axis=-1).astype(np.float32)
    return X.reshape(-1, X.shape[-1])


def mag_mat(raw, st, hist, P, hod_j, pos_mean, train_on, thr):
    """[M,N,F] 幅度头特征（全部因果，除 P 来自主干外）。"""
    M, N = P.shape
    on72, on24 = ph._activity(hist)
    last = raw[st + ke.SEQ_L - 1]
    m24 = raw[st[:, None] + np.arange(ke.SEQ_L - 24, ke.SEQ_L)[None, :]].mean(1)
    m72 = raw[st[:, None] + np.arange(ke.SEQ_L)[None, :]].mean(1)
    sn = np.sin(2 * np.pi * hod_j / 24)[:, None]
    cs = np.cos(2 * np.pi * hod_j / 24)[:, None]
    feats = [
        np.log1p(np.maximum(P, 0)), (P > 0).astype(np.float32),
        on72, on24,
        np.log1p(np.maximum(last, 0)), np.log1p(np.maximum(m24, 0)),
        np.log1p(np.maximum(m72, 0)),
        np.broadcast_to(sn, (M, N)), np.broadcast_to(cs, (M, N)),
        np.broadcast_to(pos_mean[None, :], (M, N)),
        np.broadcast_to(train_on[None, :], (M, N)),
        np.broadcast_to(thr[None, :], (M, N)),
    ]
    X = np.stack(feats, -1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def station_stats(Y, keep, N):
    """逐站：正值均值、正值 p95、活跃率（训练期，所有 horizon）。"""
    pm = np.zeros(N, np.float32)
    p95 = np.zeros(N, np.float32)
    on = np.zeros(N, np.float32)
    for s in range(N):
        v = Y[:, s, :][keep[:, s][:, None] & (Y[:, s, :] > 0)]
        hh = Y[:, s, :][keep[:, s]]
        on[s] = (hh > 0).mean() if hh.size else 0.0
        if v.size:
            pm[s] = v.mean()
            if v.size >= PEAK_MIN:
                p95[s] = np.quantile(v, 0.95)
    p95 = np.where(p95 > 0, p95, pm)
    return pm, p95, on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["MEL"])
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--out_root", default=os.path.join(BASE, "result", "kairos_l1_clean"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    j = args.horizon - 1
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    for city in args.cities:
        print(f"\n===== {city} H={args.horizon} =====", flush=True)
        ds, raw, _ = ke.load_city(city)
        N = raw.shape[1]
        model = ke.build_kairos(city, "plain", args.out_root, dev, 99)
        if model is None:
            print("  缺 plain 权重，跳过", flush=True); continue
        S = {}
        for nm, sid in (("tr", SPLIT_TRAIN), ("va", SPLIT_VALID), ("te", SPLIT_TEST)):
            st, kp, Y, hod, hist = ph.split_data(ds, raw, sid)
            P = ke.point_pass(model, raw, st, [j], dev)[j]
            S[nm] = dict(st=st, keep=kp, Y=Y, P=P, hod=hod, hist=hist)
            print(f"  {nm}: {len(st)} 窗 × {N} 站，干净 {100 * kp.mean():.1f}%", flush=True)
        del model
        torch.cuda.empty_cache()

        pm, p95, tr_on = station_stats(S["tr"]["Y"], S["tr"]["keep"], N)
        # ---- 门 ----
        def gm(nm):
            d = S[nm]
            on72, on24 = ph._activity(d["hist"])
            return gate_mat(d["P"], d["hod"][:, j], on72, on24, tr_on), \
                (d["Y"][:, :, j] > 0).reshape(-1), d["keep"].reshape(-1)

        Xtr, ytr, ktr = gm("tr")
        Xva, yva, kva = gm("va")
        pitr = rng.choice(np.flatnonzero(ktr), min(FIT_CELLS, ktr.sum()), replace=False)
        piva = rng.choice(np.flatnonzero(kva), min(FIT_CELLS, kva.sum()), replace=False)
        sc, lr, cal = ph.fit_gate(Xtr[pitr], ytr[pitr], Xva[piva], yva[piva], rng)

        # ---- 幅度头：只用训练期正值格 ----
        Mtr = mag_mat(raw, S["tr"]["st"], S["tr"]["hist"], S["tr"]["P"],
                      S["tr"]["hod"][:, j], pm, tr_on, p95)
        Mva = mag_mat(raw, S["va"]["st"], S["va"]["hist"], S["va"]["P"],
                      S["va"]["hod"][:, j], pm, tr_on, p95)
        Mte = mag_mat(raw, S["te"]["st"], S["te"]["hist"], S["te"]["P"],
                      S["te"]["hod"][:, j], pm, tr_on, p95)
        ytr_j = S["tr"]["Y"][:, :, j]
        yva_j = S["va"]["Y"][:, :, j]
        mtr = S["tr"]["keep"] & (ytr_j > 0)
        mtr_flat = mtr.reshape(-1)
        Xtr_p = Mtr.reshape(-1, Mtr.shape[-1])[mtr_flat]
        ytr_p = ytr_j.reshape(-1)[mtr_flat]
        if len(ytr_p) > POS_CELLS:
            i = rng.choice(len(ytr_p), POS_CELLS, replace=False)
            Xtr_p, ytr_p = Xtr_p[i], ytr_p[i]
        mag = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1,
                                           max_leaf_nodes=31).fit(Xtr_p, ytr_p)
        mu_tr = np.maximum(0.0, mag.predict(Mtr.reshape(-1, Mtr.shape[-1]))
                           .reshape(Mtr.shape[:2]))
        mu_va = np.maximum(0.0, mag.predict(Mva.reshape(-1, Mva.shape[-1]))
                           .reshape(Mva.shape[:2]))
        mu_te = np.maximum(0.0, mag.predict(Mte.reshape(-1, Mte.shape[-1]))
                           .reshape(Mte.shape[:2]))

        # ---- 验证期正值残差池（out-of-sample）----
        Rva = yva_j - mu_va
        mpos = S["va"]["keep"] & (yva_j > 0)
        glob = Rva[mpos]
        if len(glob) > CAP_G:
            glob = rng.choice(glob, CAP_G, replace=False)
        pools = []
        for s in range(N):
            v = Rva[mpos[:, s], s]
            if v.size >= 200:
                pools.append(v if v.size <= CAP_ST else rng.choice(v, CAP_ST, replace=False))
            else:
                pools.append(glob if len(glob) else np.zeros(1, np.float32))
        print(f"  门/幅度已拟合；训练正格 {int(mtr.sum())}，验证残差池中位 "
              f"{int(np.median([len(p) for p in pools]))}", flush=True)

        # ---- 测试期采样 ----
        on72, on24 = ph._activity(S["te"]["hist"])
        p_on = ph.gate_prob(sc, lr, cal,
                            gate_mat(S["te"]["P"], S["te"]["hod"][:, j], on72, on24, tr_on))
        p_on = p_on.reshape(len(S["te"]["st"]), N)
        Yte = S["te"]["Y"][:, :, j]
        keep = S["te"]["keep"]
        M = len(keep)
        per_st = np.zeros(N)
        cnt = np.zeros(N)
        nz_sum = 0.0
        nz_n = 0
        point = p_on * np.maximum(mu_te, 0)
        pk = keep & (Yte > p95[None, :])
        for i in range(M):
            mk = keep[i]
            if not mk.any():
                continue
            y = Yte[i]
            smp = np.zeros((U, N), np.float32)
            on = rng.random((U, N)) < p_on[i][None, :]
            for s in range(N):
                rr = pools[s][rng.integers(0, len(pools[s]), U)]
                smp[:, s] = np.maximum(0.0, mu_te[i, s] + rr)
            smp *= on
            per_st[mk] += pb.crps_fair(smp[:, mk], y[mk])
            cnt[mk] += 1
            nz = mk & (y > 0)
            if nz.any():
                nz_sum += float(pb.crps_fair(smp[:, nz], y[nz]).sum())
                nz_n += int(nz.sum())
        smean = np.nanmean(raw[ds.split_of_row == 0], axis=0)
        smean = np.maximum(smean, 1e-6)
        crps = np.mean(per_st / np.maximum(cnt, 1)) / np.mean(smean)
        cond = (nz_sum / max(nz_n, 1)) / np.mean(smean)
        m = keep
        print(f"  测试：crpsΣ {crps:.4f}  条件CRPS {cond:.4f}  "
              f"门AUC {roc_auc_score((Yte[m] > 0).astype(int), p_on[m]):.3f}", flush=True)
        for nm, mm in (("全部", m), ("真零", m & (Yte == 0)),
                       ("真正", m & (Yte > 0)), ("峰格", pk)):
            if mm.any():
                print(f"    {nm:<4} n={int(mm.sum()):>7d}  "
                      f"MAE(p·μ)={np.mean(np.abs(point[mm] - Yte[mm])):.3f}  "
                      f"bias={np.mean(point[mm] - Yte[mm]):.3f}", flush=True)


if __name__ == "__main__":
    main()
