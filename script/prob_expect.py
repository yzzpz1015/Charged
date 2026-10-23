# -*- coding: utf-8 -*-
"""便宜探针：hurdle 的 expect 点预测 vs 主干条件均值。

不训练。复用 ``prob_hurdle.py`` 的门与正值残差池，只在目标 horizon 上比较：

  mean   = 主干点预测 ``P``（条件均值）
  expect = p_on * (P + 逐站正值残差均值)     # 期望融合，不硬判
  hard   = (p_on > 0.5) * (P + 逐站正值残差均值)
  raw    = p_on * P                          # 不做逐站偏差修正

指标：全部/真零/真正/峰格上的 MAE 与 bias。峰格用训练期逐站正值 p95。

用法::

    python script/prob_expect.py --cities MEL --horizon 24 --device 0
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "script"))
sys.path.insert(0, BASE)

import prob_hurdle as ph                                            # noqa: E402
import kairos_prob_eval as ke                                      # noqa: E402
from api.dataset.common import SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST  # noqa: E402


def gate_matrix(P, hod_j, on72, on24, train_on):
    M, N = P.shape
    X = np.stack([
        np.log1p(np.maximum(P, 0)),
        (P > 0).astype(np.float32),
        on72,
        on24,
        np.broadcast_to(np.sin(2 * np.pi * hod_j / 24)[:, None], (M, N)),
        np.broadcast_to(np.cos(2 * np.pi * hod_j / 24)[:, None], (M, N)),
        np.broadcast_to(train_on[None, :], (M, N)),
    ], axis=-1).astype(np.float32)
    return X.reshape(-1, X.shape[-1])


def station_p95(Y, keep):
    N = Y.shape[1]
    thr = np.full(N, np.nan, np.float32)
    for j in range(N):
        yj = Y[:, j, :]
        v = yj[keep[:, j][:, None] & (yj > 0)]
        if v.size >= 100:
            thr[j] = np.quantile(v, 0.95)
    return thr


def metrics(yhat, y, m, pk):
    out = []
    for mm in (m, m & (y == 0), m & (y > 0), pk):
        if not mm.any():
            out.extend([np.nan, np.nan])
            continue
        e = yhat[mm] - y[mm]
        out.extend([np.mean(np.abs(e)), np.mean(e)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["MEL"])
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out_root", default=os.path.join(BASE, "result", "kairos_l1_clean"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    j = args.horizon - 1
    rng = np.random.default_rng(args.seed)

    for city in args.cities:
        ds, raw, _ = ke.load_city(city)
        model = ke.build_kairos(city, "plain", args.out_root, dev, 99)
        if model is None:
            print(f"{city}: 缺 plain 权重，跳过", flush=True)
            continue
        S = {}
        for nm, sid in (("tr", SPLIT_TRAIN), ("va", SPLIT_VALID), ("te", SPLIT_TEST)):
            st, kp, Y, hod, hist = ph.split_data(ds, raw, sid)
            P = ke.point_pass(model, raw, st, [j], dev)[j]
            S[nm] = dict(st=st, keep=kp, Y=Y, P=P, hod=hod, hist=hist)
        del model
        torch.cuda.empty_cache()

        tr_on = np.array([S["tr"]["Y"][S["tr"]["keep"][:, k], k, :].mean()
                          if S["tr"]["keep"][:, k].any() else 0.0
                          for k in range(raw.shape[1])], np.float32)
        mats = {}
        for nm in ("tr", "va", "te"):
            d = S[nm]
            on72, on24 = ph._activity(d["hist"])
            X = gate_matrix(d["P"], d["hod"][:, j], on72, on24, tr_on)
            m = d["keep"].reshape(-1)
            mats[nm] = (X[m], d["Y"][:, :, j].reshape(-1)[m] > 0)
        Xtr, ytr = mats["tr"]
        Xva, yva = mats["va"]
        pick = rng.choice(len(ytr), min(ph.FIT_CELLS, len(ytr)), replace=False)
        sc, lr, cal = ph.fit_gate(Xtr[pick], ytr[pick], Xva, yva, rng)

        R = S["tr"]["Y"][:, :, j] - S["tr"]["P"]
        vpos = S["tr"]["keep"] & (S["tr"]["Y"][:, :, j] > 0)
        glob = R[vpos]
        glob = glob[rng.choice(len(glob), min(ph.CAP_GLOBAL, len(glob)), replace=False)]
        rmean = np.full(raw.shape[1], glob.mean(), np.float32)
        for k in range(raw.shape[1]):
            v = R[vpos[:, k], k]
            if v.size >= 100:
                rmean[k] = v.mean()

        M, N = S["te"]["P"].shape
        p_on = np.full(M * N, np.nan, np.float32)
        p_on[S["te"]["keep"].reshape(-1)] = ph.gate_prob(sc, lr, cal, mats["te"][0])
        p_on = p_on.reshape(M, N)
        P = S["te"]["P"]
        Y = S["te"]["Y"][:, :, j]
        m = S["te"]["keep"]
        thr = station_p95(S["tr"]["Y"], S["tr"]["keep"])
        ok = np.isfinite(thr)
        pk = np.zeros_like(Y, bool)
        pk[:, ok] = Y[:, ok] >= thr[ok][None, :]
        pk &= m & (Y > 0)

        mu = np.maximum(0.0, P + rmean[None, :])
        yhats = {
            "mean": P,
            "expect": p_on * mu,
            "hard": (p_on > 0.5) * mu,
            "raw": p_on * np.maximum(P, 0),
        }
        print(f"\n===== {city} H={args.horizon} 测试格 {m.sum()} 峰格 {pk.sum()} "
              f"({100 * pk.sum() / max(1, m.sum()):.1f}%) =====", flush=True)
        print(f"门：测试 AUC {roc_auc_score((Y[m] > 0).astype(int), p_on[m]):.4f}  "
              f"预测正例率 {np.mean(p_on[m] > 0.5):.3f}  真值正例率 "
              f"{np.mean(Y[m] > 0):.3f}  平均 p_on {p_on[m].mean():.3f}", flush=True)
        print(f"{'口径':<8}{'MAE全部':>10}{'bias':>9}{'MAE零':>9}{'bias':>9}"
              f"{'MAE正':>9}{'bias':>9}{'MAE峰':>9}{'bias':>9}")
        for nm, yh in yhats.items():
            r = metrics(yh, Y, m, pk)
            print(f"{nm:<8}" + "".join(f"{v:>9.3f}" for v in r), flush=True)
        print("（MAE零=真零格；MAE正=真非零格；峰格=训练期逐站正值 p95 以上）",
              flush=True)


if __name__ == "__main__":
    main()
