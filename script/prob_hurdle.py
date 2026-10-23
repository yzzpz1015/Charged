# -*- coding: utf-8 -*-
"""Hurdle 基线：零/非零门 + 正值幅度残差自助（可复现，不训练新网络）。

动机（见诊断结论）：
  * 0 是真实状态，不是缺失；把零格当连续值回归会让"多零少峰"窗口整体高估 2~2.6 倍；
  * 峰值误差是上尾压扁，不是统一的乘性偏差，所以不能用"均值比例放大"修；
  * ④/⑤ 的整块扩散把两件事混在一个采样器里，门一旦误判就同时污染零格和峰格。

做法：
  1) 门（P(y>0)）：逐格逻辑回归，特征只用**因果可得**的量 —— 点预测、
     预测原点前 72h/24h 的活跃率、目标小时、该站训练期活跃率；
  2) 幅度：只在训练期 y>0 的干净格子上取残差 r=y-ŷ，按站建池（不足 200 条
     则退回全城池）；
  3) 采样：先抽 on/off，再对 on 的格子抽残差、y=max(0, ŷ+r)，off 恒 0；
  4) 指标：与 kairos_prob_eval 同口径（crpsΣ、crpsΣsum、条件 CRPS、PICP/MPIW），
     站集与 clean 掩码完全一致。

用法：
    python script/prob_hurdle.py --cities LOA --device 5
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'script'))
sys.path.insert(0, BASE)

import prob_baseline as pb                                            # noqa: E402
from kairos_prob_eval import (load_city, clean_keep, point_pass,       # noqa: E402
                              build_kairos, SEQ_L)
from api.dataset.common import SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST   # noqa: E402

U = 200                 # 每格采样条数（与 quant_pass 同量级）
CAP_STATION = 2000      # 每站残差池上限
CAP_GLOBAL = 10000      # 全城兜底池上限
FIT_CELLS = 300_000     # 门/校准用的训练格子数（随机抽样）


def split_data(ds, raw, split_id):
    starts = ds.window_starts(split_id, SEQ_L, SEQ_L)
    keep = clean_keep(raw, starts)
    off = np.arange(SEQ_L)[None, :]
    Y = np.transpose(raw[starts[:, None] + SEQ_L + off], (0, 2, 1)).astype(np.float32)
    hod = np.stack([ds.time[starts + SEQ_L + h].hour.values
                    for h in range(SEQ_L)], axis=1).astype(np.float32)
    hist = (raw[starts[:, None] + np.arange(SEQ_L)[None, :]] > 0)
    return starts, keep, Y, hod, hist


def _activity(hist):
    return hist.mean(1).astype(np.float32), hist[:, -24:, :].mean(1).astype(np.float32)


def features_at(pred, hod, on72, on24, train_on, flat_idx):
    """只对指定格子算特征（避免大站城一次展开 [M,N,H,7] 爆内存）。"""
    M, N, H = pred.shape
    m, n, h = np.unravel_index(flat_idx, (M, N, H))
    p = pred[m, n, h]
    return np.stack([
        np.log1p(np.maximum(p, 0)),
        (p > 0).astype(np.float32),
        on72[m, n], on24[m, n],
        np.sin(2 * np.pi * hod[m, h] / 24).astype(np.float32),
        np.cos(2 * np.pi * hod[m, h] / 24).astype(np.float32),
        train_on[n],
    ], axis=1)


def gate_slice(sc, lr, cal, pred_ij, on72_i, on24_i, hod_ij, train_on):
    """一个窗口 × 一个 horizon 的 P(y>0)，[N]。"""
    N = pred_ij.shape[0]
    x = np.stack([
        np.log1p(np.maximum(pred_ij, 0)),
        (pred_ij > 0).astype(np.float32),
        on72_i, on24_i,
        np.full(N, np.sin(2 * np.pi * hod_ij / 24), np.float32),
        np.full(N, np.cos(2 * np.pi * hod_ij / 24), np.float32),
        train_on,
    ], axis=1)
    z = lr.decision_function(sc.transform(x)).reshape(-1, 1)
    return cal.predict_proba(z)[:, 1]


def fit_gate(Xtr, ytr, Xva, yva, rng):
    """训练逻辑回归 + 在验证集上做 1 维 Platt 校准（不看测试期）。"""
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=1.0, max_iter=200, solver='lbfgs')
    lr.fit(sc.transform(Xtr), ytr)
    zv = lr.decision_function(sc.transform(Xva)).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, max_iter=200).fit(zv, yva)
    return sc, lr, cal


def gate_prob(sc, lr, cal, X):
    z = lr.decision_function(sc.transform(X)).reshape(-1, 1)
    return cal.predict_proba(z)[:, 1]


def residual_pools(r, y, keep, N, rng):
    """逐站正值残差池；样本不足的站退回全城池。"""
    pools, glob = [], []
    for j in range(N):
        v = r[keep[:, j], j, :]
        v = v[v > 0]
        if v.size > CAP_GLOBAL:
            v = rng.choice(v, CAP_GLOBAL, replace=False)
        glob.append(v)
        pools.append(v)
    glob = np.concatenate(glob) if glob else np.zeros(1, np.float32)
    if glob.size > CAP_GLOBAL:
        glob = rng.choice(glob, CAP_GLOBAL, replace=False)
    out = []
    for j in range(N):
        p = pools[j]
        if p.size >= 200:
            out.append(p if p.size <= CAP_STATION
                       else rng.choice(p, CAP_STATION, replace=False))
        else:
            out.append(glob)
    return out


def eval_hurdle(pred, Y, keep, hod, hist, train_on, sc, lr, cal, pools, rng):
    """逐 horizon 采样并算指标，返回 {H: dict}。"""
    M, N, H = pred.shape
    on72, on24 = _activity(hist)
    res = {}
    js = [23, 47, 71]
    for j in js:
        per_st = np.zeros(N)
        cnt = np.zeros(N)
        cs_sum = 0.0
        nz_sum, nz_cells, nz_den, nz_win = 0.0, 0, 0, 0
        pit_acc = {}
        used = 0
        for i in range(M):
            mk = keep[i]
            if not mk.any():
                continue
            y = Y[i, :, j]
            p = pred[i, :, j]
            p_on = gate_slice(sc, lr, cal, p, on72[i], on24[i], hod[i, j], train_on)
            smp = np.zeros((U, N), np.float32)
            on = rng.random((U, N)) < p_on[None, :]
            for st in range(N):
                rr = pools[st][rng.integers(0, len(pools[st]), U)]
                smp[:, st] = np.maximum(0.0, p[st] + rr)
            smp *= on
            per_st[mk] += pb.crps_fair(smp[:, mk], y[mk])
            cnt[mk] += 1
            cs_sum += float(pb.crps_fair(smp[:, mk].sum(1), y[mk].sum()))
            used += 1
            nz_den += int(mk.sum())
            nz = y[mk] > 0
            if nz.any():
                v, n_nz = _nz_crps(smp[:, mk], y[mk])
                if n_nz:
                    nz_sum += v
                    nz_cells += n_nz
                    nz_win += 1
            pm = pb.pit_metrics(smp[:, mk], y[mk])
            for k, v in pm.items():
                pit_acc[k] = pit_acc.get(k, 0.0) + v
        res[j + 1] = dict(
            per_st=per_st, cnt=cnt, cs_sum=cs_sum / max(used, 1),
            nz_sum=nz_sum, nz_win=max(nz_win, 1), nz_cells=nz_cells,
            nz_den=nz_den, used=used,
            pit={k: v / max(used, 1) for k, v in pit_acc.items()})
    return res


def _nz_crps(smp, obs):
    """非零真值格上的条件 CRPS，与 kairos_prob_eval.nz_crps 同口径。"""
    m = obs > 0
    if not m.any():
        return 0.0, 0
    return float(pb.crps_fair(smp[:, m], obs[m]).sum()), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cities', nargs='+', default=['LOA'])
    ap.add_argument('--device', type=int, default=5)
    ap.add_argument('--out_root', default=os.path.join(BASE, 'result', 'kairos_l1_clean'))
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    for city in args.cities:
        print(f'\n===== {city} =====', flush=True)
        ds, raw, _ = load_city(city)
        N = raw.shape[1]
        model = build_kairos(city, 'plain', args.out_root, dev, 99)
        if model is None:
            print('  跳过：没有 plain 权重'); continue

        S = {}
        for nm, sid in (('tr', SPLIT_TRAIN), ('va', SPLIT_VALID), ('te', SPLIT_TEST)):
            st, kp, Y, hod, hist = split_data(ds, raw, sid)
            P = np.stack([point_pass(model, raw, st, list(range(SEQ_L)), dev)[h]
                          for h in range(SEQ_L)], axis=2)
            S[nm] = dict(st=st, keep=kp, Y=Y, P=P, hod=hod, hist=hist)
            print(f'  {nm}: {len(st)} 窗 × {N} 站，干净 {100 * kp.mean():.1f}%', flush=True)
        del model
        torch.cuda.empty_cache()

        train_on = np.array([S['tr']['Y'][S['tr']['keep'][:, j], j, :].mean()
                             if S['tr']['keep'][:, j].any() else 0.0 for j in range(N)],
                            np.float32)
        # ---- 门：训练期拟合并用验证期校准 ----
        def fit_rows(nm):
            d = S[nm]
            m = d['keep'][:, :, None].repeat(SEQ_L, 2)
            flat = np.flatnonzero(m.reshape(-1))
            pick = rng.choice(flat, min(FIT_CELLS, len(flat)), replace=False)
            on72, on24 = _activity(d['hist'])
            X = features_at(d['P'], d['hod'], on72, on24, train_on, pick)
            return X, (d['Y'].reshape(-1)[pick] > 0), pick

        Xtr, ytr, _ = fit_rows('tr')
        Xva, yva, _ = fit_rows('va')
        sc, lr, cal = fit_gate(Xtr, ytr, Xva, yva, rng)
        print('  门已拟合；验证期 (校准前/后) 正例率 '
              f'{(lr.predict(sc.transform(Xva)) > 0).mean():.3f} / '
              f'{(gate_prob(sc, lr, cal, Xva) > .5).mean():.3f}，真值 {yva.mean():.3f}',
              flush=True)
        # ---- 幅度残差池：训练期 y>0 的干净格 ----
        R = S['tr']['Y'] - S['tr']['P']
        pools = residual_pools(R, S['tr']['Y'], S['tr']['keep'], N, rng)
        print('  残差池中位长度', int(np.median([len(p) for p in pools])), flush=True)
        del Xtr, Xva, R
        # ---- 测试期采样与指标 ----
        out = eval_hurdle(S['te']['P'], S['te']['Y'], S['te']['keep'], S['te']['hod'],
                          S['te']['hist'], train_on, sc, lr, cal, pools, rng)
        smean = np.nanmean(raw[ds.split_of_row == 0], axis=0)
        smean = np.maximum(smean, 1e-6)
        for H, d in out.items():
            per = d['per_st'] / np.maximum(d['cnt'], 1)
            print(f'  H={H:2d}  crpsΣ {np.mean(per) / np.mean(smean):.4f}  '
                  f'crpsΣsum {d["cs_sum"] / np.sum(smean):.4f}  '
                  f'条件CRPS {d["nz_sum"] / d["nz_win"] / np.mean(smean):.4f}  '
                  f'非零率 {d["nz_cells"] / max(d["nz_den"], 1):.3f}  '
                  f'PICP@50/80/90 {d["pit"].get("picp50", np.nan):.2f}/'
                  f'{d["pit"].get("picp80", np.nan):.2f}/{d["pit"].get("picp90", np.nan):.2f}',
                  flush=True)


if __name__ == '__main__':
    main()
