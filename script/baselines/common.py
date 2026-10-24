# -*- coding: utf-8 -*-
"""Baseline 补齐的统一协议底座。

所有新方法（#7–#20）共用这里的取数、取窗、clean mask、指标聚合与产物落盘，
口径**逐字复刻**现有 ``script/kairos_prob_eval.py`` / ``script/eval_patch_quant.py``：

* 数据：``data/<CITY>_remove_const/``；停摆行丢弃 + 可用行计数 2/1/3；
  筛站只用训练期统计（在 ``EVDataset.split_cross_validation`` 内部完成）。
* 窗口：``seq_l=72, pre_len=72``；H=24/48/72 = 输出的第 23/47/71 步，
  **不按 H 重新取窗**。这样 #1/#2 的 checkpoint 不用重训。
* clean mask：``straight_window_keep(raw, starts, 72, 72)`` → [M,N] bool，
  按「站 × 窗」剔除含注入式直线段的目标块。
* crpsΣ = ``mean_station(每站平均 CRPS) / mean_station(训练期站均值)``（mean-of-means）。
* 分位→样本：独立耦合 ``strat_u(shared=False)`` + ``quant_to_samples_ind``；
  分位先 ``np.sort`` 再转换（现有结论：排序是免费增益）。
* 采样数 U=200（清单写 100，但已有 #1–#6 全部是 200，这里跟已有结果对齐）。

产物（不落全量预测）::

    result/baselines/metrics/{method}/{city}/H{h}_seed{s}.json
    result/baselines/diag/{method}/{city}/seed{s}.npz   # 16 窗 × 32 分层站 × 72 × 99 分位
    result/baselines/ckpt/{method}/{city}/seed{s}.pth

诊断子集是给零膨胀方法开发用的：看形状、画图、做残差诊断，不需要重跑模型。
"""
import json
import os
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

from api.dataset.common import (  # noqa: E402
    EVDataset, SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST,
)
from api.utils import straight_window_keep  # noqa: E402
import prob_baseline as pb  # noqa: E402
import kairos_prob_eval as ke  # noqa: E402

SEQ_L = 72
PRE_LEN = 72
FOLD = 6
TOTAL_FOLD = 6
HORIZONS = (24, 48, 72)
JS = tuple(h - 1 for h in HORIZONS)
M_SAMPLE = 200          # 与现有 #1–#6 一致

RESULT_ROOT = os.path.join(BASE, "result", "baselines")

# 站数/窗数自检表（协议门）。任何方法跑出来的数与这里不符即判无效。
EXPECT_STATIONS = {"LOA": 218, "JHB": 29, "MEL": 56, "SPO": 33, "SZH": 843}
EXPECT_TEST_WINDOWS = {"LOA": 2125, "JHB": 2125, "MEL": 1781, "SPO": 2125, "SZH": 2125}


# ------------------------------------------------------------------ 数据
def load_city(city, data_path=None):
    """加载一城并划定协议窗口。

    Returns
    -------
    ds : EVDataset
    raw : [T,N] float64，已按训练期统计剔站
    starts : {'tr','va','te'} -> [M] 全局窗口起点
    keep : {'tr','va','te'} -> [M,N] bool clean mask
    smean : [N] 训练期站均值（crpsΣ 的归一化分母）
    """
    data_path = data_path or os.path.join(BASE, "data", f"{city}_remove_const") + "/"
    ds = EVDataset(feature="volume", auxiliary="None", data_path=data_path)
    ds.split_cross_validation(fold=FOLD, total_fold=TOTAL_FOLD,
                              train_ratio=0.8, valid_ratio=0.1)
    raw = ds.feat
    starts = {k: ds.window_starts(sid, SEQ_L, PRE_LEN)
              for k, sid in (("tr", SPLIT_TRAIN), ("va", SPLIT_VALID), ("te", SPLIT_TEST))}
    keep = {k: straight_window_keep(raw, s, SEQ_L, PRE_LEN) for k, s in starts.items()}
    smean = np.nanmean(raw[ds.split_of_row == SPLIT_TRAIN], axis=0)
    return ds, raw, starts, keep, smean


def load_distance(city, ds, data_path=None):
    """distance.csv → [N,N]，**先按 ds.feat_columns reindex**。

    坑：distance.csv 的 index 是 int、columns 是 str，且站顺序与 volume.csv 不同。
    不 reindex 会静默错位（图方法只会表现成"图没用"）。
    """
    import pandas as pd
    data_path = data_path or os.path.join(BASE, "data", f"{city}_remove_const") + "/"
    d = pd.read_csv(os.path.join(data_path, "distance.csv"), index_col=0)
    d.index = d.index.astype(str)
    d.columns = d.columns.astype(str)
    d = d.reindex(index=ds.feat_columns, columns=ds.feat_columns)
    if d.isna().any().any():
        raise AssertionError(f"{city} distance.csv reindex 后仍有缺失")
    return d.values.astype(np.float32)


def check_protocol(city, ds, raw, starts):
    """站数 / 测试窗数自检，防止静默错位。"""
    n = raw.shape[1]
    m = len(starts["te"])
    e_n, e_m = EXPECT_STATIONS.get(city), EXPECT_TEST_WINDOWS.get(city)
    if e_n is not None and n != e_n:
        raise AssertionError(f"{city} 站数 {n} != 协议 {e_n}")
    if e_m is not None and m != e_m:
        raise AssertionError(f"{city} 测试窗数 {m} != 协议 {e_m}")
    return n, m


def build_windows(raw, starts, pre_len=PRE_LEN):
    """[M,72,N] 历史与 [M,pre_len,N] 目标（window-major / site-minor）。

    注意这里保留三维，与训练脚本给单站模型的 ``[M*N,72]`` 不同：
    图方法 / 整块扩散 / 零样本都需要站维，拍平在各 runner 里做。
    """
    M = len(starts)
    off_in = np.arange(SEQ_L, dtype=np.int64)[None, :]
    off_out = np.arange(pre_len, dtype=np.int64)[None, :]
    X = raw[starts[:, None] + off_in]
    Y = raw[starts[:, None] + SEQ_L + off_out]
    return X.astype(np.float32), Y.astype(np.float32)


def flatten_windows(X, Y):
    """[M,72,N] -> [M*N,72]，与 ``train_kairos_l1.build_windows`` 同排布。"""
    M, _, N = X.shape
    return (np.transpose(X, (0, 2, 1)).reshape(M * N, SEQ_L).astype(np.float32),
            np.transpose(Y, (0, 2, 1)).reshape(M * N, -1).astype(np.float32))


# ------------------------------------------------------------------ 指标
def _empty_acc():
    return dict(per_win=[], cs_sum=0.0, nz_sum=0.0, nz_win=0, nz_cells=0,
                nz_den=0, n_scored=0, picp={}, mpiw={})


class StreamingMetrics:
    """边推理边累计；一次 ``update`` 给一个 batch 的 72 步预测。

    两种输入二选一：

    * ``update_quantiles(m0, qv)``：``qv [w,N,72,Q]`` 已反归一化（raw 单位），
      内部 ``np.sort`` 后按独立耦合转 200 条样本；
    * ``update_samples(m0, smp)``：``smp [w,N,72,S]`` 原始样本，直接算。

    两种路径对 ``crpsΣ / crps_nz / PICP`` 的定义完全相同：都是对**同一条经验
    分布**（分位类 = 99 个分位点插值、采样类 = 模型给的样本）算公平 CRPS。
    采样类的 ``CRPS_sum`` 用模型自己的联合样本，不做人工耦合——这正是扩散/
    自回归相对逐站分位头的区别，不能替它换耦合。
    """

    def __init__(self, raw, starts, keep, smean, qs=None, horizons=HORIZONS,
                 U=M_SAMPLE, diag_win_ids=None, diag_station_ids=None):
        self.raw = raw
        self.starts = np.asarray(starts)
        self.keep = np.asarray(keep)
        self.smean = np.asarray(smean, np.float64)
        self.qs = None if qs is None else np.asarray(qs, np.float64)
        self.horizons = tuple(horizons)
        self.U = U
        self.u_ind = ke.strat_u(np.random.default_rng(0), U, raw.shape[1], shared=False)
        self.acc = {h: _empty_acc() for h in self.horizons}
        self.per_win = {h: np.full((len(starts), raw.shape[1]), np.nan, np.float32)
                        for h in self.horizons}
        self.diag_win_ids = diag_win_ids
        self.diag_station_ids = diag_station_ids
        self.diag = {}

    # -- 输入 ---------------------------------------------------------
    def _samples_from_quant(self, qv):
        """[N,Q] -> [U,N] 独立耦合。"""
        qv = np.sort(qv, axis=-1)                  # 免费增益，与现有评测一致
        return ke.quant_to_samples_ind(qv, self.qs, self.u_ind)

    def _update_one(self, m, pred72, samples=None):
        """pred72：分位类 [N,72,Q]；samples：采样类 [N,72,S]。"""
        mk = self.keep[m]
        if not mk.any():
            return
        for h, j in zip(self.horizons, JS):
            truth = self.raw[self.starts[m] + SEQ_L + j]
            if samples is not None:
                smp = samples[:, j, :].T                # [S,N]
            else:
                smp = self._samples_from_quant(pred72[:, j, :])
            ym = truth[mk]
            d = self.acc[h]
            self.per_win[h][m, mk] = pb.crps_fair(smp[:, mk], ym)
            d["n_scored"] += 1
            d["nz_den"] += int(mk.sum())
            d["cs_sum"] += float(pb.crps_fair(smp[:, mk].sum(1), ym.sum()))
            nz = mk & (truth > 0)
            if nz.any():
                d["nz_sum"] += float(pb.crps_fair(smp[:, nz], truth[nz]).mean())
                d["nz_win"] += 1
                d["nz_cells"] += int(nz.sum())
            for k, v in pb.pit_metrics(smp[:, mk], ym).items():
                tgt = d["picp"] if k.startswith("picp") else d["mpiw"]
                tgt[k] = tgt.get(k, 0.0) + v

    def update_quantiles(self, m0, qv):
        qv = np.asarray(qv)
        for i in range(qv.shape[0]):
            self._update_one(m0 + i, qv[i])
            if self.diag_win_ids is not None and (m0 + i) in self.diag_win_ids:
                self._capture_diag(m0 + i, qv[i])

    def update_samples(self, m0, smp):
        smp = np.asarray(smp)
        for i in range(smp.shape[0]):
            self._update_one(m0 + i, None, samples=smp[i])
            if self.diag_win_ids is not None and (m0 + i) in self.diag_win_ids:
                # 采样类没有分位张量：用经验分位数存诊断，与分位类可同图对比。
                # np.quantile 的 q 维在前，得到 [Q,N,72]，这里换回 [N,72,Q]。
                q = np.moveaxis(np.quantile(smp[i], self.qs, axis=-1), 0, -1)
                self._capture_diag(m0 + i, q)

    # -- 诊断子集 -----------------------------------------------------
    def _capture_diag(self, m, pred72):
        if self.diag_station_ids is None:
            return
        sub = pred72[self.diag_station_ids, :, :]
        # 诊断子集只用来看形状；扩散未收敛时偶发爆值，夹到 float16 可表示范围
        sub = np.clip(sub, -6e4, 6e4)
        self.diag.setdefault("quantiles", []).append(sub.astype(np.float16))
        rows = self.starts[m] + SEQ_L + np.arange(PRE_LEN)
        truth = self.raw[rows][:, self.diag_station_ids]      # [72,32]
        self.diag.setdefault("target", []).append(truth.astype(np.float32))
        self.diag.setdefault("win_ids", []).append(m)

    # -- 结果 ---------------------------------------------------------
    def result(self):
        sn = np.nanmean(self.smean)
        ss = np.nansum(self.smean)
        out = {}
        for h, j in zip(self.horizons, JS):
            d = self.acc[h]
            used = max(d["n_scored"], 1)
            per_win = self.per_win[h]
            per_st = np.nansum(per_win, 0)
            cnt = np.maximum(np.sum(np.isfinite(per_win), 0), 1)
            raw_crps = float(np.mean(per_st / cnt))
            mpiw50 = d["mpiw"].get("mpiw50", 0.0) / used
            if mpiw50 > 0 and raw_crps < mpiw50 / 100:
                raise AssertionError(
                    f"H={h} raw CRPS({raw_crps:.3g}) 比 MPIW50({mpiw50:.3g}) 小两个"
                    f"数量级以上，聚合口径有问题")
            out[f"H{h}"] = dict(
                crps_norm_sum=raw_crps / sn,
                crps_sum_norm=float(d["cs_sum"] / used / ss),
                crps_nz=float(d["nz_sum"] / max(d["nz_win"], 1) / sn),
                nz_rate=float(d["nz_cells"] / max(d["nz_den"], 1)),
                n_scored_windows=int(d["n_scored"]),
                **{k: v / used for k, v in d["picp"].items()},
                **{k: v / used for k, v in d["mpiw"].items()},
            )
        return out

    def diag_payload(self):
        if not self.diag:
            return None
        q = np.stack(self.diag["quantiles"])          # [W,32,72,Q]
        t = np.stack(self.diag["target"])             # [W,72,32]
        return dict(quantiles=np.transpose(q, (0, 2, 1, 3)).astype(np.float16),
                    target=t.astype(np.float32),
                    win_ids=np.asarray(self.diag["win_ids"], np.int64),
                    station_ids=np.asarray(self.diag_station_ids, np.int64))


def pick_diag_subset(raw, starts, keep, n_win=16, n_station=32, seed=0):
    """固定诊断子集：16 个均匀窗 × 32 个按测试期零率分层的站。

    只用于可视化和零膨胀方法开发，不参与筛选/调参——所以用测试期零率分层
    不构成协议泄漏（它不是模型输入，也不回改任何训练决策）。
    """
    M, N = int(len(starts)), int(raw.shape[1])
    wids = np.unique(np.linspace(0, M - 1, min(n_win, M)).round().astype(int))
    tgt = raw[np.asarray(starts) + SEQ_L + (PRE_LEN - 1)]  # [M,N] 最远 horizon
    zero = np.mean(tgt == 0, axis=0)
    order = np.argsort(zero)
    sidx = order[np.linspace(0, N - 1, min(n_station, N)).round().astype(int)]
    return wids, np.sort(sidx)


# ------------------------------------------------------------------ 落盘
def out_dirs(method, city):
    d = os.path.join(RESULT_ROOT, "metrics", method, city)
    g = os.path.join(RESULT_ROOT, "diag", method, city)
    c = os.path.join(RESULT_ROOT, "ckpt", method, city)
    for x in (d, g, c):
        os.makedirs(x, exist_ok=True)
    return d, g, c


def save_metrics(method, city, seed, res, extra=None):
    d, _, _ = out_dirs(method, city)
    for h, r in res.items():
        payload = dict(method=method, city=city, seed=seed, horizon=h,
                       seq_l=SEQ_L, pre_len=PRE_LEN, n_sample=M_SAMPLE,
                       stations=int(EXPECT_STATIONS.get(city, -1)), **r)
        if extra:
            payload.update(extra)
        path = os.path.join(d, f"{h}_seed{seed}.json")
        with open(path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"[{method}/{city}/seed{seed}] {h}  crpsΣ {r['crps_norm_sum']:.4f}  "
              f"crpsΣsum {r['crps_sum_norm']:.4f}  crps_nz {r['crps_nz']:.4f}  "
              f"PICP50 {r.get('picp50', float('nan')):.3f}  "
              f"PICP90 {r.get('picp90', float('nan')):.3f}", flush=True)
    return d


def save_diag(method, city, seed, metrics):
    _, g, _ = out_dirs(method, city)
    payload = metrics.diag_payload()
    if payload is None:
        return None
    path = os.path.join(g, f"seed{seed}.npz")
    np.savez_compressed(path, **payload)
    print(f"[diag] {path}  q={payload['quantiles'].shape}  "
          f"{os.path.getsize(path)/1e6:.1f} MB", flush=True)
    return path


def make_metrics(city, raw, starts, keep, smean, qs, with_diag=True):
    wids = sidx = None
    if with_diag:
        wids, sidx = pick_diag_subset(raw, starts, keep)
    return StreamingMetrics(raw, starts, keep, smean, qs,
                            diag_win_ids=set(wids.tolist()) if wids is not None else None,
                            diag_station_ids=sidx)


def torch_device(spec="0"):
    if isinstance(spec, torch.device):
        return spec
    return torch.device(f"cuda:{spec}" if torch.cuda.is_available() else "cpu")
