# -*- coding: utf-8 -*-
"""Kairos 侧的概率评测：做法③（分位数头）vs 做法①（点预测 + 残差自助）。

口径与 ``script/prob_cond_probe.py`` **逐字一致**——同一组起点、同一个公平 CRPS
估计量（``pb.crps_fair``）、同一套 crpsΣ 归一化（按训练期站均量级）。唯一变的是
"这个分布是从哪来的"：

  kairos_honest    Kairos-plain 点预测 + 验证段残差池·按小时              （现状）
  kairos_roll28    Kairos-plain 点预测 + 目标前 28 天残差池·按小时         （因果）
  kairos_oracleWA  Kairos-plain 点预测 + 测试期残差池·按(小时,周末,幅度)   （作弊，偏松的上限）
  kairos_quant     **分位数头**（pinball 训的 99 个均匀分位）              ← 做法③

  patch_roll28 / patch_oracleWA   同一个池、换成 PatchTST 点预测（参照）

为什么 kairos_quant 对 kairos_roll28 是有意义的对照
--------------------------------------------------
两者**共用同一根冻结的 Kairos 主干、同一批起点、同一份测试集**，差别只在头：
一个头输出一条点预测、分布靠外挂残差池；另一个头直接输出整条分位数曲线、
用 pinball（≡ CRPS）端到端训。所以两者的差 = **"直接训分布"相对"点预测外挂
残差池"的净收益**，主干与数据都不构成解释。

落到盘上的东西
--------------
只存 ``report.txt`` / ``report_<city>.txt`` / ``metrics_<city>.json``。分位数
张量**一律不落地**（AMS 上 [M*N,72,99] 是 TB 量级），全部边算边评。

用法
----
    python3 script/kairos_prob_eval.py [--cities LOA ...] [--device 0]
输出：``result/kairos_prob_eval/``
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from api.dataset.common import EVDataset, SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST  # noqa: E402
from api.utils import straight_window_keep  # noqa: E402

OUT = f"{BASE}/result/kairos_prob_eval"
os.makedirs(OUT, exist_ok=True)

ROLL_DAYS = 28


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pb = _load("prob_baseline", os.path.join(BASE, "script", "prob_baseline.py"))
pa = _load("prob_anchor_swap", os.path.join(BASE, "script", "prob_anchor_swap.py"))
tk = _load("train_kairos_l1", os.path.join(BASE, "script", "train_kairos_l1.py"))

SEQ_L = pa.SEQ_L          # 72，与基线的历史窗一致
M_SAMPLE = pa.M_SAMPLE    # 200，与基线一致
FOLD = 6


# ------------------------------------------------------------------ 数据
def load_city(city):
    """数据集侧：raw / 各段窗口起点 / 各段起点对应的时刻。"""
    ds = EVDataset(feature='volume', auxiliary='None',
                   data_path=f"{BASE}/data/{city}_remove_const/")
    ds.split_cross_validation(fold=FOLD, total_fold=FOLD, train_ratio=0.8, valid_ratio=0.1)
    raw = ds.feat                                    # [T, N]，已剔恒值站
    starts = {k: ds.window_starts(sid, SEQ_L, SEQ_L)
              for k, sid in (('va', SPLIT_VALID), ('te', SPLIT_TEST))}
    return ds, raw, starts


def obs_at(raw, starts, j):
    """horizon j（0-based，对应提前量 j+1 小时）的真值 [M, N]。"""
    return raw[starts + SEQ_L + j]


STRAIGHT_CSV = f"{BASE}/result/prob_baseline/straight_windows.csv"


def straight_keep(city, win_starts, cols, N):
    """旧口径：[M, N] bool，False = 该 (站, 窗) 的**目标块**是纯直线（CSV 标签）。

    只用于复现 ``--drop_straight`` 的历史结果。新口径请用 :func:`clean_keep`：
    不依赖 CSV、按原始序列直接判、且历史窗也检查。

    CSV 里的 ``station`` 是 ``pb.load`` 的 y0 列号，而评测用的 ``raw`` 只含
    ``pb.live_cols`` 选出的活跃列，所以要经 ``cols`` 映射一次。
    """
    keep = np.ones((len(win_starts), N), bool)
    if not os.path.exists(STRAIGHT_CSV):
        raise FileNotFoundError(f"缺少 {STRAIGHT_CSV}；先跑 script/detect_straight_windows.py")
    df = pd.read_csv(STRAIGHT_CSV)
    sub = df[(df.city == city) & (df.kind == 'A')]
    pos = {int(c): k for k, c in enumerate(cols)}
    wi = {int(s): i for i, s in enumerate(win_starts)}
    for ws, st in zip(sub.win_start.values, sub.station.values):
        if int(st) in pos and int(ws) in wi:
            keep[wi[int(ws)], pos[int(st)]] = False
    return keep


def clean_keep(raw, win_starts):
    """新口径：[M, N] bool，整段 72h 历史 + 72h 目标都不含合成直线段。

    模型逐站预测，所以剔除按 站×窗 做（同一时刻其他站的真实数据不受影响）。
    直接从 ``raw`` 算，和训练脚本 ``straight_window_keep`` 是同一实现。
    """
    return straight_window_keep(raw, win_starts, SEQ_L, SEQ_L)


# ------------------------------------------------------------------ 推理
TRAIN_BATCH = 512        # 每次前向的行数上限。**不是正确性要求**：Gate 的 bias 已在
                         # eval 下冻结，换批大小结果逐位相同（见 point_pass）。


def _forward(model, X, device, batch=TRAIN_BATCH):
    """X [n, seq_l] -> np [n, pre_len]（点头）或 [n, pre_len, Q]（分位头）。"""
    outs = []
    with torch.no_grad():
        for b in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[b:b + batch]).to(device)
            y_norm, (loc, scale) = model(xb)
            outs.append(model.denorm(y_norm, loc, scale).float().cpu().numpy())
    return np.concatenate(outs, 0)


def point_pass(model, raw, starts, js, device):
    """逐窗建窗 + 前向，**只留需要的三个 horizon**，不整存 [M,N,72]。

    AMS 上 M*N 有 230 万行，整存 72 步是 660 MB；只留 3 个 horizon 是 28 MB。

    分块按 512 行的整数倍切，**只是为了跟训练脚本的批边界一致、便于对账**，不是
    正确性要求。先前这里写着"批大小必须与训练时一致"，依据是"换批大小预测会有
    肉眼可见的差别（max|Δ| 9.5、19% 的元素超 1e-3）"——那个观测本身是真的，但
    归因错了：原因不是前向对批形状敏感，而是 Kairos 的 MoE Gate **每次前向都改
    自己的 bias**（load_error 对整个 batch 求和，批组成一变 bias 的走向就变）。
    见 api/model/kairos_l1.py 的 train()：eval 下已冻结该更新，任意批大小都可复现。
    """
    N = raw.shape[1]
    M = len(starts)
    step = TRAIN_BATCH // math.gcd(N, TRAIN_BATCH)         # 让 wchunk*N 是 512 的整数倍
    wchunk = max(step, (262144 // N // step) * step)
    off = np.arange(SEQ_L)[None, :]
    out = {j: np.zeros((M, N), np.float32) for j in js}
    for c0 in range(0, M, wchunk):
        st = starts[c0:c0 + wchunk]
        X = raw[st[:, None] + off]                       # [c, seq_l, N]
        X = np.transpose(X, (0, 2, 1)).reshape(-1, SEQ_L).astype(np.float32)
        P = _forward(model, X, device).reshape(len(st), N, SEQ_L)
        for j in js:
            out[j][c0:c0 + len(st)] = P[:, :, j]
    return out


def quant_pass(model, raw, starts, js, qs, u, device, cop_pool=None, keep=None):
    """分位头：**边算边评**，返回 {j: {'per_win': [M,N], ...}}。

    ``per_win[i]`` 是第 i 个窗口的**逐站 CRPS**（[N]，**每个窗口一行，不是累加**）。
    注意这与 :func:`bootstrap_variants` 里那个 [N] 的 ``per_st`` 约定不同 —— 那边是
    跨窗口累加、调用方再除以窗口数；这里存逐窗口原值，聚合在 :func:`run_city` 里做。
    （曾经这里叫 per_st 而 run_city 又按"累加"口径除了一个窗口数，结果把 crpsΣ 缩小了
    M=2125 倍，quant 一度看起来比一切基线好三个数量级。改名字就是为了让这个坑显形。）

    一次前向给全 72 步，但只切出需要的 j，且每批只放 w 个窗口（大数据集上 w=1），
    否则 [w*N, 72, Q] 直接爆显存。
    """
    N = raw.shape[1]
    M = len(starts)
    Q = len(qs)
    off = np.arange(SEQ_L)[None, :]
    # 每批窗口数：整窗处理（向量抽样要 N 个站齐），故按 512 行折算，大站退化成一次一窗。
    w = max(1, TRAIN_BATCH // N)
    res = {j: {'per_win': np.full((M, N), np.nan), 'cs_sum': 0.0, 'cs_sum_comono': 0.0,
               'cs_sum_cop': 0.0, 'med': np.zeros((M, N)), 'n_scored': 0,
               'nz_sum': 0.0, 'nz_win': 0, 'nz_cells': 0, 'nz_den': 0,
               'picp': {}, 'mpiw': {}} for j in js}
    # 分位头的"点预测"就是 τ=0.5 那根。存下来是为了跟 plain 变体的点预测头比 R²/MAE：
    # 边际 CRPS 变好，可能来自中心更准、也可能来自宽度更合适，不分开就说不清。
    k50 = int(np.argmin(np.abs(qs - 0.5)))
    assert abs(qs[k50] - 0.5) < 1e-9, f"分位网格里没有正好 0.5（k50={k50}, qs={qs[k50]}）"
    # 分位头只有边际 -> 联合分布没被参数化，把边际拼成"场景"必须人为选一个耦合。
    # 默认取**独立**（每站各自乱序取点）：JHB 上 ΣCRPS/Σ边际CRPS 的比值，独立耦合
    # 0.380~0.400，四种残差自助法 0.357~0.382 —— 同一条带；同单调是 0.482~0.508，
    # 高出一倍。所以独立耦合是唯一有数据支撑的口径，同单调仅作敏感性上界列出。
    u_ind = strat_u(np.random.default_rng(0), len(u), N, shared=False)
    # 经验耦合的随机源。每窗从验证期残差场**不放回**抽 U 条：放回会产生并列值，
    # 并列处 argsort 的次序是任意的，会污染 copula 的名次。
    rng_cop = np.random.default_rng(0)
    for c0 in range(0, M, w):
        st = starts[c0:c0 + w]
        X = raw[st[:, None] + off]
        X = np.transpose(X, (0, 2, 1)).reshape(-1, SEQ_L).astype(np.float32)
        P = _forward(model, X, device).reshape(len(st), N, SEQ_L, Q)
        for j in js:
            qv = P[:, :, j, :]                            # [w, N, Q]
            for m in range(len(st)):
                y = raw[st[m] + SEQ_L + j]                # [N] 真值
                mk = np.ones(N, bool) if keep is None else keep[c0 + m]
                if not mk.any():
                    continue
                smp = quant_to_samples(qv[m], qs, u)               # [U, N] 同单调（只用于边际）
                smp_ind = quant_to_samples_ind(qv[m], qs, u_ind)   # [U, N] 独立
                ym = y[mk]
                res[j]['per_win'][c0 + m, mk] = pb.crps_fair(smp[:, mk], ym)
                res[j]['med'][c0 + m] = qv[m][:, k50]
                res[j]['n_scored'] += 1
                res[j]['nz_den'] += int(mk.sum())
                v_nz, n_nz = nz_crps(smp[:, mk], ym)         # 条件 CRPS，见 nz_crps
                if n_nz:
                    res[j]['nz_sum'] += v_nz
                    res[j]['nz_win'] += 1
                    res[j]['nz_cells'] += n_nz
                # CRPS_sum（主指标）。边际指标与耦合无关（两种拼法逐站的分位数一模一样，
                # PICP/MPIW/边际 CRPS 数值相同），差别只在跨站那一步：独立压 Σ_站 方差、
                # 同单调顶到最大。真实残差场落在两者之间，且按上面的比值证据更靠独立端。
                res[j]['cs_sum'] += float(pb.crps_fair(smp_ind[:, mk].sum(1), ym.sum()))
                res[j]['cs_sum_comono'] += float(pb.crps_fair(smp[:, mk].sum(1), ym.sum()))
                # 第三点：把**同一批边际**按验证期残差的跨站排序模式重排。边际指标
                # （crpsΣ/PICP/MPIW）与上面两行逐值相同，只有 crpsΣsum 会动。
                if cop_pool is not None and len(cop_pool.get(j, ())) >= 2:
                    pool = cop_pool[j]
                    sel = rng_cop.choice(len(pool), len(u),
                                         replace=len(pool) < len(u))
                    smp_cop = quant_to_samples_copula(qv[m], qs, u_ind, pool[sel])
                    res[j]['cs_sum_cop'] += float(
                        pb.crps_fair(smp_cop[:, mk].sum(1), ym.sum()))
                pm = pb.pit_metrics(smp[:, mk], ym)
                for k, v in pm.items():
                    d = res[j]['picp'] if k.startswith('picp') else res[j]['mpiw']
                    d[k] = d.get(k, 0.0) + v
    for j in js:
        n = max(1, res[j]['n_scored'])
        for k in res[j]['picp']:
            res[j]['picp'][k] /= n
        for k in res[j]['mpiw']:
            res[j]['mpiw'][k] /= n
    return res


def quant_to_samples(qv, qs, u):
    """把 [N, Q] 的分位数曲线插值成 [U, N] 样本（逆 CDF 线性插值）。

    u 用**分层**样本 (k+0.5)/U 而不是随机数：同样 200 条，估的是同一条曲线，
    但没有蒙特卡洛抖动，跑两次结果一致。
    """
    k = np.clip(np.searchsorted(qs, u), 1, len(qs) - 1)
    lo, hi = qv[:, k - 1], qv[:, k]                       # [N, U]
    frac = ((u - qs[k - 1]) / (qs[k] - qs[k - 1]))[None, :]
    return (lo + frac * (hi - lo)).T                      # [U, N]


def strat_u(rng, U, N, shared):
    """生成 U 条样本的分位点网格 [U,N]。

    ``shared=True``  : 所有站共用同一列 —— **完全同单调**，站间波动被强行绑死，
                       Σ_站 的方差被顶到最大（给定边际下）。
    ``shared=False`` : 每站把同一组分层点**随机重排** —— 站间**独立**耦合，
                       Σ_站 的方差被压到最小。

    两者都不是真实跨站相关：逐站分位头只建模边际，联合分布根本没被参数化。
    所以**两个极端都要量**。真实那一端已由 script/prob_resid_corr.py 实测：残差场
    标准化后的聚合波动是独立假设的 2.2~202 倍（**不是 1**），但落到 CRPS_sum 上
    只差 1~16%，且方向是**实测耦合比独立耦合更低**——即独立耦合对分位头偏保守，
    不是偏袒。第三点由 :func:`quant_to_samples_copula` 给出。
    """
    base = (np.arange(U) + 0.5) / U
    if shared:
        return np.repeat(base[:, None], N, axis=1)
    return (np.argsort(rng.random((U, N)), axis=0) + 0.5) / U


def quant_to_samples_ind(qv, qs, u2d):
    """同 :func:`quant_to_samples`，但 u 是 [U,N]（每站一列，可各不相同）。"""
    U, N = u2d.shape
    qvT = qv.T                                            # [Q, N]
    k = np.clip(np.searchsorted(qs, u2d.ravel()), 1, len(qs) - 1).reshape(U, N)
    nidx = np.arange(N)[None, :]
    lo, hi = qvT[k - 1, nidx], qvT[k, nidx]               # [U, N]
    frac = (u2d - qs[k - 1]) / (qs[k] - qs[k - 1])
    return lo + frac * (hi - lo)


def quant_to_samples_copula(qv, qs, u2d, Z):
    """经验 copula（Iman–Conover 重排）：借真实残差向量的**跨站排序模式**。

    qv[N,Q] 分位曲线，u2d[U,N] 分层网格（与独立那路完全同一张），Z[U,N] 真实残差
    向量的抽样。做法是：先按独立那路算出 [U,N] 样本，再**逐列排序、按 Z 的名次重排**。

    这样 **每一列的取值集合与独立那路逐值相同**（只是换了个顺序）——所以 crpsΣ、
    PICP、MPIW 一个字都不会变，唯一变的是站与站的依赖。这正是我们要隔离的那一项。

    为什么需要它：分位头只参数化边际，跨站耦合是代码里选的。独立（Σ_站 方差最小）
    与同单调（最大）是两个**人造极端**；真实结构由 script/prob_resid_corr.py 实测为
    "独立耦合偏低 1~16%"（整条/独立 0.84~0.99，六城六 horizon 全部 <1）。所以独立
    是**保守**的那一端，不是偏袒。本函数给出的是有数据依据的第三点。
    """
    base = quant_to_samples_ind(qv, qs, u2d)                  # [U, N] 独立
    rank = np.argsort(np.argsort(Z, axis=0), axis=0)          # [U, N] 每列名次
    return np.take_along_axis(np.sort(base, axis=0), rank, axis=0)


def build_kairos(city, variant, out_root, device, n_quant):
    """重建模型并载入该城该变体的 .pth（含冻结主干，strict 载入成立）。"""
    ckpt = os.path.join(out_root, city, f"{variant}_pre{SEQ_L}_fold{FOLD}.pth")
    if not os.path.exists(ckpt):
        return None
    sys.path.insert(0, '/tmp/Kairos_repo')
    import tsfm.model.kairos  # noqa: F401
    from transformers import AutoModel
    from api.model.kairos_l1 import KairosL1
    bb = AutoModel.from_pretrained('mldi-lab/Kairos_23m',
                                   trust_remote_code=True).to(device).eval()
    qs = np.arange(1, n_quant + 1) / (n_quant + 1) if variant == 'quant' else None
    model = KairosL1(bb, pre_len=SEQ_L, quantiles=qs).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


# ------------------------------------------------------------------ 残差自助
def _tertile(x, grp, min_n=3):
    b = np.zeros(len(x), np.int8)
    for g in np.unique(grp):
        m = grp == g
        if m.sum() < min_n:
            continue
        q1, q2 = np.quantile(x[m], [1 / 3, 2 / 3])
        b[m] = (x[m] > q1).astype(np.int8) + (x[m] > q2).astype(np.int8)
    return b


def nz_crps(smp, obs):
    """**只在真值 > 0 的格子上**算 CRPS，返回 (该窗逐格均值, 非零格数)。

    为什么单列这一项 —— 满分门卫（``--csdi_mask oracle``）下 csdi_pass 里那句
    ``ysim *= m`` 会把真零格**直接写成 0**，那些格子对做法④ 是白送的满分（真值也
    是 0，CRPS=0）；而做法③ 在同样的格子上还得输出一条 P(y=0)≈0.4 的分布，照罚。
    所以"满分门卫下 ④ 的边际第一"里混着"零格白送"这一笔，**不能**直接当成
    "画师画得好"。剔掉真零格，剩下的才是"非零格上谁更准"——那才是画师的成绩。

    注意这不是"去掉零"的意思：非零格上的条件分布形状（宽度、偏度）仍然是画师
    的全部责任，这里只是把"这一格该不该是 0"这个**门卫的**问题排除掉。
    """
    m = obs > 0
    if not m.any():
        return np.nan, 0
    return float(pb.crps_fair(smp[:, m], obs[m]).mean()), int(m.sum())


def bootstrap_variants(pred_seg, y, starts_seg, ok, hod, wkd, js, smean, N, rng,
                       keep_te=None):
    """做法①：点预测（kairos 或 PatchTST）+ 残差池抽样。

    pred_seg: {'va': {j:[M,N]}, 'te': {j:[M,N]}}，与 starts_seg 同序。
    与 prob_cond_probe 保持同一套池与条件（hour / wk / amp）。
    keep_te: [M,N] bool，False 的格子不计入任何指标（'--drop_straight' 用）。
    每站的除数用**它自己实际被计分的窗口数**，不是总窗口数——否则被剔除的格子
    相当于按 0 分计入，和以前 nan_to_num 那个坑是同一种美化。
    """
    out = {}
    for j in js:
        H = j + 1
        lag = pb.LAG * int(np.ceil(H / pb.LAG))
        src = {}
        for nm in ('va', 'te'):
            s = np.asarray(starts_seg[nm]) + SEQ_L + j
            m = ok[s - lag]
            s = s[m]
            src[nm] = dict(s=s, obs=y[s], pred=pred_seg[nm][j][m],
                           hour=hod[s], wk=wkd[s],
                           # keep_te 与 starts_seg['te'] 同序；这里同一个 m 筛过
                           keep=(keep_te[m] if (nm == 'te' and keep_te is not None) else None))
        all_pred = np.concatenate([src['va']['pred'], src['te']['pred']], 0)
        all_hour = np.concatenate([src['va']['hour'], src['te']['hour']])
        amp = _tertile(all_pred.sum(1), all_hour)
        nva = len(src['va']['s'])
        src['va']['amp'], src['te']['amp'] = amp[:nva], amp[nva:]

        s_h = np.concatenate([src['va']['s'], src['te']['s']])
        res_h = np.concatenate([src['va']['obs'] - src['va']['pred'],
                                src['te']['obs'] - src['te']['pred']])
        hour_h = all_hour
        wk_h = np.concatenate([src['va']['wk'], src['te']['wk']])
        amp_h = amp

        te = src['te']
        P = len(te['s'])
        sn = float(np.nanmean(smean))
        for name, kind, conds in (('honest', 'va', ('hour',)),
                                  ('roll28', 'roll', ('hour',)),
                                  ('oracleWA', 'te', ('hour', 'wk', 'amp'))):
            per_st = np.zeros(N)
            cell_cnt = np.zeros(N)      # 每站实际被计分的窗口数（剔除后逐站不同）
            sum_crps, pit_acc = 0.0, {}
            nz_sum, nz_win, nz_cells, nz_den = 0.0, 0, 0, 0
            # T4/T5：与 baselines.StreamingMetrics 同一口径
            c0_st, cp_st = np.zeros(N), np.zeros(N)
            c0_sum, cp_sum, c0_n, cp_n = 0.0, 0.0, 0, 0
            wis_st, wis_st_cnt = np.zeros(N), np.zeros(N)
            wis_nz_sum, wis_nz_win = 0.0, 0
            wis_zero_sum, wis_zero_win = 0.0, 0
            pool_sz = []
            used = 0        # 真正有池、被计分的窗口数（跳过的不该按 0 分算进平均）
            for i in range(P):
                if kind == 'roll':
                    base = ((hour_h == te['hour'][i]) & (s_h < te['s'][i]) &
                            (s_h >= te['s'][i] - ROLL_DAYS * 24))
                    pool_res, pool_hour, pool_wk, pool_amp = res_h, hour_h, wk_h, amp_h
                else:
                    base = np.ones(len(src[kind]['s']), bool)
                    pool_res = src[kind]['obs'] - src[kind]['pred']
                    pool_hour, pool_wk, pool_amp = (src[kind]['hour'],
                                                    src[kind]['wk'], src[kind]['amp'])
                m = base
                if 'hour' in conds:
                    m = m & (pool_hour == te['hour'][i])
                if 'wk' in conds:
                    m = m & (pool_wk == te['wk'][i])
                if 'amp' in conds:
                    m = m & (pool_amp == te['amp'][i])
                idx = np.where(m)[0]
                if len(idx) == 0:
                    continue
                # 站×窗级剔除：同一条残差向量只作用在保留站上。
                kp = te.get('keep')
                mk = np.ones(N, bool) if kp is None else kp[i]
                if not mk.any():
                    continue
                pool_sz.append(len(idx))
                used += 1
                nz_den += int(mk.sum())
                pick = rng.choice(idx, min(M_SAMPLE, len(idx)), replace=True)
                smp = te['pred'][i][None, :] + pool_res[pick]      # [m, N] 整条向量
                obs = te['obs'][i]
                c_st = pb.crps_fair(smp[:, mk], obs[mk])
                # 这里以前是 np.nan_to_num —— 把算不出来的站**静默记成 0 分**（完美预测），
                # 于是边际指标被悄悄美化。现在不兜底了：真出非有限值就炸出来看。
                assert np.all(np.isfinite(c_st)), f"{name} H={H} 窗口 {i}: 逐站 CRPS 非有限"
                per_st[mk] += c_st
                cell_cnt[mk] += 1

                # T5：零格/正格精确分解（逐站累计）
                obs_m = obs[mk]
                zero = (obs_m == 0)
                pos = ~zero
                mk_idx = np.flatnonzero(mk)
                if zero.any():
                    c0_st[mk_idx[zero]] += c_st[zero]
                    c0_sum += float(c_st[zero].sum())
                    c0_n += int(zero.sum())
                if pos.any():
                    cp_st[mk_idx[pos]] += c_st[pos]
                    cp_sum += float(c_st[pos].sum())
                    cp_n += int(pos.sum())

                # T4：WIS（同一批冻结残差池样本）
                wis_cells = pb.wis_from_samples(smp[:, mk], obs_m)
                wis_st[mk_idx] += wis_cells
                wis_st_cnt[mk_idx] += 1.0
                if pos.any():
                    wis_nz_sum += float(wis_cells[pos].mean())
                    wis_nz_win += 1
                if zero.any():
                    wis_zero_sum += float(wis_cells[zero].mean())
                    wis_zero_win += 1
                # CRPS_sum 是**主指标**（先把同一时刻所有站的值加总、再算 CRPS）。
                # 抽的是整条残差向量，站间同源结构原样带进来 —— 这正是逐站分位头
                # 做不到的部分：分位头只能产边际，跨站要么独立、要么强行同分位。
                cs = float(pb.crps_fair(smp[:, mk].sum(1), obs[mk].sum()))
                assert np.isfinite(cs), f"{name} H={H} 窗口 {i}: CRPS_sum 非有限"
                sum_crps += cs
                v_nz, n_nz = nz_crps(smp[:, mk], obs[mk])
                if n_nz:
                    nz_sum, nz_win, nz_cells = nz_sum + v_nz, nz_win + 1, nz_cells + n_nz
                for k, v in pb.pit_metrics(smp[:, mk], obs[mk]).items():
                    pit_acc[k] = pit_acc.get(k, 0.0) + v
            # 三个统计量同除 `used`（被计分的窗口数），跳过的不参与任何平均。
            # 以前除的是全量 P —— 跳过的窗口等于按"0 CRPS"计入，同样是美化。
            assert used > 0, f"{name} H={H} 一个窗口都没计分"
            # 逐站除自己的计分窗口数：剔除是站×窗级的，不能拿全城统一的 used。
            cnt = np.maximum(cell_cnt, 1)
            c0_raw = float(np.mean(c0_st / cnt))
            cp_raw = float(np.mean(cp_st / cnt))
            crps_norm = float(np.mean(per_st / cnt) / sn)
            if not np.isclose(c0_raw + cp_raw,
                              np.mean(per_st / cnt), rtol=1e-6, atol=1e-8):
                raise AssertionError(
                    f"{name} H={H}: CRPS 零/正分解不闭合 "
                    f"{c0_raw:.8f}+{cp_raw:.8f}")
            wis_raw = float(np.mean(wis_st / cnt))
            out.setdefault(name, {})[j] = dict(
                crps_norm_sum=crps_norm,
                crps_sum_norm=float(sum_crps / used / np.nansum(smean)),
                n_pool_med=int(np.median(pool_sz)) if pool_sz else 0,
                n_used=used, n_skip=P - used,
                # 条件 CRPS：剔掉真零格后的逐格 CRPS / 非零格占比（见 nz_crps 的说明）
                crps_nz=float(nz_sum / nz_win / sn) if nz_win else float('nan'),
                nz_rate=float(nz_cells / max(nz_den, 1)) if nz_win else float('nan'),
                # T4
                wis=float(wis_raw / sn),
                wis_nz=float(wis_nz_sum / max(wis_nz_win, 1) / sn),
                wis_zero=float(wis_zero_sum / max(wis_zero_win, 1) / sn),
                # T5（normalized 后两项严格加回 crps_norm_sum）
                crps_zero_contrib=float(c0_raw / sn),
                crps_pos_contrib=float(cp_raw / sn),
                crps_zero_mean=float(c0_sum / max(c0_n, 1)),
                crps_pos_mean=float(cp_sum / max(cp_n, 1)),
                zero_cell_frac=float(c0_n / max(c0_n + cp_n, 1)),
                n_zero_cells=int(c0_n),
                n_pos_cells=int(cp_n),
                **{k: v / used for k, v in pit_acc.items()})
    return out


# ------------------------------------------------------------------ 主流程
# ------------------------------------------------------------------ 整块扩散
CSDI_ROOT = f"{BASE}/result/csdi"
CSDI_NSAMPLE = 24          # 每窗采样条数。CRPS 用公平估计量，24 条的估计噪声在
                           # 2125 个窗口上平均掉；再大就纯粹是算力换不到的精度。
CSDI_STRIDE = 50           # DDIM 跳步：1000 步只走 20 步。**步数改了这里要同步放大**，
                           # 否则采样从 20 步变回 1000 步，慢 50 倍。
CSDI_CHUNK = 16            # 一次进扩散循环的窗口数**上限**（样本维会再乘 CSDI_NSAMPLE）。
# 每块的格子预算（样本×窗×站×时长）。固定窗口数在大站城市会炸：AMS 1383 站、72 时长，
# chunk_w=16 时中间特征图 [24·16, 64, 1383, 72] 单张就要 ~9.8 GB，整条采样链峰值 >30 GB，
# 23.7 GB 的卡直接 OOM（2026-10-14 五城评测就死在这）。按格子数折算窗口数，站多的城
# 自动缩到 2~3，站少的城仍是 CSDI_CHUNK。预算 8M 格 ≈ 单张特征图 2 GB。
CSDI_CELL_BUDGET = 8_000_000

# 离散度扫描：把扩散生成的**残差部分**整体乘 k（pivot 不动），在**同一次采样**里顺带
# 算几档指标，不额外重采样，所以是零成本的。
# 为什么要有它：实测做法④/⑤ 的 PICP@90 只有 0.61~0.81（名义 0.90），即样本"太窄"。
#   * 若某个 k>1 能把 crpsΣ 明显压到 ③（分位头）以下  →  欠离散是个**全局尺度**问题，
#     等于免费的标定旋钮，换扩散花式（flow matching / x0 参数化 / 别的调度）救不到点上；
#   * 若整条 k 曲线都压不下去、或者一放大就更差  →  问题在**形状**（边际分布本身就是歪的），
#     那才轮到"换一种扩散"这条路。
# 这一列只回答归因，不进正式表（正式表里 k=1 就是现状）。
SPREAD_SWEEP = []

# 只用来**打印一个具体例子**：在第 PEEK[0] 个测试窗口、第 PEEK[1] 号站上，把做法④
# 生成的 24 条样本原样打出来，配真值。目的是让人肉眼看清"24 条样本"长什么样、
# "太窄/罩不住"具体指什么，不是指标计算的一部分。
PEEK = None


def load_csdi(city, device):
    """载入该城训练好的 CSDI（整块扩散头 + on/off 分类器）。没有就返回 None。"""
    p = f"{CSDI_ROOT}/{city}_csdi.pth"
    if not os.path.exists(p):
        return None
    from api.model.csdi import ScoreNet, OnOffNet, GaussianDiffusion
    ck = torch.load(p, map_location=device)
    # 参数化必须对得上：x0-预测的权重拿去当 ε-预测用不会报形状错，只会静默跑出垃圾
    # （这正是上一版症状 —— 样本系统性偏高、点预测 R² 变负）。
    assert ck.get('param') == ScoreNet.PARAM, (
        f"{p} 是旧参数化（{ck.get('param')!r}），请用 script/train_csdi.py 重训")
    net = ScoreNet(cond_ch=7, base=ck['base']).to(device).eval()
    ono = OnOffNet(cond_ch=6, base=ck['base']).to(device).eval()
    net.load_state_dict(ck['net'])
    ono.load_state_dict(ck['ono'])
    for q in (net, ono):
        for par in q.parameters():
            par.requires_grad_(False)
    return dict(net=net, ono=ono, dif=GaussianDiffusion(steps=ck['steps']).to(device),
                hid=np.asarray(ck['hid']), scale=ck['scale'], bias=ck['bias'],
                # Platt 校准的两个标量（训练脚本在校验段上拟的）。老 ckpt 没有就退化成
                # 恒等，不会因为缺字段而崩。
                cal=tuple(ck.get('cal', (1.0, 0.0))),
                # 条件逐路标准化的统计量（训练侧算好存进来的）。老 ckpt 没有就退化成
                # 不标准化 —— 但那种 ckpt 的权重是按另一套口径训的，见 param 断言。
                cstats=ck.get('cond_stats'))


def csdi_pass(core, pred, hod, ytrue, hist_on, device, js, n_samples=CSDI_NSAMPLE,
              stride=CSDI_STRIDE, chunk_w=CSDI_CHUNK, mask_mode='clf', keep=None,
              eta=0.0):
    """整块扩散采样 -> 逐窗指标；返回结构与 :func:`quant_pass` 一致。

    **这里没有"跨站怎么拼"这个自由度**：一次采样的产物就是一整块 [站 × 72 小时]
    的场景，站与站的关系由去噪网络自己学。所以 cs_sum 这一列在它身上是干净的，
    可以直接跟做法① 比（做法① 抽的是整条残差向量，同样没有耦合自由度）。

    pred / ytrue / hist_on 都是 [M,N,72]（冻结主干的点预测 / 真值 / **历史**窗口的
    on/off 图案），hod 是 [M,72] 目标小时数。hist_on 是预测原点之前那 72 小时的真实
    观测 —— 分类器靠它判断"这个站最近活跃吗"，训练与评测必须给同一份口径，否则
    分类器在评测时看到的通道全是 0，会退化成常数。
    站轴按 ckpt 里存的 Hilbert 空间序排（卷积要在站轴上做"相邻即空间相邻"的假设），
    算完再排回原始列序，保证与 ytrue 对得上。
    """
    from api.model.csdi import base_cond, pack_cond, hist_feats
    N, M = pred.shape[1], pred.shape[0]
    # 窗口数按格子预算折算（站多自动缩小），否则 AMS 这种大站城会 OOM。见 CSDI_CELL_BUDGET。
    chunk_w = max(1, min(chunk_w, CSDI_CELL_BUDGET // (n_samples * N * SEQ_L)))
    hid, inv = core['hid'], np.argsort(core['hid'])
    m_t = torch.from_numpy(np.asarray(core['bias'])[hid]).float().to(device)
    s_t = torch.from_numpy(np.asarray(core['scale'])[hid]).float().to(device)
    hd = torch.from_numpy(hod).float()
    ho = torch.from_numpy(np.ascontiguousarray(hist_on[:, hid, :])).float()
    gen = torch.Generator(device=device).manual_seed(0)
    res = {j: {'per_win': np.full((M, N), np.nan), 'cs_sum': 0.0, 'med': np.zeros((M, N)),
               'n_scored': 0, 'nz_sum': 0.0, 'nz_win': 0, 'nz_cells': 0, 'nz_den': 0,
               'picp': {}, 'mpiw': {}} for j in js}
    # k≠1 那几档的独立累加器（结构同上，只留诊断要用的三样）。
    sw = {float(kk): {j: {'per_win': np.zeros((M, N)), 'cell_cnt': np.zeros(N),
                          'cs_sum': 0.0, 'picp': {}}
                      for j in js} for kk in SPREAD_SWEEP}
    psp = pred[:, hid, :]
    for c0 in range(0, M, chunk_w):
        hi = min(c0 + chunk_w, M)
        with torch.no_grad():
            pr = torch.from_numpy(np.ascontiguousarray(psp[c0:hi])).float().to(device)
            a_c, o_c = hist_feats(ho[c0:hi].to(device))
            base = base_cond(pr, s_t, hd[c0:hi].to(device)[:, None, :],
                             a_c, o_c, N, SEQ_L)
            ca, cb = core['cal']
            p_on = torch.sigmoid(ca * core['ono'](base) + cb)
            if mask_mode == 'oracle':
                # **诊断专用**：拿真值 on/off 当掩码，量"扩散本身还剩多少"。它泄漏真值，
                # 只能当上限看，不能进正式表。掩码要在**空间序**上（卷积在站轴上做），
                # 而 ytrue 是原始列序，所以先按 hid 重排。
                m = torch.from_numpy(np.ascontiguousarray(
                    ytrue[c0:hi][:, hid, :] > 0)).float().to(device)[:, None]
            else:
                # on/off 与数值都**在整块上一起抽**：分类器是全卷积的，它的相关性也来自
                # 空间上下文；逐格独立抽会把这份相关性丢掉。
                m = (torch.rand(p_on.shape, device=device, generator=gen) < p_on).float()
            cond = pack_cond(base, p_on, N, SEQ_L, core.get('cstats'))
            smp = core['dif'].sample(core['net'], cond, m, torch.zeros_like(m),
                                     n_samples=n_samples, generator=gen,
                                     stride=stride, eta=eta)             # [S,b,1,N,72]
            # 还原：gen=1 的格子 y = pred + 偏置 + 尺度×扩散输出；gen=0 的格子恒为 0。
            ysim = pr[None, :, None] + m_t[None, None, :, None] \
                + s_t[None, None, :, None] * smp
            ysim = (ysim * m[None]).cpu().numpy()[:, :, 0][:, :, inv, :]   # [S,b,N,72]
            if SPREAD_SWEEP:
                # 离散度扫描：pivot（点预测+偏置）不动，只把扩散残差乘 k，掩码原样。
                # 同一次采样的产物换个尺度再用一遍，**不额外重采样**，所以是零成本诊断。
                piv = pr[None, :, None] + m_t[None, None, :, None]
                rds = s_t[None, None, :, None] * smp
                swv = {float(kk): ((piv + kk * rds) * m[None]).cpu().numpy()[:, :, 0][:, :, inv, :]
                       for kk in SPREAD_SWEEP}
            yt = ytrue[c0:hi]
        for j in js:
            for k in range(hi - c0):
                sm = ysim[:, k, :, j]                                     # [S,N]
                truth = yt[k, :, j]
                mk = np.ones(N, bool) if keep is None else keep[c0 + k]
                if not mk.any():
                    continue
                res[j]['per_win'][c0 + k, mk] = pb.crps_fair(sm[:, mk], truth[mk])
                res[j]['cs_sum'] += float(pb.crps_fair(sm[:, mk].sum(1), truth[mk].sum()))
                res[j]['med'][c0 + k] = np.median(sm, 0)
                res[j]['n_scored'] += 1
                res[j]['nz_den'] += int(mk.sum())
                v_nz, n_nz = nz_crps(sm[:, mk], truth[mk])              # 条件 CRPS，见 nz_crps
                if n_nz:
                    res[j]['nz_sum'] += v_nz
                    res[j]['nz_win'] += 1
                    res[j]['nz_cells'] += n_nz
                for kk, v in pb.pit_metrics(sm[:, mk], truth[mk]).items():
                    d = res[j]['picp'] if kk.startswith('picp') else res[j]['mpiw']
                    d[kk] = d.get(kk, 0.0) + v
                if PEEK is not None and (c0 + k) == PEEK[0]:
                    # 打印这一个格子上的全部 24 条样本（排好序）与真值，供人眼核对。
                    st = PEEK[1]
                    v = np.sort(sm[:, st])
                    lo, hi_ = v[0], v[-1]
                    q = np.quantile(v, [0.10, 0.50, 0.90])
                    cover = "落在范围里" if lo <= truth[st] <= hi_ else "**跑到范围外了**"
                    print(f"[peek] 窗口{PEEK[0]} 第{st}号站 H={j + 1}：", flush=True)
                    print("       24 条样本 = " + " ".join(f"{x:.2f}" for x in v), flush=True)
                    print(f"       真值 = {truth[st]:.2f}   样本范围 [{lo:.2f}, {hi_:.2f}]  → 真值{cover}",
                          flush=True)
                    print(f"       样本的 10%/50%/90% 分位点 = {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}",
                          flush=True)
                if SPREAD_SWEEP:
                    for kk in SPREAD_SWEEP:
                        smk = swv[float(kk)][:, k, :, j]
                        acc = sw[float(kk)][j]
                        acc['per_win'][c0 + k, mk] = pb.crps_fair(smk[:, mk], truth[mk])
                        acc['cell_cnt'][mk] += 1
                        acc['cs_sum'] += float(pb.crps_fair(smk[:, mk].sum(1), truth[mk].sum()))
                        for kp, v in pb.pit_metrics(smk[:, mk], truth[mk]).items():
                            if kp.startswith('picp'):
                                acc['picp'][kp] = acc['picp'].get(kp, 0.0) + v
        del smp, ysim
    for j in js:
        n_sc = max(1, res[j]['n_scored'])
        for d in (res[j]['picp'], res[j]['mpiw']):
            for kk in d:
                d[kk] /= n_sc
        # 扫描那几档聚成 [N]（与主口径同一条公式，只是换了尺度），标量留给 run_city 收尾。
        res[j]['sweep'] = {}
        for kk in SPREAD_SWEEP:
            a = sw[float(kk)][j]
            res[j]['sweep'][float(kk)] = dict(per_st=np.nansum(a['per_win'], 0),
                                              cell_cnt=np.maximum(a['cell_cnt'], 1),
                                              cs_sum=a['cs_sum'],
                                              picp={k2: v for k2, v in a['picp'].items()})
    return res


def load_zidiff(city, device):
    """载入该城训练好的**吸收态扩散**。没有就返回 None。"""
    p = f"{CSDI_ROOT}/{city}_zidiff.pth"
    if not os.path.exists(p):
        return None
    from api.model.zidiff import ZiScoreNet, ZiDiffusion
    ck = torch.load(p, map_location=device)
    assert ck.get('param') == ZiScoreNet.PARAM, (
        f"{p} 的参数化是 {ck.get('param')!r}，不是吸收态扩散")
    net = ZiScoreNet(cond_ch=6, base=ck['base']).to(device).eval()
    net.load_state_dict(ck['net'])
    for par in net.parameters():
        par.requires_grad_(False)
    cst = ck.get('cond_stats')
    return dict(net=net, dif=ZiDiffusion(steps=ck['steps']).to(device),
                hid=np.asarray(ck['hid']), scale=ck['scale'], bias=ck['bias'],
                cstats=tuple(x.to(device) for x in cst) if cst is not None else None)


def zidiff_pass(core, pred, hod, ytrue, hist_on, device, js, n_samples=CSDI_NSAMPLE,
                stride=CSDI_STRIDE, chunk_w=CSDI_CHUNK, keep=None, eta=0.0):
    """吸收态扩散采样 -> 逐窗指标；返回结构与 :func:`csdi_pass` 一致。

    与 csdi_pass 的两处**关键差别**：

    1. **没有独立的 on/off 分类器**，掩码由同一个网络在采样链上反复重抽，所以这里
       根本不构造 ``m``、也没有 ``oracle`` 这个作弊口径（没有可替换的门卫）。
    2. 还原时**不乘掩码**：被吸收的格子归零这件事发生在 :meth:`ZiDiffusion.sample`
       内部（``x * g`` 那一步），出来的 ``smp`` 已经是"零格恒为 0、正格带值"的场景。
    """
    from api.model.csdi import base_cond, hist_feats
    N, M = pred.shape[1], pred.shape[0]
    chunk_w = max(1, min(chunk_w, CSDI_CELL_BUDGET // (n_samples * N * SEQ_L)))
    hid, inv = core['hid'], np.argsort(core['hid'])
    m_t = torch.from_numpy(np.asarray(core['bias'])[hid]).float().to(device)
    s_t = torch.from_numpy(np.asarray(core['scale'])[hid]).float().to(device)
    hd = torch.from_numpy(hod).float()
    ho = torch.from_numpy(np.ascontiguousarray(hist_on[:, hid, :])).float()
    gen = torch.Generator(device=device).manual_seed(0)
    res = {j: {'per_win': np.full((M, N), np.nan), 'cs_sum': 0.0, 'med': np.zeros((M, N)),
               'n_scored': 0, 'nz_sum': 0.0, 'nz_win': 0, 'nz_cells': 0, 'nz_den': 0,
               'picp': {}, 'mpiw': {}} for j in js}
    # k≠1 那几档的独立累加器（结构同上，只留诊断要用的三样）。
    sw = {float(kk): {j: {'per_win': np.zeros((M, N)), 'cell_cnt': np.zeros(N),
                          'cs_sum': 0.0, 'picp': {}}
                      for j in js} for kk in SPREAD_SWEEP}
    psp = pred[:, hid, :]
    for c0 in range(0, M, chunk_w):
        hi = min(c0 + chunk_w, M)
        with torch.no_grad():
            pr = torch.from_numpy(np.ascontiguousarray(psp[c0:hi])).float().to(device)
            a_c, o_c = hist_feats(ho[c0:hi].to(device))
            cond = base_cond(pr, s_t, hd[c0:hi].to(device)[:, None, :], a_c, o_c, N, SEQ_L)
            if core.get('cstats') is not None:
                cm, cs = core['cstats']
                cond = (cond - cm) / cs
            smp = core['dif'].sample(core['net'], cond, n_samples=n_samples,
                                     generator=gen, stride=stride,
                                     eta=eta)                          # [S,b,1,N,72]
            ysim = pr[None, :, None] + m_t[None, None, :, None] \
                + s_t[None, None, :, None] * smp
            ysim = ysim.cpu().numpy()[:, :, 0][:, :, inv, :]               # [S,b,N,72]
            if SPREAD_SWEEP:
                # 同 csdi_pass：只缩放扩散残差。这里**没有**额外乘掩码 —— 掩码归零已经
                # 发生在 sample() 内部（x*g），乘 k 只放大"有值格"上的起伏。
                piv = pr[None, :, None] + m_t[None, None, :, None]
                rds = s_t[None, None, :, None] * smp
                swv = {float(kk): (piv + kk * rds).cpu().numpy()[:, :, 0][:, :, inv, :]
                       for kk in SPREAD_SWEEP}
            yt = ytrue[c0:hi]
        for j in js:
            for k in range(hi - c0):
                sm = ysim[:, k, :, j]
                truth = yt[k, :, j]
                mk = np.ones(N, bool) if keep is None else keep[c0 + k]
                if not mk.any():
                    continue
                res[j]['per_win'][c0 + k, mk] = pb.crps_fair(sm[:, mk], truth[mk])
                res[j]['cs_sum'] += float(pb.crps_fair(sm[:, mk].sum(1), truth[mk].sum()))
                res[j]['med'][c0 + k] = np.median(sm, 0)
                res[j]['n_scored'] += 1
                res[j]['nz_den'] += int(mk.sum())
                v_nz, n_nz = nz_crps(sm[:, mk], truth[mk])
                if n_nz:
                    res[j]['nz_sum'] += v_nz
                    res[j]['nz_win'] += 1
                    res[j]['nz_cells'] += n_nz
                for kk, v in pb.pit_metrics(sm[:, mk], truth[mk]).items():
                    d = res[j]['picp'] if kk.startswith('picp') else res[j]['mpiw']
                    d[kk] = d.get(kk, 0.0) + v
                if PEEK is not None and (c0 + k) == PEEK[0]:
                    # 打印这一个格子上的全部 24 条样本（排好序）与真值，供人眼核对。
                    st = PEEK[1]
                    v = np.sort(sm[:, st])
                    lo, hi_ = v[0], v[-1]
                    q = np.quantile(v, [0.10, 0.50, 0.90])
                    cover = "落在范围里" if lo <= truth[st] <= hi_ else "**跑到范围外了**"
                    print(f"[peek] 窗口{PEEK[0]} 第{st}号站 H={j + 1}：", flush=True)
                    print("       24 条样本 = " + " ".join(f"{x:.2f}" for x in v), flush=True)
                    print(f"       真值 = {truth[st]:.2f}   样本范围 [{lo:.2f}, {hi_:.2f}]  → 真值{cover}",
                          flush=True)
                    print(f"       样本的 10%/50%/90% 分位点 = {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}",
                          flush=True)
                if SPREAD_SWEEP:
                    for kk in SPREAD_SWEEP:
                        smk = swv[float(kk)][:, k, :, j]
                        acc = sw[float(kk)][j]
                        acc['per_win'][c0 + k, mk] = pb.crps_fair(smk[:, mk], truth[mk])
                        acc['cell_cnt'][mk] += 1
                        acc['cs_sum'] += float(pb.crps_fair(smk[:, mk].sum(1), truth[mk].sum()))
                        for kp, v in pb.pit_metrics(smk[:, mk], truth[mk]).items():
                            if kp.startswith('picp'):
                                acc['picp'][kp] = acc['picp'].get(kp, 0.0) + v
        del smp, ysim
    for j in js:
        n_sc = max(1, res[j]['n_scored'])
        for d in (res[j]['picp'], res[j]['mpiw']):
            for kk in d:
                d[kk] /= n_sc
        # 扫描那几档聚成 [N]（与主口径同一条公式，只是换了尺度），标量留给 run_city 收尾。
        res[j]['sweep'] = {}
        for kk in SPREAD_SWEEP:
            a = sw[float(kk)][j]
            res[j]['sweep'][float(kk)] = dict(per_st=np.nansum(a['per_win'], 0),
                                              cell_cnt=np.maximum(a['cell_cnt'], 1),
                                              cs_sum=a['cs_sum'],
                                              picp={k2: v for k2, v in a['picp'].items()})
    return res


def run_city(city, args, device, js, u, qs):
    ds, raw, starts = load_city(city)
    N = raw.shape[1]
    y0, t0 = pb.load(city)
    ok, trm, vam, tem, _, span = pb.split_rows(y0, t0)
    cols, smean = pb.live_cols(y0, trm)
    if len(cols) != N:
        raise AssertionError(f"{city} 站数不一致：Kairos {N} vs 基线 {len(cols)}")
    hod, wkd = t0.hour.values, (t0.dayofweek.values >= 5).astype(np.int8)
    y = y0[:, cols]
    # 站×窗级剔除：全城统一；所有方法都用这一张表。
    # 两套口径：--clean = 新口径（整段 144h 无合成直线，直接从 raw 算）；
    #           --drop_straight = 旧口径（CSV 里目标块为 A，仅用于复现历史结果）。
    if getattr(args, 'clean', False) and args.drop_straight:
        raise SystemExit("--clean 与 --drop_straight 是两套口径，只能选一个")
    if getattr(args, 'clean', False):
        keep = clean_keep(raw, starts['te'])
        tag = '--clean'
    elif args.drop_straight:
        keep = straight_keep(city, starts['te'], cols, N)
        tag = '--drop_straight'
    else:
        keep, tag = None, None
    if keep is not None:
        print(f"[{city}] {tag}: 剔除 {100 * (~keep).mean():.1f}% 的站×窗格子，"
              f"保留 {(keep.sum(0) > 0).sum()}/{N} 个站", flush=True)

    # 站序对齐自检：Kairos 侧从 raw 读的真值，必须与基线 y 的同一批下标逐值相等
    for nm in ('va', 'te'):
        for j in js:
            a = obs_at(raw, starts[nm], j)
            b = y[starts[nm] + SEQ_L + j]
            if not np.allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True):
                raise AssertionError(f"{city} {nm} H={j+1} 站序/取值与基线不一致")

    print(f"[{city}] 站 {N}  验证窗口 {len(starts['va'])}  测试窗口 {len(starts['te'])}",
          flush=True)
    rows = {}

    # ---- 点预测：Kairos-plain 与 PatchTST 各跑一遍残差自助 ----
    providers = {}
    km = build_kairos(city, 'plain', args.out_root, device, args.n_quant)
    core, pred72 = None, None
    if km is not None:
        providers['kairos'] = {nm: point_pass(km, raw, starts[nm], js, device)
                               for nm in ('va', 'te')}
        # CSDI 的条件是**整块 72 步**的点预测（一次生成一整个 [站×72] 场景），
        # 所以要用主干多跑一遍全 horizon。只留 [M,N,72] 这一份。
        impl = getattr(args, 'csdi_impl', 'csdi')
        core = None if args.skip_csdi else (
            load_zidiff(city, device) if impl == 'zidiff' else load_csdi(city, device))
        if core is not None:
            P72 = point_pass(km, raw, starts['te'], list(range(SEQ_L)), device)
            pred72 = np.stack([P72[h] for h in range(SEQ_L)], axis=2)
            del P72
        del km
        torch.cuda.empty_cache()
        # 与训练脚本落盘的 predict_*.npy 对账：两边是**独立**的取窗+推理路径，
        # 对得上才说明评测用的窗口与训练时是同一批（不写盘，只比）。
        pnpy = os.path.join(args.out_root, city, f"predict_plain_pre{SEQ_L}_fold{FOLD}.npy")
        if os.path.exists(pnpy):
            saved = np.load(pnpy)
            want = (len(starts['te']) * N, SEQ_L)
            if saved.shape == want:
                saved = saved.reshape(len(starts['te']), N, SEQ_L)
                for j in js:
                    d = np.abs(saved[:, :, j] - providers['kairos']['te'][j]).max()
                    # 这条是"eval 路径 == 训练路径"的回归测试：两边都是同一份权重、
                    # 冻结的 Gate bias，差别只该是浮点尾差。放 1e-3 留余量；一旦窗口
                    # 起点排布或站序错了，这里会飙到 1e+1 量级。
                    flag = "✓" if d < 1e-3 else "✗ 窗口/站序没对齐？"
                    print(f"[align] {city} H={j+1} 重推理 vs 训练落盘 max|Δ|={d:.2e} {flag}",
                          flush=True)
            else:
                print(f"[warn] {city} 落盘 predict 形状 {saved.shape} != {want}", flush=True)

    # PatchTST 参照：复用 prob_cond_probe 的同一套 loader/推理，只换点预测来源
    try:
        ds_m, model_m, _ = pa.build_model(city, args.ckpt_dir, device)
        va_ds, te_ds = ds_m.valid_loader.dataset, ds_m.test_loader.dataset
        if len(te_ds.starts) != len(starts['te']) or len(va_ds.starts) != len(starts['va']):
            raise AssertionError(f"PatchTST 起点数 {len(va_ds.starts)}/{len(te_ds.starts)} "
                                 f"!= {len(starts['va'])}/{len(starts['te'])}")
        pv = pa.inverse_3d(ds_m.scaler, pa.predict(model_m, va_ds, device))
        pt = pa.inverse_3d(ds_m.scaler, pa.predict(model_m, te_ds, device))
        providers['patch'] = {'va': {j: pv[:, :, j] for j in js},
                              'te': {j: pt[:, :, j] for j in js}}
        del model_m
        torch.cuda.empty_cache()
    except Exception as e:                                   # noqa: BLE001
        print(f"[warn] {city} PatchTST 参照不可用：{e}", flush=True)

    rng = np.random.default_rng(0)
    for tag, pr in providers.items():
        r = bootstrap_variants({'va': pr['va'], 'te': pr['te']},
                               y, starts, ok, hod, wkd, js, smean, N, rng,
                               keep_te=keep)
        for name, per_j in r.items():
            for j, d in per_j.items():
                rows[f"{tag}_{name}_H{j+1}"] = d

    # ---- 做法③：分位头 ----
    qm = build_kairos(city, 'quant', args.out_root, device, args.n_quant)
    has_quant = qm is not None          # qm 用完就 del 了（92MB 权重），后面还要判断有没有
    # 经验 copula 的"依赖供体"：**验证期**残差场，整条向量。验证段排在测试段之前，
    # 天然因果；做法① 的 honest/roll28 池也取自验证段，两边同源，比较才公平。
    # 用测试期残差当供体就是拿评测数据选耦合 = 泄漏，这里不碰。
    cop_pool = None
    if has_quant and 'kairos' in providers:
        cop_pool = {}
        for j in js:
            lag = pb.LAG * int(np.ceil((j + 1) / pb.LAG))
            s = np.asarray(starts['va']) + SEQ_L + j
            m = ok[s - lag]
            cop_pool[j] = y[s[m]] - providers['kairos']['va'][j][m]
    if has_quant:
        qr = quant_pass(qm, raw, starts['te'], js, qs, u, device, cop_pool, keep=keep)
        del qm
        torch.cuda.empty_cache()
        for j in js:
            # 聚合口径必须与 bootstrap_variants 逐字一致：那边是 [N] 的 per_st
            # 跨窗口累加后 mean(per_st / P)；这边 per_win 是 [M,N] 逐窗口原值，
            # 所以先按窗口轴累加回 [N]，再套同一个公式。两边都除以窗口数**恰好一次**。
            P = max(1, qr[j]['n_scored'])
            q_per_st = np.nansum(qr[j]['per_win'], 0)              # [N]
            # 每站除自己的计分窗口数（剔除后逐站不同）；不剔除时与旧的 P 完全一致。
            q_cnt = np.maximum(np.sum(np.isfinite(qr[j]['per_win']), 0), 1)
            # 量纲自检：CRPS 与"平均绝对误差"同量级，不可能远小于 50% 区间宽度。
            # 上面那个缩小 2125 倍的 bug 会让 raw_crps 比 MPIW50 小三个数量级 —— 放两个
            # 数量级的余量，正常情况绝不会触发。
            raw_crps = float(np.mean(q_per_st / q_cnt))
            mpiw50 = qr[j]['mpiw'].get('mpiw50', 0.0)
            if mpiw50 > 0 and raw_crps < mpiw50 / 100:
                raise AssertionError(
                    f"{city} H={j+1} quant 的原始空间 CRPS({raw_crps:.3g}) 比 MPIW50"
                    f"({mpiw50:.3g}) 小两个数量级以上，聚合口径有问题（曾因此少除/多除窗口数）")
            rows[f"quant_H{j+1}"] = dict(
                crps_norm_sum=float(np.mean(q_per_st / q_cnt) / np.nanmean(smean)),
                crps_sum_norm=float(qr[j]['cs_sum'] / P / np.nansum(smean)),
                crps_sum_norm_comono=float(qr[j]['cs_sum_comono'] / P / np.nansum(smean)),
                crps_sum_norm_copula=float(qr[j]['cs_sum_cop'] / P / np.nansum(smean)),
                crps_nz=float(qr[j]['nz_sum'] / max(1, qr[j]['nz_win']) / np.nanmean(smean)),
                nz_rate=float(qr[j]['nz_cells'] / max(1, qr[j]['nz_den'])),
                **{k: v for k, v in qr[j]['picp'].items()},
                **{k: v for k, v in qr[j]['mpiw'].items()})

    # ---- 整块扩散（CSDI）：唯一一个站间关系由模型自己负责的方法 ----
    cr = None
    if pred72 is not None:
        hod72 = np.stack([ds.time[starts['te'] + SEQ_L + h].hour.values
                          for h in range(SEQ_L)], axis=1).astype(np.float32)
        ytrue72 = np.transpose(y[starts['te'][:, None] + SEQ_L
                                  + np.arange(SEQ_L)[None, :]], (0, 2, 1))
        # 分类器要的那两路活动度取自**预测原点之前 72h 的真实观测**（即主干看的
        # 那段历史本身），训练/评测必须同一份，否则评测时该通道恒 0、分类器退化成常数。
        hist_on = np.transpose((y[starts['te'][:, None]
                                  + np.arange(SEQ_L)[None, :]] > 0), (0, 2, 1))
        t0 = time.time()
        if impl == 'zidiff':
            # 吸收态扩散没有独立门卫，所以**没有** oracle 这个作弊口径。
            cr = zidiff_pass(core, pred72, hod72, ytrue72, hist_on, device, js, keep=keep,
                             eta=getattr(args, 'csdi_eta', 0.0))
        else:
            cr = csdi_pass(core, pred72, hod72, ytrue72, hist_on, device, js,
                           mask_mode=getattr(args, 'csdi_mask', 'clf'), keep=keep,
                           eta=getattr(args, 'csdi_eta', 0.0))
        for j in js:
            P = max(1, cr[j]['n_scored'])
            cper = np.nansum(cr[j]['per_win'], 0)
            ccnt = np.maximum(np.sum(np.isfinite(cr[j]['per_win']), 0), 1)
            rows[f"csdi_H{j+1}"] = dict(
                crps_norm_sum=float(np.mean(cper / ccnt) / np.nanmean(smean)),
                crps_nz=float(cr[j]['nz_sum'] / max(1, cr[j]['nz_win']) / np.nanmean(smean)),
                nz_rate=float(cr[j]['nz_cells'] / max(1, cr[j]['nz_den'])),
                # 没有 comono / copula 两列：整块采样根本没有那个自由度，
                # 这里列出来的就是模型自己拼出来的联合，不用人为补规则。
                crps_sum_norm=float(cr[j]['cs_sum'] / P / np.nansum(smean)),
                **{k: v for k, v in cr[j]['picp'].items()},
                **{k: v for k, v in cr[j]['mpiw'].items()})
        print(f"[{city}] CSDI 整块采样完成（{len(starts['te'])} 窗 × "
              f"{CSDI_NSAMPLE} 条 × DDIM {CSDI_STRIDE}×跳步 η={args.csdi_eta:g}），"
              f"{time.time() - t0:.0f}s", flush=True)
        if SPREAD_SWEEP:
            # 离散度扫描：**只打印，不进 rows/metrics json**，所以污染不了正式汇总表。
            print(f"[sweep] {city} 离散度扫描（残差整体 ×k；k=1 即上表现状）", flush=True)
            for j in js:
                parts = []
                for kk in SPREAD_SWEEP:
                    a = cr[j]['sweep'][float(kk)]
                    c1 = float(np.mean(a['per_st'] / a['cell_cnt']) / np.nanmean(smean))
                    c2 = float(a['cs_sum'] / P / np.nansum(smean))
                    # picp 累加器里存的是**逐窗比例的和**（与 res[j]['picp'] 一样，只是在
                    # 这里没做归一化），所以打印前必须除以窗口数 P —— 忘了除会打出
                    # 0.64×2125≈1361 这种数（第一次跑就踩了）。
                    p50 = float(a['picp'].get('picp50', np.nan)) / P
                    p80 = float(a['picp'].get('picp80', np.nan)) / P
                    p90 = float(a['picp'].get('picp90', np.nan)) / P
                    parts.append(f"k={kk:g}: crpsΣ {c1:.4f} Σsum {c2:.4f} "
                                 f"PICP {p50:.2f}/{p80:.2f}/{p90:.2f}")
                print(f"  H={j+1}  " + " | ".join(parts), flush=True)

    # ---- 点预测自身的精度（对照 PatchTST 的逐 horizon 指标）----
    point = {}
    preds = {tag: providers[tag]['te'] for tag in ('kairos', 'patch') if tag in providers}
    if has_quant:
        preds['quant'] = {j: qr[j]['med'] for j in js}      # 分位头取 τ=0.5 当点预测
    if cr is not None:
        preds['csdi'] = {j: cr[j]['med'] for j in js}       # 扩散样本的中位数
    for tag, prd in preds.items():
        for j in js:
            pr = prd[j]
            ob = y[starts['te'] + SEQ_L + j]
            if keep is not None:              # 点指标也按同一批格子算
                pr, ob = pr[keep], ob[keep]
            e = pr - ob
            point[f"{tag}_H{j+1}"] = dict(
                R2=float(1 - (e ** 2).sum() / ((ob - ob.mean()) ** 2).sum()),
                MAE=float(np.abs(e).mean()))
    return {'city': city, 'n_station': N, 'span': span, 'rows': rows, 'point': point}


def report_city(r, js, args):
    L, P = [], None
    P = L.append
    P("=" * 120)
    P(f"Kairos 概率评测  {r['city']}   站 {r['n_station']}   （口径同 prob_cond_probe，同一个 crpsΣ）")
    P("  kairos_* = 冻结 Kairos 主干；quant = 分位头(做法③)，其余 = 点预测 + 残差自助(做法①)")
    P("  csdi     = 整块 [站×72] 扩散 + on/off 分类（做法④）；站间关系由模型自己学，无需人为耦合")
    P("  patch_*  = 同一个池、换成 PatchTST 点预测（参照）")
    P("=" * 120)
    for j in js:
        H = j + 1
        P(f"\n--- H = {H}（提前量 {H} 小时）---")
        P(f"{'口径':<22}{'crpsΣsum':>10}{'crpsΣ':>9}{'池中位':>9}   说明")
        P("-" * 128)
        note = {
            'kairos_honest': 'Kairos点 + 验证段池·按小时',
            'kairos_roll28': 'Kairos点 + 前28天池·按小时（因果）',
            'kairos_oracleWA': 'Kairos点 + 测试期池·(小时,周末,幅度)（作弊）',
            'quant': '分位头 · 独立耦合（做法③，保守端）',
            'quant_copula': '分位头 · 经验耦合（照验证期残差的跨站排序重排）',
            'quant_comono': '分位头 · 同单调耦合（敏感性上界，分位数同上一行）',
            'csdi': '整块扩散 + on/off（做法④，联合由模型自己拼）',
            'patch_roll28': 'PatchTST点 + 前28天池·按小时',
            'patch_oracleWA': 'PatchTST点 + 测试期池·(小时,周末,幅度)（作弊）',
        }
        if getattr(args, 'csdi_impl', 'csdi') == 'zidiff':
            note['csdi'] = '吸收态扩散（生死与数值同一个网络自学，无独立门卫）'
        for tag in ('kairos_honest', 'kairos_roll28', 'kairos_oracleWA',
                    'quant', 'quant_copula', 'quant_comono', 'csdi',
                    'patch_roll28', 'patch_oracleWA'):
            # 三种耦合共用同一行数据（边际完全相同），只是 crpsΣsum 取不同的列。
            k = f"{'quant' if tag in ('quant_copula', 'quant_comono') else tag}_H{H}"
            if k not in r['rows']:
                continue
            d = r['rows'][k]
            extra = ""
            if 'picp50' in d and tag in ('quant', 'csdi'):
                extra = (f"  PICP@50/80/90 {d['picp50']:.2f}/{d['picp80']:.2f}/{d['picp90']:.2f}"
                         f"  MPIW {d['mpiw50']:.2f}/{d['mpiw80']:.2f}/{d['mpiw90']:.2f}")
            if tag in ('quant_copula', 'quant_comono'):
                extra = "  （PICP/MPIW 同上一行，不再重复）"
            cs = d.get({'quant_copula': 'crps_sum_norm_copula',
                        'quant_comono': 'crps_sum_norm_comono'}.get(tag, 'crps_sum_norm'),
                       float('nan'))
            # 分位头和整块扩散都没有"残差池"这个东西（它们直接把分布参数化/生成出来），
            # 这一列打 '-'。
            n_pool = ('-' if tag.startswith('quant') or tag == 'csdi'
                      else str(d.get('n_pool_med', 0)))
            P(f"{tag:<22}{cs:>10.4f}{d['crps_norm_sum']:>9.4f}{n_pool:>9}"
              f"   {note[tag]}{extra}")
        if f'kairos_roll28_H{H}' in r['rows'] and f'quant_H{H}' in r['rows']:
            a = r['rows'][f'kairos_roll28_H{H}']
            b = r['rows'][f'quant_H{H}']
            for key, nm in (('crps_sum_norm', 'crpsΣsum(独立耦合·保守端)'),
                            ('crps_sum_norm_copula', 'crpsΣsum(经验耦合·有数据依据)'),
                            ('crps_sum_norm_comono', 'crpsΣsum(同单调·敏感性上界)'),
                            ('crps_norm_sum', 'crpsΣ(边际，与耦合无关)')):
                if key not in a or key not in b:
                    continue        # 自助法抽的是整条残差向量，没有"耦合"这个自由度
                u, v = a[key], b[key]
                P(f"* 做法③ 相对同主干的做法①（kairos_roll28）{nm}: {(v - u) / u:+.1%}"
                  f"（{'更差' if v > u else '更好'}）")
            # 把三个耦合点摆出来。分位头的 crpsΣsum 不是一个数，是随耦合假设移动的一条线；
            # 只有**经验耦合**那点有数据依据（照验证期残差场的跨站排序重排），另外两点是
            # 人造端。判据取经验耦合，独立那端只当保守下界看。
            pts = {nm: b[k] for k, nm in (('crps_sum_norm', '独立'),
                                          ('crps_sum_norm_copula', '经验'),
                                          ('crps_sum_norm_comono', '同单调'))
                   if k in b and np.isfinite(b[k])}
            if pts:
                lo, hi = min(pts.values()), max(pts.values())
                P("  分位头三个耦合点：" + "、".join(f"{n} {v:.4f}" for n, v in pts.items())
                  + f"；做法①（roll28）= {a['crps_sum_norm']:.4f}")
                if '经验' in pts:
                    d = pts['经验']
                    P(f"  → 按**有数据依据**的经验耦合计："
                      f"{'做法③更好' if d < a['crps_sum_norm'] else '做法①更好'}"
                      f"（{d:.4f} vs {a['crps_sum_norm']:.4f}，"
                      f"{(d - a['crps_sum_norm']) / a['crps_sum_norm']:+.1%}）。"
                      f"独立耦合那端偏保守，不作判据。")
                if lo <= a['crps_sum_norm'] <= hi:
                    P(f"  ! 提醒：做法① 的值落在三个耦合点构成的区间 [{lo:.4f}, {hi:.4f}] 内部，"
                      f"所以 crpsΣsum 上两者的**量级差**主要由耦合假设决定；"
                      f"结论应以经验耦合那一行为准，边际口径（crpsΣ）作为无此自由度的对照。")
        # 做法④（整块扩散）单独比：它**没有**耦合自由度，所以 crpsΣsum 这一列
        # 是可以直接跟做法① 逐字比的 —— 这正是把 CRPS_sum 降级时留的那个边界条件。
        if 'csdi_H%d' % H in r['rows'] and f'kairos_roll28_H{H}' in r['rows']:
            a = r['rows'][f'kairos_roll28_H{H}']
            c = r['rows'][f'csdi_H{H}']
            for key, nm in (('crps_norm_sum', 'crpsΣ(边际)'),
                            ('crps_sum_norm', 'crpsΣsum(聚合，整块采样无耦合自由度)')):
                if key in a and key in c:
                    P(f"* 做法④ 相对做法①（kairos_roll28）{nm}: "
                      f"{(c[key] - a[key]) / a[key]:+.1%}"
                      f"（{'更好' if c[key] < a[key] else '更差'}）"
                      f"  {a[key]:.4f} → {c[key]:.4f}")
    # ---- 条件 CRPS：把所有方法的真零格一并剔掉，只留"非零格上谁画得准"----
    have_nz = [j + 1 for j in js
               if np.isfinite(r['rows'].get(f'kairos_roll28_H{j+1}', {}).get('crps_nz', np.nan))]
    if have_nz:
        P("\n--- 条件 CRPS（**只计真值 > 0 的格子**，零格对每个方法一视同仁地剔掉）---")
        P(f"{'':<22}" + "".join(f"{'H=' + str(H):>10}" for H in have_nz))
        for tag, nm in (('kairos_roll28', '做法① 点+残差池'),
                        ('quant', '做法③ 分位头'),
                        ('csdi', '做法④ 整块扩散')):
            vals = [r['rows'].get(f'{tag}_H{H}', {}).get('crps_nz', np.nan) for H in have_nz]
            if not any(np.isfinite(v) for v in vals):
                continue
            P(f"{nm:<22}" + "".join(f"{v:>10.4f}" if np.isfinite(v) else f"{'-':>10}"
                                    for v in vals))
        for H in have_nz:
            nr = r['rows'].get(f'quant_H{H}', {}).get('nz_rate', np.nan)
            if np.isfinite(nr):
                P(f"  非零格占比 H={H}：{nr:.3f}（其余为真零格，已剔除）")
        P("  为什么单列这一列：满分门卫（--csdi_mask oracle）下 csdi_pass 里 `ysim *= m` 把真零格")
        P("    直接写成 0，那些格子对做法④ 是**白送**的满分。所以「满分门卫下 ④ 边际第一」里混着")
        P("    「零格白送」这一笔。剔掉之后剩下的才是「画师在非零格上画得像不像」。")
        P("    若 ④ 在这一列仍然输 → 瓶颈不在门卫，换扩散方式也救不回来。")
    if r['point']:
        P("\n点预测精度（同一批测试窗口、同一批真值）：")
        P("  quant 行指的是分位头自己的 τ=0.5 —— 与 kairos 行同主干、只换头，可直接比。")
        P(f"{'':<10}" + "".join(f"{'H=' + str(j + 1):>16}" for j in js))
        for tag in ('kairos', 'quant', 'csdi', 'patch'):
            if f'{tag}_H{js[0] + 1}' not in r['point']:
                continue
            P(f"{tag:<10}" + "".join(
                f"{'R² %.4f' % r['point'][f'{tag}_H{j+1}']['R2']:>16}"
                f"{'MAE %.3f' % r['point'][f'{tag}_H{j+1}']['MAE']:>16}".replace('MAE', ' MAE')
                for j in js))
    P("\n读法：")
    P("  * 指标优先级按约定的来：**crpsΣsum > crpsΣ > PICP > MPIW**。")
    P("    crpsΣsum = Σ_时刻 CRPS(Σ_站 样本, Σ_站 真值) / Σ_站 smean，是「先加总再算」的联合口径。")
    P("    crpsΣ    = 逐站 CRPS 按站点量级加权后平均，是边际口径。")
    P("  * quant 与 kairos_roll28 是**同一根冻结主干、同一批起点**的对照——只差一个头。")
    P("    所以两者的差就是「直接训分布」相对「点预测外挂残差池」的净收益。")
    P("  * 分位头只参数化**边际**（每站自己一套分位数）。要把它变成「一整天的场景」必须额外")
    P("    规定站与站之间怎么捆，而这一步没有数据依据——所以同一批分位数能拼出三个点：")
    P("    quant 让每站各自乱序取点（独立，Σ_站 方差最小）；quant_comono 用同一列分位点铺在")
    P("    所有站上（完全同单调，跨站波动绑死，Σ_站 方差最大）；quant_copula 照**验证期残差**")
    P("    的跨站排序模式重排（经验 copula，唯一有数据依据的那一点）。")
    P("    残差自助没这个自由度——它抽的是**整条残差向量**，站间相关是数据里原样来的。")
    P("  * **判据取经验耦合（quant_copula）。** 实测（script/prob_resid_corr.py，六城）：")
    P("    站间两两相关很弱（ρ̄ 0.018~0.146），但全城求和后波动被放大到独立假设的 2.2~202 倍")
    P("    ——「站数多」会把弱相关累加成强聚合效应。所以独立耦合不成立；它与真实结构的")
    P("    crpsΣsum 差 1~16%，**方向是独立更差**，即独立那行对分位头偏保守、不是偏袒。")
    P("    同单调那行纯粹是敏感性上界，**不要拿它跟自助法比**。")
    P("  * kairos_oracleWA 仍是作弊（偷看测试期残差），只用来标出「还剩多少」。")
    P("  * PICP@50/80/90 应贴近 0.50/0.80/0.90；MPIW 是原始量纲（只在同城内可比）。")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cities', nargs='*', default=pb.CITIES)
    ap.add_argument('--out_root', default=f"{BASE}/result/kairos_l1",
                    help='Kairos 权重根目录，实际读 <out_root>/<city>/<variant>_pre72_fold6.pth')
    ap.add_argument('--ckpt_dir', default=f"{BASE}/result/univariate",
                    help='PatchTST 参照权重根目录（与 prob_cond_probe 同一套）')
    ap.add_argument('--n_quant', type=int, default=99)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--rerender', action='store_true',
                    help='不跑推理，直接读已存好的 metrics_<city>.json 重出报告'
                         '（只改排版时用，全量推理要几十分钟）')
    ap.add_argument('--skip_csdi', action='store_true',
                    help='不跑整块扩散（它没有预存结果，跳过能省一大截时间）')
    ap.add_argument('--csdi_root', default=None,
                    help='整块扩散权重目录；清理后重训的结果应指向 result/csdi_clean'
                         '（默认 result/csdi）')
    ap.add_argument('--csdi_impl', default='csdi', choices=['csdi', 'zidiff'],
                    help='整块扩散用哪个模型：csdi = 做法④（独立 on/off 分类器切硬掩码）；'
                         'zidiff = 吸收态扩散（掩码与数值同一个网络联合建模，掩码沿采样链'
                         '反复重抽，没有独立门卫、也没有 oracle 口径）。')
    ap.add_argument('--suffix', default='',
                    help='实验变体后缀，追加到输出文件名（绝不覆盖已有正式结果）。')
    ap.add_argument('--csdi_eta', type=float, default=0.0,
                    help='DDIM 随机性系数。0=确定性（旧默认，会系统性欠离散、PICP 偏低）；'
                         '1=在跳步处注入与被跳过 DDPM 链等方差的噪声，评测分散度时必须用 1。')
    ap.add_argument('--csdi_mask', default='clf', choices=['clf', 'oracle'],
                    help="做法④ 的 on/off 掩码口径。oracle = 拿真值掩码，**泄漏真值**，"
                         "只用来量『扩散本身还剩多少、差距有多少来自分类器』，"
                         "出来的数不能进正式对比表。")
    ap.add_argument('--spread_sweep', nargs='*', type=float, default=[],
                    help='离散度扫描：把扩散残差整体乘这些 k（同一次采样复用，零额外成本），'
                         '只打印、不进 metrics json。用来归因「欠离散是尺度问题还是形状问题」，'
                         '见 SPREAD_SWEEP 常量的说明。')
    ap.add_argument('--clean', action='store_true',
                    help='新口径：逐站剔除「整段 72h 历史 + 72h 目标含合成直线」的站×窗'
                         '格子（与训练脚本 straight_window_keep 同一实现）。'
                         '所有方法用同一张掩码，结果落盘到 _clean 后缀。')
    ap.add_argument('--drop_straight', action='store_true',
                    help='旧口径（仅复现历史结果）：剔除 CSV 里目标块为纯直线的站×窗格子。')
    ap.add_argument('--peek', nargs=2, type=int, default=None, metavar=('窗口号', '站号'),
                    help='只打印：在指定测试窗口、指定站上，做法④ 生成的 24 条样本与真值，'
                         '让人肉眼看一次"样本长什么样"。不影响任何指标。')
    args = ap.parse_args()

    global SPREAD_SWEEP, PEEK, CSDI_ROOT
    SPREAD_SWEEP = list(args.spread_sweep)
    PEEK = tuple(args.peek) if args.peek else None
    if args.csdi_root:
        CSDI_ROOT = args.csdi_root

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    js = [h - 1 for h in pb.HORIZONS]
    qs = np.arange(1, args.n_quant + 1) / (args.n_quant + 1)
    u = (np.arange(M_SAMPLE) + 0.5) / M_SAMPLE

    # oracle 掩码泄漏真值，**必须与正式报告分开落盘**，不能覆盖 report_<city>.txt。
    sfx = "" if args.csdi_mask == 'clf' else f"_{args.csdi_mask}"
    # 吸收态扩散是**另一个模型**，产物单独落盘，绝不覆盖做法④ 的正式结果。
    if getattr(args, 'csdi_impl', 'csdi') == 'zidiff':
        sfx = "_zidiff"
    # 剔除纯直线是**另一个口径**，单独落盘，不覆盖原结果。
    if getattr(args, 'drop_straight', False):
        sfx += "_dropA"
    if getattr(args, 'clean', False):
        sfx += "_clean"
    # 采样随机性系数是另一个口径（η=0 欠离散），单独落盘，不覆盖 η=0 的结果。
    if getattr(args, 'csdi_eta', 0.0):
        sfx += f"_eta{args.csdi_eta:g}"
    # 任意实验变体都可以带一个后缀，绝不覆盖已有正式结果。
    if getattr(args, 'suffix', ''):
        sfx += f"_{args.suffix}"

    done = []
    for c in args.cities:
        if args.rerender:
            p = f"{OUT}/metrics_{c}{sfx}.json"
            if not os.path.exists(p):
                print(f"[skip] {c} 没有 {p}")
                continue
            r = json.load(open(p))
        else:
            r = run_city(c, args, device, js, u, qs)
            json.dump(r, open(f"{OUT}/metrics_{c}{sfx}.json", "w"), ensure_ascii=False,
                      indent=1, default=str)
        open(f"{OUT}/report_{c}{sfx}.txt", "w").write(report_city(r, js, args) + "\n")
        done.append(r)

    if len(done) < 2:
        print(f"\n单城模式：已保存 {OUT}/report_{done[0]['city']}{sfx}.txt（不写汇总）")
        return

    L = ["=" * 120,
         "Kairos 概率评测汇总（5 城平均，不含 AMS；AMS 单列）",
         "  判据：crpsΣ（单站口径，无耦合自由度）→ 然后 PICP > MPIW",
         "  crpsΣsum 作为**聚合口径对照**并列报：它有「跨站怎么拼」这个人为自由度，",
         "  实测真实跨站关联只值 ~1.5%、拼法选错能值 20~34%（信噪比 1:15），故不单独作判据。",
         "=" * 120]
    # 三项一组 (tag, 显示名, 取哪一列)；第三项 None = 用本块的主键。
    # 自助法没有"耦合"这个自由度，所以整组都是 None。
    BOOT = (('kairos_roll28', '做法① Kairos点+因果池', None),
            ('kairos_oracleWA', '作弊上限 Kairos点', None),
            ('patch_roll28', '参照 PatchTST点+因果池', None),
            ('patch_oracleWA', '参照 PatchTST点作弊上限', None))
    # 分位头只参数化边际，"拼成跨站场景"这一步没有数据依据，所以三个耦合点都列：
    # 独立（和最小，保守端）/ 经验（照验证期残差排序重排，唯一有数据依据）/ 同单调（上界）。
    for key, nm, tags in (
            ('crps_norm_sum', 'crpsΣ（主判据：单站口径，无耦合自由度）',
             (BOOT[0], ('quant', '做法③ 分位头', None),
              ('csdi', '做法④ 整块扩散', None), BOOT[1], BOOT[2], BOOT[3])),
            ('crps_sum_norm', 'crpsΣsum（聚合口径 · 对照，不作判据）',
             (BOOT[0], ('quant', '做法③ 分位头·独立(保守端)', None),
              ('quant', '做法③ 分位头·经验耦合(判据)', 'crps_sum_norm_copula'),
              ('quant', '做法③ 分位头·同单调(上界)', 'crps_sum_norm_comono'),
              # 做法④ 整块采样**没有**耦合自由度（一次采样就是完整场景），所以三行合一，
              # 与做法① 同处一个口径，可直接对撞。
              ('csdi', '做法④ 整块扩散(无耦合自由度)', None),
              BOOT[1], BOOT[2], BOOT[3]))):
        L.append(f"\n--- {nm} ---")
        for tag, name, kk in tags:
            k2 = kk or key
            line = f"{name:<26}"
            for H in pb.HORIZONS:
                vals = [r['rows'][f'{tag}_H{H}'][k2] for r in done
                        if r['city'] != 'AMS' and k2 in r['rows'].get(f'{tag}_H{H}', {})]
                line += f"{np.mean(vals):>10.4f}" if vals else f"{'-':>10}"
            L.append(line)
        L.append(f"{'':<26}" + "".join(f"{'H=' + str(H):>10}" for H in pb.HORIZONS))
    # 条件 CRPS：剔掉真零格后再比。**这是"画师行不行"的判据**，不是边际 CRPS 的替代 ——
    # 满分门卫下 `ysim *= m` 把真零格白送给做法④，边际那一列因此被混淆（见 nz_crps）。
    L.append("\n--- 条件 CRPS（**只计真值 > 0 的格子**；5 城平均，不含 AMS）---")
    L.append("    零格对所有方法一视同仁地剔掉，剩下的才是'非零格上谁画得准'。")
    L.append(f"{'':<26}" + "".join(f"{'H=' + str(H):>10}" for H in pb.HORIZONS))
    for tag, name in (('kairos_roll28', '做法① Kairos点+因果池'),
                      ('quant', '做法③ 分位头'),
                      # 标签跟着 --csdi_impl 走：metrics json 里 csdi_H* 装的就是那一次跑的东西，
                      # 写死成"④/⑤"会让人误以为两套模型都进了这张表。
                      ('csdi', '⑤ 吸收态扩散' if getattr(args, 'csdi_impl', 'csdi') == 'zidiff'
                       else '做法④ 整块扩散')):
        line = f"{name:<26}"
        for H in pb.HORIZONS:
            vals = [v for v in (r['rows'].get(f'{tag}_H{H}', {}).get('crps_nz', np.nan)
                                for r in done if r['city'] != 'AMS')
                    if np.isfinite(v)]
            line += f"{np.mean(vals):>10.4f}" if vals else f"{'-':>10}"
        L.append(line)
    L.append("\n--- 点预测 R²（5 城平均，不含 AMS；quant 行 = 分位头的 τ=0.5）---")
    L.append(f"{'':<26}" + "".join(f"{'H=' + str(H):>10}" for H in pb.HORIZONS))
    for tag, name in (('kairos', 'Kairos-L1 点预测头'),
                      ('quant', 'Kairos-L1 分位头中位数'),
                      ('patch', 'PatchTST（针对本任务训练）')):
        line = f"{name:<26}"
        for H in pb.HORIZONS:
            vals = [r['point'][f'{tag}_H{H}']['R2'] for r in done
                    if r['city'] != 'AMS' and f'{tag}_H{H}' in r['point']]
            line += f"{np.mean(vals):>10.4f}" if vals else f"{'-':>10}"
        L.append(line)
    L.append("\n  * 分位头的三行是**同一批分位数**拼出来的三个耦合点，边际指标逐值相同；")
    L.append("    做法①的两行（roll28 / oracleWA）没有这个自由度，因为它抽的是整条残差向量。")
    L.append("    判据取**经验耦合**那一行（照验证期残差的跨站排序重排，唯一有数据依据）；")
    L.append("    独立那行偏保守（实测：实测耦合比独立耦合的 crpsΣsum 低 1~16%），")
    L.append("    同单调是敏感性上界，比 crpsΣsum 时不要拿它当分位头的成绩。")
    L.append("  * 各城训练脚本的 shuffle 没有设种子，同一个配置重跑有 ~0.01 R² 量级的波动；")
    L.append("    比这个量级小的差异不要当结论读（同一批权重、同一批起点的横向对比不受影响）。")
    for r in done:
        if r['city'] == 'AMS':
            L.append("\nAMS（单列，不进汇总；停摆 901h + 测试期 95.9% 站冻结，不可与他人同比）：")
            for H in pb.HORIZONS:
                a = r['rows'].get(f'kairos_roll28_H{H}', {})
                b = r['rows'].get(f'quant_H{H}', {})
                d4 = r['rows'].get(f'csdi_H{H}', {})
                if a and b:
                    L.append(f"  H={H:<3} crpsΣsum 做法① {a['crps_sum_norm']:.4f} -> "
                             f"做法③ 独立 {b['crps_sum_norm']:.4f} "
                             f"({(b['crps_sum_norm']-a['crps_sum_norm'])/a['crps_sum_norm']:+.1%})"
                             f"/ 经验 {b.get('crps_sum_norm_copula', float('nan')):.4f} "
                             f"({(b.get('crps_sum_norm_copula', float('nan'))-a['crps_sum_norm'])/a['crps_sum_norm']:+.1%})"
                             f"/ 同单调 {b['crps_sum_norm_comono']:.4f}"
                             f"   |  crpsΣ {a['crps_norm_sum']:.4f} -> "
                             f"{b['crps_norm_sum']:.4f} "
                             f"({(b['crps_norm_sum']-a['crps_norm_sum'])/a['crps_norm_sum']:+.1%})"
                             + (f"   |  做法④ crpsΣ {d4['crps_norm_sum']:.4f} / "
                                f"crpsΣsum {d4['crps_sum_norm']:.4f}"
                                if d4 else ""))
    txt = "\n".join(L)
    # 汇总也必须带 sfx：否则 oracle 那一轮（泄漏真值）会把正式汇总 report.txt 覆盖掉。
    open(f"{OUT}/report{sfx}.txt", "w").write(txt + "\n")
    print("\n" + txt)
    print(f"\n已保存 {OUT}/")


if __name__ == "__main__":
    main()
