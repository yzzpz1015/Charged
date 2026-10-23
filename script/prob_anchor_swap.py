# -*- coding: utf-8 -*-
"""在基线框架里**只换锚点**：seasonal naive -> PatchTST 点预测。

动机
----
概率预测 = 点预测 + 不确定性。现行基线 ``script/prob_baseline.py`` 的分布部分
是「残差场 bootstrap」：训练期残差 ``r = 观测 - 点预测``，测试时抽整条残差向量
加到点预测上。因此 CRPS 的改善只可能来自两处——

  (a) 点预测变准（锚点好）;  (b) 残差分布描述得更准（宽度/形状对）。

本脚本把 (a) 单独隔离：锚点换成训好的 PatchTST，**其余原封不动**（同一划分、
同一站集、同一按目标小时分层的残差场 bootstrap、同一 CRPS/CRPS_sum/PICP 口径）。
于是两行的差可以干净地归因到点预测能力，也就能回答「分布侧还剩多少空间」。

口径对齐（**关键**，不看清会得出错误结论）
------------------------------------------
基线的 ``H=24`` 是「单步、提前 24h」；模型的 ``pre_len=72`` 是「72 步块」。
同一个数字两种含义。为了在同一批起点上比，统一以**模型窗口**为准：

    起点 t 需满足 [t, t+72+72) 整段可用，且目标块 [t+72, t+144) 全落在 test 段。
    对给定时长 H，两边都预测同一个目标时刻 s = t + 72 + (H-1)：
        基线锚点 = y(s - 24*ceil(H/24))       （Hyndman 季节朴素）
        模型锚点 = PatchTST 第 H 步输出

两边的残差池都在这批起点对应的**训练窗口**上重算——锚点换了，残差当然要重算。

用法
----
    python3 script/prob_anchor_swap.py [--cities LOA ...] [--ckpt_dir result/univariate]

输出：``result/prob_anchor_swap/report.txt`` 与 ``metrics.json``
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

BASE = "/home/data/yzzpz/Charged"
sys.path.insert(0, BASE)

# 复用基线的度量定义与划分口径——**单一实现**，避免两份 CRPS 代码漂移。
_spec = importlib.util.spec_from_file_location(
    "prob_baseline", os.path.join(BASE, "script", "prob_baseline.py"))
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)

SEQ_L = 72
PRE_LEN = 72          # 一次训到 72 步，H=24/48/72 都在里面取，省两次训练
M_SAMPLE = 200        # 与基线一致
OUT = f"{BASE}/result/prob_anchor_swap"
os.makedirs(OUT, exist_ok=True)

# 与 report.txt 逐行一致的划分边界，用来断言两边口径没跑偏。
# 末位站数是**剔掉训练期退化站之后**的（死站 + 常数平台，见 prob_baseline.live_cols）。
EXPECT = {
    'LOA': ('04-01', '05-31', '06-01', '07-01', '07-01', '09-30', 0, 218),
    'MEL': ('04-01', '05-31', '05-31', '07-01', '07-01', '09-30', 19, 56),
    'JHB': ('04-01', '05-31', '06-01', '07-01', '07-01', '09-30', 0, 29),
    'AMS': ('05-07', '06-25', '06-26', '07-20', '07-20', '09-30', 901, 1383),
    'SPO': ('04-01', '05-31', '06-01', '07-01', '07-01', '09-30', 0, 33),
    'SZH': ('04-01', '05-31', '06-01', '07-01', '07-01', '09-30', 0, 843),
}


def inverse_3d(scaler, arr):
    """``arr`` = [M, N, P] 标准化空间 -> 原空间。与 trainer 里的实现同式。"""
    M, N, P = arr.shape
    r = arr.transpose(0, 2, 1).reshape(-1, N)
    r = scaler.inverse_transform(r)
    return r.reshape(M, P, N).transpose(0, 2, 1)


def find_ckpt(ckpt_root, city):
    pat = f"{ckpt_root}/{city}/patchtst/volume-None-site-{SEQ_L}-{PRE_LEN}-6*/train.pth"
    hits = sorted(glob.glob(pat), key=os.path.getmtime)
    if not hits:
        raise FileNotFoundError(f"没找到 {city} 的 {SEQ_L}-{PRE_LEN} 检查点（{pat}）")
    return hits[-1]      # 最新一个（输出目录自动加 #n 后缀）


def build_model(city, ckpt_root, device):
    """复现 example/univariate_prediction.py 的建库建模型流程，加载检查点。"""
    from api.dataset.common import EVDataset
    from api.model.config import PredictionModel

    ds = EVDataset(feature='volume', auxiliary='None',
                   data_path=f"{BASE}/data/{city}_remove_const/")
    ds.split_cross_validation(fold=6, total_fold=6, train_ratio=0.8, valid_ratio=0.1)
    ds.create_loaders(seq_l=SEQ_L, pre_len=PRE_LEN, batch_size=32,
                      device=device, model_name='patchtst')
    model = PredictionModel(num_node=ds.feat.shape[1], n_fea=1,
                            model_name='patchtst', seq_l=SEQ_L, pre_len=PRE_LEN)
    ckpt = find_ckpt(ckpt_root, city)
    model.load_model(ckpt)
    model.model.to(device).eval()
    return ds, model, ckpt


@torch.no_grad()
def predict(model, dataset, device, batch_size=16):
    """在整个 dataset（CreateDataset，带 starts）上推理，返回原空间的 [M, N, P]。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outs = []
    for batch in loader:
        x = batch[0].to(device)
        outs.append(model.model(x).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def run_city(city, ckpt_root, device):
    # ---- 1. 基线口径的数据、划分、站集 ----
    y0, t0 = pb.load(city)
    ok, trm, vam, tem, u, span = pb.split_rows(y0, t0)
    cols, smean = pb.live_cols(y0, trm)
    y = y0[:, cols]
    N = len(cols)
    hod = t0.hour.values

    f = lambda ts: ts.strftime('%m-%d')
    got = (f(span['tr'][0]), f(span['tr'][1]), f(span['va'][0]), f(span['va'][1]),
           f(span['te'][0]), f(span['te'][1]), span['drop_h'], N)
    if got != EXPECT[city]:
        raise AssertionError(f"{city} 划分/站集与 report.txt 不一致：{got} != {EXPECT[city]}")

    # ---- 2. 模型：同一划分 -> 检查点 -> 训练段/测试段的 72 步点预测 ----
    ds, model, ckpt = build_model(city, ckpt_root, device)
    if ds.feat.shape[1] != N:
        raise AssertionError(f"{city} 模型侧站数 {ds.feat.shape[1]} != 基线 {N}")
    # 站序也要一致（都是按 volume.csv 原列序剔除，理论上必然一致，仍断言一次）
    if [str(c) for c in ds.feat_columns] != [str(c) for c in
                                             pd.read_csv(f"{BASE}/data/{city}_remove_const/volume.csv",
                                                         index_col=0).columns[cols]]:
        raise AssertionError(f"{city} 模型侧与基线侧站序不一致")

    tr_ds = ds.train_loader.dataset
    va_ds = ds.valid_loader.dataset
    te_ds = ds.test_loader.dataset
    pred = {name: inverse_3d(ds.scaler, predict(model, d, device))   # [M, N, 72]
            for name, d in (('tr', tr_ds), ('va', va_ds), ('te', te_ds))}
    print(f"[{city}] 起点数 训练 {len(tr_ds.starts)} / 验证 {len(va_ds.starts)} / "
          f"测试 {len(te_ds.starts)}", flush=True)

    stats = {}
    for H in pb.HORIZONS:
        j = H - 1                                   # 块内第 H 步 = 提前量 H 小时
        lag = pb.LAG * int(np.ceil(H / pb.LAG))     # 基线的季节朴素锚点回看步数

        # 目标时刻（模型口径）：历史结束于 t+71，第 j 步落在 t+72+j；锚点 = 目标 - lag。
        segs = {}
        for name, d in (('tr', tr_ds), ('va', va_ds), ('te', te_ds)):
            s = np.asarray(d.starts) + SEQ_L + j
            a = s - lag
            m = ok[a]                    # 锚点必须有效，否则基线那一行没定义 -> 起点集不同
            segs[name] = dict(s=s[m], obs=y[s[m]], naive=y[a[m]],
                              model=pred[name][m][:, :, j], hour=hod[s[m]])
        te = segs['te']

        row = {'n_origin': int(len(te['s'])),
               'ckpt': os.path.relpath(ckpt, BASE)}
        rng = np.random.default_rng(0)
        for src in ('va', 'tr'):
            # 残差池的来源段。**主看 va**：模型在 train 上是最小化过损失的，
            # train 残差是样本内残差、系统性偏窄，拿它当预测分布会同时压低 CRPS 与
            # PICP（乐观偏置）。valid 段对模型是样本外的（仅有 early-stop 选轮次的
            # 轻微污染，属常规做法），naive 那边则本来就无拟合，两边同源可比。
            # tr 池一并算出，只为与 prob_baseline 的 train 池口径对得上，作一致性参照。
            row[f'n_pool_{src}'] = int(len(segs[src]['s']))
            pools = {
                'naive':    {'res': segs[src]['obs'] - segs[src]['naive'],
                             'hour': segs[src]['hour'], 'point': te['naive']},
                'patchtst': {'res': segs[src]['obs'] - segs[src]['model'],
                             'hour': segs[src]['hour'], 'point': te['model']},
            }
            for name, P in pools.items():
                crps_marg, crps_sum, point = [], [], []
                pit = {k: [] for k in ('picp50', 'mpiw50', 'picp80', 'mpiw80',
                                       'picp90', 'mpiw90')}
                per_st, nacc = np.zeros(N), 0
                for i in range(len(te['s'])):
                    idx = np.where(P['hour'] == te['hour'][i])[0]
                    if len(idx) == 0:
                        continue
                    pick = rng.choice(idx, min(M_SAMPLE, len(idx)), replace=True)
                    smp = P['point'][i][None, :] + P['res'][pick]        # [m, N] 整场样本
                    cs = pb.crps_fair(smp, te['obs'][i])
                    per_st += np.nan_to_num(cs); nacc += 1
                    crps_marg.append(cs.mean())
                    crps_sum.append(float(pb.crps_fair(smp.sum(1), te['obs'][i].sum())))
                    point.append(float(np.abs(P['point'][i] - te['obs'][i]).mean()))
                    for k, v in pb.pit_metrics(smp, te['obs'][i]).items():
                        pit[k].append(v)
                pm = per_st / max(nacc, 1) / smean
                row[f'{src}_{name}'] = dict(
                    crps_marg=float(np.mean(crps_marg)),
                    crps_norm_sum=float(np.mean(per_st / max(nacc, 1)) / np.nanmean(smean)),
                    crps_norm_med=float(np.nanmedian(pm)),
                    crps_sum=float(np.mean(crps_sum)),
                    crps_sum_norm=float(np.mean(crps_sum) / smean.sum()),
                    point=float(np.mean(point)),
                    **{k: float(np.mean(v)) for k, v in pit.items()},
                )
        stats[H] = row
        print(f"[{city} H={H}] 池 va{row['n_pool_va']}/tr{row['n_pool_tr']}  "
              f"naive crpsΣ={row['va_naive']['crps_norm_sum']:.4f} -> "
              f"PT crpsΣ={row['va_patchtst']['crps_norm_sum']:.4f}  "
              f"(起点 {row['n_origin']})", flush=True)

    return {'city': city, 'n_station': N, 'span': span, **stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cities', nargs='*', default=pb.CITIES)
    ap.add_argument('--ckpt_dir', default=f"{BASE}/result/univariate")
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    allres = []
    for c in args.cities:
        allres.append(run_city(c, args.ckpt_dir, device))
        json.dump(allres, open(f"{OUT}/metrics.json", "w"), ensure_ascii=False,
                  indent=1, default=str)

    L = []
    P = L.append
    P("=" * 126)
    P("换锚点实验：seasonal naive  ->  PatchTST 72 步点预测")
    P("  只换点预测，残差场 bootstrap（按目标小时分层、整条向量抽样）与 CRPS 口径完全不动。")
    P("  起点集 = 模型口径（[t, t+144) 全可用、目标块落 test），两边**同一批起点、同一批站**。")
    P("  crpsΣ = Σ_j CRPS_j / Σ_j smean_j（按站规模加权，跨城可比，主看这个）")
    P("  Δ点误差 = (PatchTST 点 MAE - naive 点 MAE) / naive 点 MAE，负=点预测变好")
    P("=" * 126)

    def table(src, title):
        P(f"\n--- {title} ---")
        for H in pb.HORIZONS:
            P(f"\n  H = {H}（提前量 {H} 小时；池 = {src} 段 {allres[0][H]['n_pool_' + src]} 个起点）")
            P(f"  {'城':<5}{'站数':>6}{'起点':>7}"
              f"{'naive点':>10}{'PatchTST点':>12}{'Δ点误差':>10}"
              f"{'naivecrps':>11}{'PTcrps':>10}{'Δcrps':>10}"
              f"{'naivecrpsΣ':>12}{'PTcrpsΣ':>11}{'ΔcrpsΣ':>10}"
              f"{'PT_PICP50':>11}{'PT_PICP90':>11}")
            P("  " + "-" * 120)
            for r in allres:
                d = r[H]; a, b = d[f'{src}_naive'], d[f'{src}_patchtst']
                P(f"  {r['city']:<5}{r['n_station']:>6}{d['n_origin']:>7}"
                  f"{a['point']:>10.3f}{b['point']:>12.3f}"
                  f"{(b['point']-a['point'])/a['point']:>9.1%}"
                  f"{a['crps_marg']:>11.3f}{b['crps_marg']:>10.3f}"
                  f"{(b['crps_marg']-a['crps_marg'])/a['crps_marg']:>10.1%}"
                  f"{a['crps_norm_sum']:>12.4f}{b['crps_norm_sum']:>11.4f}"
                  f"{(b['crps_norm_sum']-a['crps_norm_sum'])/a['crps_norm_sum']:>10.1%}"
                  f"{b['picp50']:>11.1%}{b['picp90']:>11.1%}")

    table('va', '主口径：残差池取 valid 段（对模型样本外）')
    table('tr', '参照：残差池取 train 段（样本内，乐观偏置；此表可与 prob_baseline 对照）')

    P("\n读法：")
    P("  * 主看 va 表。tr 表把 PatchTST 的残差池换成样本内残差，数字必然比 va 表好看——")
    P("    两者的差距就是「样本内残差偏窄」带来的乐观幅度，别拿 tr 表的 crpsΣ 当结论。")
    P("  * 若 PatchTST 点误差明显更小、crpsΣ 却几乎不动 —— 说明瓶颈在**残差分布**（宽度/形状），")
    P("    而不在点预测；这正是扩散模型该发力的地方。")
    P("  * 若 crpsΣ 跟着点误差一起降 —— 说明当前瓶颈还在点预测，先别急着上分布模型。")
    P("  * 也看 PICP：点预测变好但 PICP 掉到 50% 以下，说明残差池（valid）与测试期的宽度")
    P("    仍不匹配，需要区间校准。")
    P("  * 注：本脚本的起点集要求整段 [t, t+144) 可用（模型口径），比 prob_baseline 的单锚点")
    P("    口径更严，故起点数略少、naive 的 crpsΣ 与 report.txt 不会逐位相等。")
    open(f"{OUT}/report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n写入 {OUT}/report.txt")


if __name__ == '__main__':
    main()
