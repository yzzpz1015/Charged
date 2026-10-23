# -*- coding: utf-8 -*-
"""测底：**不确定性这一侧**最多还能榨出多少？

问题
----
概率预测 = 点预测 + 不确定性。点预测那一侧之前的结论是「时序内部已经被榨干」
（L1 / PatchTST / FreTS 挤在 R² 0.78~0.80），外部模态（POI / 卫星图）又已证明
无增量。那么剩下的杠杆只有不确定性描述。**在投入做分布模型之前，先量清楚这根
杠杆最多能撬动多少。**

怎么量：让方法作弊
------------------
现行做法（诚实版）：残差池取自**验证期**——测试时对每个起点，抽 200 条"目标小时
相同"的验证期残差向量（每站只抽它自己的残差），加到点预测上，凑出一团可能值。

作弊版：把残差池换成**测试期自己**的残差。等于事先告诉方法"你在这一段上、
每一站、每一个小时会错成什么样"。它在答案上偷看了，任何诚实做法都赢不过它。

于是：

    诚实版 CRPS  -  作弊版 CRPS  =  不确定性这一侧**最多**能拿到的改善

点预测**两边完全相同**（同一个模型、同一批起点、同一个目标时刻），所以这个差
干净地只归因于"分布描述得好不好"，不掺点预测。

怎么读（这是本脚本的全部意义）
------------------------------
  * 差得少（比如 <10%）→ 连作弊都榨不出多少，**说明不确定性已经描述得差不多了**，
    "每站每小时的误差分布"已是我们能给到的最强信息（外部特征又没增量）→ 分布这
    条线不值得投，扩散模型应当重新论证。
  * 差得多（比如 >25%）→ 现在的这团不确定性描述得很差，空间明确 → 值得投。

边界（诚实交代，别当成精确值）
------------------------------
1. 作弊版抽样本的池和被评的样本来自**同一段数据**（约 90 天 × 每小时），有共用，
   所以它会**偏乐观**——是个偏松的上限。
2. 它只按「站 + 小时」两个条件分组。理论上用上更多条件的方法可以超过它；但既然
   外部特征已被证明无增量，这个理论空间实际很窄。
3. 这里只报 crpsΣ 与 PICP，不报 CRPS_sum（后者要额外讨论站间关系，见 prob_baseline
   的文末表）。

用法
----
    python3 script/prob_oracle_floor.py [--cities LOA ...] [--ckpt_dir result/univariate]

输出：``result/prob_oracle_floor/report.txt`` 与 ``metrics.json``
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

BASE = "/home/data/yzzpz/Charged"
sys.path.insert(0, BASE)

OUT = f"{BASE}/result/prob_oracle_floor"
os.makedirs(OUT, exist_ok=True)

# 复用基线（度量 + 划分 + 剔站）与换锚点脚本（建库建模型 + 推理）。
# **单一实现**，避免 CRPS 口径或站集在两处漂移。
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pb = _load("prob_baseline", os.path.join(BASE, "script", "prob_baseline.py"))
pa = _load("prob_anchor_swap", os.path.join(BASE, "script", "prob_anchor_swap.py"))

SEQ_L, PRE_LEN = pa.SEQ_L, pa.PRE_LEN
M_SAMPLE = pa.M_SAMPLE


def run_city(city, ckpt_root, device):
    """返回每城每 H 的 {诚实版, 作弊版} 指标。

    点预测与起点集在两边完全一致，唯一变量是残差池取自 **valid 段** 还是 **test 段**。
    """
    # ---- 1. 基线口径的数据、划分、站集（与 prob_baseline / anchor_swap 同一实现）
    y0, t0 = pb.load(city)
    ok, trm, vam, tem, u, span = pb.split_rows(y0, t0)
    cols, smean = pb.live_cols(y0, trm)
    y = y0[:, cols]
    N = len(cols)
    hod = t0.hour.values

    f = lambda ts: ts.strftime('%m-%d')
    got = (f(span['tr'][0]), f(span['tr'][1]), f(span['va'][0]), f(span['va'][1]),
           f(span['te'][0]), f(span['te'][1]), span['drop_h'], N)
    if got != pa.EXPECT[city]:
        raise AssertionError(f"{city} 划分/站集与基线不一致：{got} != {pa.EXPECT[city]}")

    # ---- 2. 模型：在新进程里 reload train.pth，对 valid / test 两段推理
    ds, model, ckpt = pa.build_model(city, ckpt_root, device)
    if ds.feat.shape[1] != N:
        raise AssertionError(f"{city} 模型侧站数 {ds.feat.shape[1]} != 基线 {N}")

    va_ds, te_ds = ds.valid_loader.dataset, ds.test_loader.dataset
    pred = {'va': pa.inverse_3d(ds.scaler, pa.predict(model, va_ds, device)),   # [M,N,72]
            'te': pa.inverse_3d(ds.scaler, pa.predict(model, te_ds, device))}
    print(f"[{city}] 站 {N}  验证起点 {len(va_ds.starts)}  测试起点 {len(te_ds.starts)}",
          flush=True)

    stats = {}
    for H in pb.HORIZONS:
        j = H - 1                                   # 块内第 H 步 = 提前量 H 小时
        lag = pb.LAG * int(np.ceil(H / pb.LAG))     # 诚实/作弊两边都只走模型，lag 仅用于起点筛

        # 起点集只看 test；锚点有效性判据与锚点无关，故两边必然同一批起点
        s = np.asarray(te_ds.starts) + SEQ_L + j
        m = ok[s - lag]
        s = s[m]
        obs_te = y[s]
        hour_te = hod[s]
        pred_te = pred['te'][m][:, :, j]

        # valid 段：同一个 H 的残差池来源
        sv = np.asarray(va_ds.starts) + SEQ_L + j
        mv = ok[sv - lag]
        sv = sv[mv]
        pred_va = pred['va'][mv][:, :, j]
        hour_va = hod[sv]

        # 两个池，**同一个点预测**（PatchTST），只有残差来源不同
        pools = {
            'honest': {'res': y[sv] - pred_va, 'hour': hour_va, 'n': int(sv.size)},
            'oracle': {'res': obs_te - pred_te, 'hour': hour_te, 'n': int(s.size)},
        }

        row = {'n_origin': int(len(s))}
        rng = np.random.default_rng(0)
        for name, P in pools.items():
            crps_marg, per_st, nacc = [], np.zeros(N), 0
            pit = {k: [] for k in ('picp50', 'mpiw50', 'picp80', 'mpiw80',
                                   'picp90', 'mpiw90')}
            for i in range(len(s)):
                idx = np.where(P['hour'] == hour_te[i])[0]
                if len(idx) == 0:
                    continue
                pick = rng.choice(idx, min(M_SAMPLE, len(idx)), replace=True)
                smp = pred_te[i][None, :] + P['res'][pick]     # [m, N] 整场样本
                cs = pb.crps_fair(smp, obs_te[i])
                per_st += np.nan_to_num(cs); nacc += 1
                crps_marg.append(cs.mean())
                for k, v in pb.pit_metrics(smp, obs_te[i]).items():
                    pit[k].append(v)
            pm = per_st / max(nacc, 1) / smean
            row[name] = dict(
                n_pool=P['n'],
                crps_norm_sum=float(np.mean(per_st / max(nacc, 1)) / np.nanmean(smean)),
                crps_norm_med=float(np.nanmedian(pm)),
                crps_marg=float(np.mean(crps_marg)),
                **{k: float(np.mean(v)) for k, v in pit.items()},
            )
        row['point_mae'] = float(np.abs(pred_te - obs_te).mean())
        row['gain'] = float(row['honest']['crps_norm_sum'] - row['oracle']['crps_norm_sum'])
        row['gain_pct'] = row['gain'] / row['honest']['crps_norm_sum']
        stats[H] = row
        print(f"[{city} H={H}] 诚实 {row['honest']['crps_norm_sum']:.4f} -> "
              f"作弊 {row['oracle']['crps_norm_sum']:.4f}  "
              f"可榨 {row['gain']:.4f} ({row['gain_pct']:.1%})  "
              f"PICP90 {row['honest']['picp90']:.1%} -> {row['oracle']['picp90']:.1%}",
              flush=True)

    return {'city': city, 'n_station': N, 'span': span, 'ckpt': os.path.relpath(ckpt, BASE),
            **stats}


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
    P("=" * 112)
    P("测底：不确定性这一侧最多还能榨出多少？（残差池 valid 段 -> 测试期自己）")
    P("  点预测两边**完全相同**（都是 PatchTST，同一批起点、同一目标时刻），")
    P("  唯一变量是残差从哪段抽。所以差值干净地只归因于「不确定性的描述」。")
    P("  作弊版 = 事先知道每站每小时会错成什么样（在答案上偷看）-> 诚实的做法赢不过它。")
    P("  '可榨' = 诚实 crpsΣ - 作弊 crpsΣ，即这条杠杆的**可用上限**。")
    P("=" * 112)
    for H in pb.HORIZONS:
        P(f"\n--- H = {H}（提前量 {H} 小时；池 = 验证段 vs 测试段自己）---")
        P(f"{'城':<5}{'站数':>6}{'起点':>7}{'点MAE':>8}"
          f"{'诚实crpsΣ':>11}{'作弊crpsΣ':>11}{'可榨':>9}{'可榨%':>8}"
          f"{'PICP90诚':>10}{'PICP90弊':>10}{'PICP50诚':>10}{'PICP50弊':>10}"
          f"{'MPIW50诚':>10}{'MPIW50弊':>10}")
        P("-" * 112)
        for r in allres:
            d = r[H]; a, b = d['honest'], d['oracle']
            P(f"{r['city']:<5}{r['n_station']:>6}{d['n_origin']:>7}{d['point_mae']:>8.3f}"
              f"{a['crps_norm_sum']:>11.4f}{b['crps_norm_sum']:>11.4f}"
              f"{d['gain']:>9.4f}{d['gain_pct']:>8.1%}"
              f"{a['picp90']:>10.1%}{b['picp90']:>10.1%}"
              f"{a['picp50']:>10.1%}{b['picp50']:>10.1%}"
              f"{a['mpiw50']:>10.2f}{b['mpiw50']:>10.2f}")
    P("\n读法：")
    P("  * '可榨%' 小（<10%）→ 连作弊都榨不出来：不确定性已经描述得差不多了，")
    P("    '每站每小时的误差分布'已是能拿到的最强信息（外部特征又无增量）→ 分布这条线不值投。")
    P("  * '可榨%' 大（>25%）→ 现在这团不确定性描述得很差，空间明确 → 值得投。")
    P("  * PICP90 诚实列若明显低于 90%，说明现在区间系统性偏窄；作弊列会回到 90% 附近")
    P("    （用测试期自己的残差必然校准）——两者的差就是'校准'能修回来的那部分。")
    P("  * 边界：作弊版池与被评样本同段，偏乐观，是**偏松的上限**；且只按「站+小时」条件分组。")
    txt = "\n".join(L)
    open(f"{OUT}/report.txt", "w").write(txt + "\n")
    print("\n" + txt)
    print(f"\n已保存 {OUT}/report.txt  和  metrics.json")


if __name__ == "__main__":
    main()
