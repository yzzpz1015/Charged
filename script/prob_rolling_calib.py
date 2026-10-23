# -*- coding: utf-8 -*-
"""把「可榨」拆开：多少属于**点预测的偏移**（简单方法就能修），多少真正属于**分布**。

为什么必须拆
------------
``script/prob_oracle_floor.py`` 量出「不确定性这一侧最多还能榨出多少」（作弊版 vs
诚实版）。但那个差里混着两样东西：

  (i)  点预测在测试期的**系统性偏移** —— 诚实版的区间以点预测为中心铺开，点预测
       整体偏了，铺得再宽也盖不住真值。作弊版因为残差取自测试期自己，自动带上
       那个偏移，等于顺手把中心也纠正了。
  (ii) 真正的**不确定性宽度/形状**描述得准不准。

光看那个总差会把 (i) 误记成"分布的空间"。本脚本用一条**因果的**滚动校准把两者
分开：只用目标时刻**之前**已完成的预测误差来定中心与宽度 —— 不偷看答案，部署时
真能这么干。

四种口径（同一个点预测，只有残差池不同）
----------------------------------------
  honest   : 残差池 = 验证段（现状基线）
  roll{N}  : 残差池 = 目标时刻前 N 天的已完成误差，按目标小时取同小时的
             —— **因果**，不偷看。N ∈ {7, 14, 28}
  center   : 验证段残差池，但按「站 × 小时」平移到测试期的均值 —— **作弊**，
             只用来做诊断：它隔离出"只修中心"能拿到多少
  oracle   : 残差池 = 测试段自己 —— **作弊**，上限

读法
----
  滚动吃掉了多少 = (honest - roll28) / (honest - oracle)

  * 接近 1 → 「可榨」几乎全是"验证段和测试期的误差对不上"，**因果的滚动校准就能
    拿到**，分布这条线没有额外空间。
  * 明显小于 1 → 有一部分是滚动校准吃不掉的，那才是真正属于分布模型的空间。

  **实测（2026-10-06，6 城，AMS 排除出汇总）**：滚动 28 天吃掉 58%~65%
  （H=24 58% / H=48 64% / H=72 65%），滚完只剩约 5%。

  主导因素**不是**中心偏移：center 那一列（作弊，只把中心修到测试期均值）
  只吃掉 10%~22%，SZH 上甚至为负（中心修对了反而更差）。所以大头是**误差的
  宽度/形状**在验证段与测试期之间不一样，而滚动校准顺带把宽度也对齐了。

  注意 roll7 在 H=24 上吃掉是负的（比 honest 还差）——7 天太短，样本不够，
  窗口大小本身是个要调的量。

用法
----
    python3 script/prob_rolling_calib.py [--cities LOA MEL ...] [--ckpt_dir result/univariate]

输出：``result/prob_rolling_calib/report.txt`` 与 ``metrics.json``
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

OUT = f"{BASE}/result/prob_rolling_calib"
os.makedirs(OUT, exist_ok=True)

ROLL_DAYS = (7, 14, 28)
# AMS 训练期只有约 40 天有效数据、且与测试期量级差 1.53 倍，中心偏移极大，
# 会把跨城平均值拉偏。**默认排除出汇总**，表里仍列出（带 * 标记）供单独查看。
SKEWED = {"AMS"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pb = _load("prob_baseline", os.path.join(BASE, "script", "prob_baseline.py"))
pa = _load("prob_anchor_swap", os.path.join(BASE, "script", "prob_anchor_swap.py"))

SEQ_L, PRE_LEN = pa.SEQ_L, pa.PRE_LEN
M_SAMPLE = pa.M_SAMPLE


def _seg(ds, pred_all, j, lag, ok, y, hod, split):
    """取出该 split 里、步长 j 的目标时刻上的 (起止下标, 目标行号, 观测, 预测, 小时)。"""
    s = np.asarray(ds.starts) + SEQ_L + j
    m = ok[s - lag]
    s = s[m]
    return dict(s=s, obs=y[s],
                pred=pred_all[split][m][:, :, j], hour=hod[s])


def run_city(city, ckpt_root, device):
    y0, t0 = pb.load(city)
    ok, trm, vam, tem, u, span = pb.split_rows(y0, t0)
    cols, smean = pb.live_cols(y0, trm)
    y = y0[:, cols]
    N = len(cols)
    hod = t0.hour.values
    smean_sum, smean_mean = smean.sum(), np.nanmean(smean)

    f = lambda ts: ts.strftime('%m-%d')
    got = (f(span['tr'][0]), f(span['tr'][1]), f(span['va'][0]), f(span['va'][1]),
           f(span['te'][0]), f(span['te'][1]), span['drop_h'], N)
    if got != pa.EXPECT[city]:
        raise AssertionError(f"{city} 划分/站集与基线不一致：{got} != {pa.EXPECT[city]}")

    ds, model, ckpt = pa.build_model(city, ckpt_root, device)
    if ds.feat.shape[1] != N:
        raise AssertionError(f"{city} 模型侧站数 {ds.feat.shape[1]} != 基线 {N}")

    va_ds, te_ds = ds.valid_loader.dataset, ds.test_loader.dataset
    pred_all = {'va': pa.inverse_3d(ds.scaler, pa.predict(model, va_ds, device)),
                'te': pa.inverse_3d(ds.scaler, pa.predict(model, te_ds, device))}
    print(f"[{city}] 站 {N}  验证起点 {len(va_ds.starts)}  测试起点 {len(te_ds.starts)}",
          flush=True)

    stats = {}
    for H in pb.HORIZONS:
        j = H - 1
        lag = pb.LAG * int(np.ceil(H / pb.LAG))
        te = _seg(te_ds, pred_all, j, lag, ok, y, hod, 'te')
        va = _seg(va_ds, pred_all, j, lag, ok, y, hod, 'va')

        # 历史池 = 验证段 + 测试段的**已完成**目标（因果筛选靠 s_hist < s_i）
        s_h = np.concatenate([va['s'], te['s']])
        res_h = np.concatenate([va['obs'] - va['pred'], te['obs'] - te['pred']])
        hour_h = np.concatenate([va['hour'], te['hour']])

        # center 口径：按「站 × 小时」把验证段残差平移到测试期均值（**作弊，仅诊断**）
        shift = np.zeros((24, N))
        for hh in range(24):
            mi = te['hour'] == hh
            if mi.any():
                shift[hh] = (te['obs'][mi] - te['pred'][mi]).mean(0)
        res_va = va['obs'] - va['pred']
        res_va_center = res_va - np.stack(
            [res_va[va['hour'] == hh].mean(0) if (va['hour'] == hh).any() else np.zeros(N)
             for hh in range(24)])[va['hour']] + shift[va['hour']]

        P = len(te['s'])
        specs = {f'roll{d}': dict(kind='roll', days=d) for d in ROLL_DAYS}
        specs['honest'] = dict(kind='fixed', res=res_va, hour=va['hour'])
        specs['center'] = dict(kind='fixed', res=res_va_center, hour=va['hour'])
        specs['oracle'] = dict(kind='fixed', res=te['obs'] - te['pred'], hour=te['hour'])

        row = {'n_origin': int(P), 'point_mae': float(np.abs(te['pred'] - te['obs']).mean())}
        rng = np.random.default_rng(0)
        for name, sp in specs.items():
            per_st, nacc = np.zeros(N), 0
            crps_marg = []
            pit90 = []
            for i in range(P):
                if sp['kind'] == 'roll':
                    m = ((hour_h == te['hour'][i]) & (s_h < te['s'][i]) &
                         (s_h >= te['s'][i] - sp['days'] * 24))
                    res, hour = res_h[m], hour_h[m]
                else:
                    m = sp['hour'] == te['hour'][i]
                    res, hour = sp['res'][m], None
                if len(res) == 0:
                    continue
                pick = rng.choice(len(res), min(M_SAMPLE, len(res)), replace=True)
                smp = te['pred'][i][None, :] + res[pick]
                cs = pb.crps_fair(smp, te['obs'][i])
                per_st += np.nan_to_num(cs); nacc += 1
                crps_marg.append(cs.mean())
                lo, hi = np.percentile(smp, [5, 95], axis=0)
                pit90.append(float(np.mean((te['obs'][i] >= lo) & (te['obs'][i] <= hi))))
            pm = per_st / max(nacc, 1) / smean
            row[name] = dict(
                crps_norm_sum=float(np.mean(per_st / max(nacc, 1)) / smean_mean),
                crps_norm_med=float(np.nanmedian(pm)),
                crps_marg=float(np.mean(crps_marg)),
                picp90=float(np.mean(pit90)),
            )
        # 滚动吃掉了「可榨」的多少
        for d in ROLL_DAYS:
            den = row['honest']['crps_norm_sum'] - row['oracle']['crps_norm_sum']
            num = row['honest']['crps_norm_sum'] - row[f'roll{d}']['crps_norm_sum']
            row[f'eaten{d}'] = float(num / den) if den > 0 else np.nan
        stats[H] = row
        print(f"[{city} H={H}] honest {row['honest']['crps_norm_sum']:.4f}  "
              + "  ".join(f"roll{d} {row[f'roll{d}']['crps_norm_sum']:.4f}" for d in ROLL_DAYS)
              + f"  oracle {row['oracle']['crps_norm_sum']:.4f}  "
              + f"吃掉 {row['eaten28']:.0%} (28d)  正中心(作弊) {row['center']['crps_norm_sum']:.4f}",
              flush=True)

    return {'city': city, 'n_station': N, 'span': span,
            'ckpt': os.path.relpath(ckpt, BASE), **stats}


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
    P("=" * 124)
    P("把「可榨」拆开：点预测的偏移（因果方法能修） vs 真正的不确定性描述（分布模型的空间）")
    P("  同一个点预测（PatchTST）、同一批起点、同一目标时刻；只有残差池的来源不同。")
    P("  honest = 验证段（现状）；rollN = 目标时刻前 N 天的已完成误差（**因果，不偷看**）")
    P("  center = 验证段残差按「站×小时」平移到测试期均值（**作弊，仅诊断**，隔离『只修中心』）")
    P("  oracle = 测试段自己（**作弊上限**）")
    P("  '吃掉' = (honest - roll) / (honest - oracle)，即**因果校准能拿到可榨量的比例**")
    P("=" * 124)
    for H in pb.HORIZONS:
        P(f"\n--- H = {H}（提前量 {H} 小时）---")
        P(f"{'城':<6}{'站数':>6}{'起点':>7}{'点MAE':>8}"
          f"{'honest':>9}{'roll7':>9}{'roll14':>9}{'roll28':>9}"
          f"{'center*':>9}{'oracle*':>9}{'吃掉7':>8}{'吃掉28':>8}"
          f"{'PICP90诚':>10}{'PICP90r28':>11}{'PICP90弊':>10}")
        P("-" * 124)
        for r in allres:
            d = r[H]
            star = '*' if r['city'] in SKEWED else ' '
            P(f"{r['city'] + star:<6}{r['n_station']:>6}{d['n_origin']:>7}{d['point_mae']:>8.3f}"
              f"{d['honest']['crps_norm_sum']:>9.4f}"
              + "".join(f"{d[f'roll{x}']['crps_norm_sum']:>9.4f}" for x in ROLL_DAYS)
              + f"{d['center']['crps_norm_sum']:>9.4f}{d['oracle']['crps_norm_sum']:>9.4f}"
              f"{d['eaten7']:>8.0%}{d['eaten28']:>8.0%}"
              f"{d['honest']['picp90']:>10.1%}{d['roll28']['picp90']:>11.1%}"
              f"{d['oracle']['picp90']:>10.1%}")

    # ---- 汇总：只用非偏斜城，看"因果校准到底吃掉了多少"
    P("\n--- 汇总（不含 * 标记的城）---")
    P(f"{'H':>4}{'honest':>10}{'roll7':>10}{'roll14':>10}{'roll28':>10}"
      f"{'oracle':>10}{'吃掉7':>9}{'吃掉14':>9}{'吃掉28':>9}")
    P("-" * 92)
    for H in pb.HORIZONS:
        rs = [r for r in allres if r['city'] not in SKEWED]
        g = lambda k: float(np.mean([r[H][k]['crps_norm_sum'] for r in rs]))
        eh = g('honest') - g('oracle')
        P(f"{H:>4}{g('honest'):>10.4f}"
          + "".join(f"{g(f'roll{x}'):>10.4f}" for x in ROLL_DAYS)
          + f"{g('oracle'):>10.4f}"
          + "".join(f"{(g('honest') - g(f'roll{x}')) / eh:>9.0%}" for x in ROLL_DAYS))
    P("\n读法：")
    P("  * '吃掉' 接近 100% → 「可榨」几乎全是'验证段与测试期的误差对不上'，")
    P("    **因果的滚动校准就能拿到**，分布这条线没有额外空间。")
    P("  * '吃掉' 明显小于 100% → 剩下的那截才是真正属于分布模型的空间。")
    P("  * 【实测结论 2026-10-06】滚动 28 天吃掉 58%~65%，滚完只剩约 5%。")
    P("    主导因素**不是**中心偏移：center 列（作弊，只修中心）只吃掉 10%~22%，")
    P("    SZH 上甚至为负。所以大头是**误差的宽度/形状**在验证段与测试期之间不一样，")
    P("    滚动校准顺带把宽度也对齐了。")
    P("  * roll7 在 H=24 上吃掉为负（比 honest 还差）——窗口太短样本不够，")
    P("    '滚动多少天'本身是个要调的量。")
    P("  * 带 * 的 AMS：训练期仅约 40 天有效、与测试期量级差 1.53 倍，中心偏移极大，")
    P("    会拉偏跨城平均，故排除出汇总；其单独数值仍列在表内。")
    txt = "\n".join(L)
    open(f"{OUT}/report.txt", "w").write(txt + "\n")
    print("\n" + txt)
    print(f"\n已保存 {OUT}/report.txt  和  metrics.json")


if __name__ == "__main__":
    main()
