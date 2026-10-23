# -*- coding: utf-8 -*-
"""作弊版到底是不是上限？—— 给它加上「条件化」，看能不能跌破原来的天花板。

问题的由来
----------
``prob_oracle_floor.py`` 拿一个"作弊"的残差池（测试期自己）当**不确定性这一侧的
上限**，量出「最多还能榨多少」。但那个池**只按「站 + 小时」分组**：

    预测 8/5 14 点 → 从"测试期所有 14 点的残差"里抽

它对"今天是哪一天"一无所知 —— 周末和平常日一样宽、爆单日和清淡日一样宽。
这其实是**气候学分布**（climatological distribution，即"长期平均长什么样"），
不是真正的条件分布。而概率预测的全部本事恰恰在于**条件化**：能从输入里看出
"今天会特别难猜"的方法，本来就该给出更宽的区间。所以：

    **原来的天花板，只是「不条件化」这一族的天花板，不是概率预测的天花板。**

本脚本直接测这个：给作弊版加上条件，如果分数跌破原来的 0.xxxx，就证明它不是上限。

条件（都保留整条向量抽样，站间结构不拆）
----------------------------------------
  wk  : 工作日 / 周末（从时间戳读，完全已知）
  amp : 全城预测幅度的三分位（该小时上把 `预测值求和` 分三档）——
        量级大时绝对误差天然更大，混在一起会白白铺宽

口径（同一个 PatchTST 点预测、同一批起点，唯一变量是残差池）
------------------------------------------------------------
  honest    : 验证段，按小时                        —— 现状
  roll28    : 目标前 28 天，按小时                  —— 因果
  rollWA28  : 目标前 28 天，按 (小时,周末,幅度)      —— 因果 + 条件化
  oracleH   : 测试期自己，按小时                    —— 原"天花板"
  oracleW   : 测试期自己，(小时,周末)
  oracleA   : 测试期自己，(小时,幅度)
  oracleWA  : 测试期自己，(小时,周末,幅度)
  ※ oracle* 全部作弊（偷看未来），只用来量"空间有多大"

读法
----
  * oracleW / oracleA / oracleWA **低于** oracleH  → 原天花板被推翻，条件化是真空间，
    且差额就是"光靠'今天是哪天/量级多大'这一层条件"能拿到的量。
  * rollWA28 与 oracleWA 的差值 → 条件化之后还剩多少（这时的剩余才更接近真上限）。

边界
----
  1. 仍然只有两个条件，一个真正的分布模型能条件在**输入本身**（近 72h 的形状）上，
     理论上还能再低。所以本脚本给出的仍是**偏松的上限**，只是比原来松得少。
  2. 条件越细，池越小，bootstrap 噪声越大 —— 所以每行都打印池大小 n_pool。
  3. 只报 crpsΣ（边际、按站量级加权）。加条件后池按整条向量抽，CRPS_sum 依然可用，
     但本表不报。

用法
----
    python3 script/prob_cond_probe.py [--cities LOA ...] [--device 0]
输出：``result/prob_cond_probe/{report_<CITY>.txt, metrics_<CITY>.json, report.txt}``
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

OUT = f"{BASE}/result/prob_cond_probe"
os.makedirs(OUT, exist_ok=True)

ROLL_DAYS = 28


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pb = _load("prob_baseline", os.path.join(BASE, "script", "prob_baseline.py"))
pa = _load("prob_anchor_swap", os.path.join(BASE, "script", "prob_anchor_swap.py"))

SEQ_L = pa.SEQ_L
M_SAMPLE = pa.M_SAMPLE


def _tertile(x, grp, min_n=3):
    """按 grp 分组做三分位分桶，返回桶号 [len(x)]，取值 0/1/2。"""
    b = np.zeros(len(x), np.int8)
    for g in np.unique(grp):
        m = grp == g
        if m.sum() < min_n:
            continue
        q1, q2 = np.quantile(x[m], [1 / 3, 2 / 3])
        b[m] = (x[m] > q1).astype(np.int8) + (x[m] > q2).astype(np.int8)
    return b


def run_city(city, ckpt_root, device):
    y0, t0 = pb.load(city)
    ok, trm, vam, tem, u, span = pb.split_rows(y0, t0)
    cols, smean = pb.live_cols(y0, trm)
    y = y0[:, cols]
    N = len(cols)
    hod = t0.hour.values
    wkd = (t0.dayofweek.values >= 5).astype(np.int8)

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

        # ---- 源起点：valid 段与 test 段，各自带 小时/周末/预测
        src = {}
        for nm, d, key in (('va', va_ds, 'va'), ('te', te_ds, 'te')):
            s = np.asarray(d.starts) + SEQ_L + j
            m = ok[s - lag]
            s = s[m]
            src[nm] = dict(s=s, obs=y[s], pred=pred_all[key][m][:, :, j],
                           hour=hod[s], wk=wkd[s])

        # ---- 全城预测幅度分桶（按小时三分位），valid/test 共用同一套阈值
        all_pred = np.concatenate([src['va']['pred'], src['te']['pred']], 0)
        all_hour = np.concatenate([src['va']['hour'], src['te']['hour']])
        amp = _tertile(all_pred.sum(1), all_hour)
        nva = len(src['va']['s'])
        src['va']['amp'], src['te']['amp'] = amp[:nva], amp[nva:]

        # ---- 因果池的语料 = valid + test 的**全部**源起点（靠 s_hist < s_i 筛）
        s_h = np.concatenate([src['va']['s'], src['te']['s']])
        res_h = np.concatenate([src['va']['obs'] - src['va']['pred'],
                                src['te']['obs'] - src['te']['pred']])
        hour_h = all_hour
        wk_h = np.concatenate([src['va']['wk'], src['te']['wk']])
        amp_h = amp

        te = src['te']
        P = len(te['s'])
        row = {'n_origin': int(P), 'point_mae': float(np.abs(te['pred'] - te['obs']).mean())}
        rng = np.random.default_rng(0)

        # 变体：(名字, 池来源, 用哪些条件)
        VARIANTS = [
            ('honest', 'va', ('hour',)),
            ('roll28', 'roll', ('hour',)),
            ('rollWA28', 'roll', ('hour', 'wk', 'amp')),
            ('oracleH', 'te', ('hour',)),
            ('oracleW', 'te', ('hour', 'wk')),
            ('oracleA', 'te', ('hour', 'amp')),
            ('oracleWA', 'te', ('hour', 'wk', 'amp')),
        ]

        for name, kind, conds in VARIANTS:
            per_st, nacc = np.zeros(N), 0
            pool_sz = []
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
                pool_sz.append(len(idx))
                pick = rng.choice(idx, min(M_SAMPLE, len(idx)), replace=True)
                smp = te['pred'][i][None, :] + pool_res[pick]      # [m,N] 整条向量
                cs = pb.crps_fair(smp, te['obs'][i])
                per_st += np.nan_to_num(cs)
                nacc += 1
            pm = per_st / max(nacc, 1) / smean
            row[name] = dict(
                crps_norm_sum=float(np.mean(per_st / max(nacc, 1)) / np.nanmean(smean)),
                crps_norm_med=float(np.nanmedian(pm)),
                n_pool_med=int(np.median(pool_sz)) if pool_sz else 0,
            )
        # 相对"不条件化"的作弊上限，条件化又往下走了多少
        base_c = row['oracleH']['crps_norm_sum']
        for k in ('oracleW', 'oracleA', 'oracleWA'):
            row[f'gain_{k}'] = (base_c - row[k]['crps_norm_sum']) / base_c
        row['remain_after_cond'] = ((row['rollWA28']['crps_norm_sum']
                                     - row['oracleWA']['crps_norm_sum'])
                                    / row['rollWA28']['crps_norm_sum'])
        stats[H] = row
        print(f"[{city} H={H}] honest {row['honest']['crps_norm_sum']:.4f}  "
              f"roll28 {row['roll28']['crps_norm_sum']:.4f}  "
              f"rollWA28 {row['rollWA28']['crps_norm_sum']:.4f}  |  "
              f"oracleH {row['oracleH']['crps_norm_sum']:.4f} "
              f"oracleW {row['oracleW']['crps_norm_sum']:.4f} "
              f"oracleA {row['oracleA']['crps_norm_sum']:.4f} "
              f"oracleWA {row['oracleWA']['crps_norm_sum']:.4f}  |  "
              f"条件化再降 {row['gain_oracleWA']:.1%}  剩 {row['remain_after_cond']:.1%}",
              flush=True)

    return {'city': city, 'n_station': N, 'span': span,
            'ckpt': os.path.relpath(ckpt, BASE), **stats}


def report_city(r):
    L = []
    P = L.append
    P("=" * 118)
    P(f"条件化探针  {r['city']}   站 {r['n_station']}   作弊版到底是不是上限？")
    P("  同一个 PatchTST 点预测、同一批起点、同一个目标时刻；唯一变量是残差池的**分组键**。")
    P("  honest   验证段·按小时           roll28    目标前28天·按小时（因果）")
    P("  rollWA28 目标前28天·按(小时,周末,幅度)（因果+条件）")
    P("  oracle*  测试期自己·作弊。H=按小时 / W=加周末 / A=加幅度 / WA=都加")
    P("  '条件化再降' = (oracleH - oracleWA)/oracleH，即原天花板被往下推了多少")
    P("=" * 118)
    for H in pb.HORIZONS:
        d = r[H]
        P(f"\n--- H = {H}（提前量 {H} 小时）---")
        P(f"{'口径':<11}{'crpsΣ':>9}{'crps中位':>10}{'池中位':>9}   说明")
        P("-" * 118)
        note = {'honest': '验证段·按小时（现状）', 'roll28': '目标前28天·按小时（因果）',
                'rollWA28': '目标前28天·按(小时,周末,幅度)（因果）',
                'oracleH': '测试期自己·按小时  ← 原"天花板"',
                'oracleW': '测试期自己·按(小时,周末)', 'oracleA': '测试期自己·按(小时,幅度)',
                'oracleWA': '测试期自己·按(小时,周末,幅度)'}
        for k in ('honest', 'roll28', 'rollWA28', 'oracleH', 'oracleW', 'oracleA', 'oracleWA'):
            v = d[k]
            P(f"{k:<11}{v['crps_norm_sum']:>9.4f}{v['crps_norm_med']:>10.4f}"
              f"{v['n_pool_med']:>9}{'   ' + note[k]}")
        P(f"点预测 MAE {d['point_mae']:.3f}   起点 {d['n_origin']}")
        P(f"* 条件化把天花板又往下推了 {d['gain_oracleWA']:.1%}"
          f"（W 单独 {d['gain_oracleW']:.1%}，A 单独 {d['gain_oracleA']:.1%}）")
        P(f"* 因果侧（rollWA28）到新天花板还差 {d['remain_after_cond']:.1%}")
    P("\n读法：")
    P("  * oracleW/A/WA 低于 oracleH → 原天花板是假的，条件化是真空间。")
    P("  * 但这仍只是「周末 / 幅度」两个条件；条件在输入本身上还能更低 → 仍是偏松的上限。")
    P("  * 池中位列：条件越细池越小，bootstrap 噪声越大，看这个数判断数字是否可信。")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cities', nargs='*', default=pb.CITIES)
    ap.add_argument('--ckpt_dir', default=f"{BASE}/result/univariate")
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    done = []
    for c in args.cities:
        r = run_city(c, args.ckpt_dir, device)
        json.dump(r, open(f"{OUT}/metrics_{c}.json", "w"), ensure_ascii=False,
                  indent=1, default=str)
        txt = report_city(r)
        open(f"{OUT}/report_{c}.txt", "w").write(txt + "\n")
        done.append(r)

    if len(done) < 2:
        print(f"\n单城模式：已保存 {OUT}/report_{done[0]['city']}.txt（不写汇总）")
        return

    alltxt = [
        "=" * 118,
        "条件化探针汇总（5 城平均，AMS 单列）",
        "  oracleH = 原「作弊上限」（只按站+小时）   oracleWA = 再加(周末,幅度)条件",
        "=" * 118,
        f"{'H':>4}{'honest':>10}{'roll28':>10}{'rollWA28':>10}"
        f"{'oracleH':>10}{'oracleW':>10}{'oracleA':>10}{'oracleWA':>10}"
        f"{'再降':>9}{'剩':>8}",
    ]
    for H in pb.HORIZONS:
        rs = [r for r in done if r['city'] != 'AMS']
        g = lambda k: float(np.mean([r[H][k]['crps_norm_sum'] for r in rs]))
        gg = lambda k: float(np.mean([r[H][k] for r in rs]))
        alltxt.append(f"{H:>4}{g('honest'):>10.4f}{g('roll28'):>10.4f}{g('rollWA28'):>10.4f}"
                      f"{g('oracleH'):>10.4f}{g('oracleW'):>10.4f}{g('oracleA'):>10.4f}"
                      f"{g('oracleWA'):>10.4f}{gg('gain_oracleWA'):>9.1%}"
                      f"{gg('remain_after_cond'):>8.1%}")
    for r in done:
        if r['city'] == 'AMS':
            alltxt.append("\nAMS（单列，不进汇总）：")
            for H in pb.HORIZONS:
                d = r[H]
                alltxt.append(f"  H={H:<3} oracleH {d['oracleH']['crps_norm_sum']:.4f} "
                              f"-> oracleWA {d['oracleWA']['crps_norm_sum']:.4f} "
                              f"条件化再降 {d['gain_oracleWA']:.1%}")
    txt = "\n".join(alltxt)
    open(f"{OUT}/report.txt", "w").write(txt + "\n")
    print("\n" + txt)
    print(f"\n已保存 {OUT}/")


if __name__ == "__main__":
    main()
