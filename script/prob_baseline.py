# -*- coding: utf-8 -*-
"""概率预测基线 + CRPS_sum 评测脚手架。

方法
----
点预测：seasonal naive, m=24，Hyndman 形式。对目标时刻 s、步长 h：
    lag = 24 * ceil(h / 24)
    ŷ(s) = y(s - lag)
h=24 时 ŷ = y(s-24)（"明天此刻=今天此刻"）；h=48 时 ŷ = y(s-48)，锚点离目标
更远 -> 误差更大。这正是 horizon 梯度的来源（见 prob_profiling.py）。

预测分布：**残差场 bootstrap**
    训练期每个起点 t'、每个 h，取整条残差*向量*
        r(t') = y(t'+h) - y(锚点)      in R^{站点}
    测试时对起点 t，抽 m 个"目标小时相同"的训练起点 t'，把整条残差向量
    加到点预测上：sample_k = ŷ(t) + r(t'_k)

为什么不用逐点经验分位数
------------------------
CRPS_sum 先把同一时刻所有站的值*加起来*再算 CRPS，因此要求每个样本是一次
完整的"场景"——第 k 个样本在所有站上的取值必须同源。若逐站独立取分位数，
求和的涨落互相抵消，聚合方差系统性偏小，**CRPS_sum 会虚低**，正是这个指标
要惩罚的"忽略站间关系"被掩盖了。残差场 bootstrap 每次抽一整个向量，站间
结构原样保留。

参照基线
--------
  point  : 点预测本身（CRPS 退化为 MAE）——能力下界
  analog : 历史同小时整行——完全不学的参照
  main   : seasonal naive + 残差场 bootstrap —— 主基线

用法：python3 script/prob_baseline.py
输出：result/prob_baseline/report.txt, metrics.json
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITIES = ["LOA", "MEL", "JHB", "AMS", "SPO", "SZH"]
FEATURE = "volume"
HORIZONS = [24, 48, 72]
LAG = 24
M_SAMPLE = 200            # 每次预测抽多少个残差场
# 划分比例：train 2 / valid 1 / test 3。**停摆丢弃后**在剩余连续可用时间上切。
TRAIN_FRAC = 2 / 6
VALID_FRAC = 1 / 6
OUTAGE_FRAC = 0.8         # >80% 站同时为零 => 整城停摆，置 NaN
MIN_SPAN = 96             # seq_l 72 + pre_len 24，短于此的碎片切不出窗口
# 训练期退化站的判据（见 live_cols）。两边（这里与 api/dataset/common.py）必须一致。
DEGEN_REL = 0.01          # 训练期均值 < 全城活跃站中位数的 1%  => 死站
DEGEN_CV = 0.01           # 训练期 std / 训练期均值 < 1%        => 常数平台
OUT = f"{BASE}/result/prob_baseline"
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------ 载入
def load(city):
    v = pd.read_csv(f"{BASE}/data/{city}_remove_const/{FEATURE}.csv", index_col=0)
    v.index = pd.to_datetime(v.index)
    y = v.values.astype(np.float64)
    y[(y == 0).mean(1) > OUTAGE_FRAC] = np.nan       # 停摆屏蔽
    return y, v.index


# ------------------------------------------------------------- CRPS
def crps_fair(samples, obs):
    """公平 CRPS（无偏样本估计量）。

        CRPS = (1/m)Σ|x_i-y| - 1/(m(m-1)) Σ_i (2i-m-1) x_(i)   [x 已排序]

    samples: [m] 或 [m, N]；obs: 标量或 [N]。返回与 obs 同形。

    ``m == 1`` 时修正项的系数是 ``1/(m(m-1)) = 1/0``，原式直接产生 NaN（曾经在
    oracleWA 那种强条件池里真的出现过）。此时退化为**插入式**估计 ``|x-y|`` ——
    这正是"点质量分布"的 CRPS，是该极限下的正确值，不是补丁。
    """
    m = samples.shape[0]
    t1 = np.abs(samples - obs).mean(0)
    if m < 2:
        return t1
    s = np.sort(samples, axis=0)
    shape = [m] + [1] * (s.ndim - 1)
    w = ((2 * np.arange(1, m + 1) - m - 1) / (m * (m - 1))).reshape(shape)
    return t1 - (w * s).sum(0)


def pit_metrics(samples, obs, levels=(0.5, 0.8, 0.9)):
    """PICP + MPIW。levels=0.5 用 [q25,q75]，0.8 用 [q10,q90]，0.9 用 [q5,q95]。"""
    out = {}
    for lv in levels:
        lo = np.percentile(samples, (1 - lv) / 2 * 100, axis=0)
        hi = np.percentile(samples, (1 + lv) / 2 * 100, axis=0)
        out[f"picp{int(lv*100)}"] = float(np.mean((obs >= lo) & (obs <= hi)))
        out[f"mpiw{int(lv*100)}"] = float(np.mean(hi - lo))
    return out


# ------------------------------------------------------------- 划分（对外复用）
def split_rows(y0, t0):
    """停摆丢弃 + 剩余可用行按计数切 2/1/3。

    返回 ``(ok, trm, vam, tem, u, span)``：

    * ``ok``  : [T] 该行可用（停摆行已被 :func:`load` 置 NaN）；
    * ``trm`` / ``vam`` / ``tem`` : [T] 三个 split 的行掩码；
    * ``u``   : 可用行的全局下标（时间序），``u[:i1]`` 是 train；
    * ``span``: 三段的起止时间戳，便于打印/对照。

    **模型侧 ``api/dataset/common.py:split_cross_validation`` 必须与此逐行一致**
    （``script/check_split_alignment.py`` 是这条不变量的回归测试）。
    """
    T_all = y0.shape[0]
    ok = np.isfinite(y0).all(1)                          # [T_all] 该行可用
    u = np.where(ok)[0]
    U = len(u)
    if U < MIN_SPAN:
        raise RuntimeError(f"可用行只有 {U} < {MIN_SPAN}")
    i1 = int(U * TRAIN_FRAC)
    i2 = i1 + int(U * VALID_FRAC)
    trm = np.zeros(T_all, bool); trm[u[:i1]] = True
    vam = np.zeros(T_all, bool); vam[u[i1:i2]] = True
    tem = np.zeros(T_all, bool); tem[u[i2:]] = True
    span = dict(drop_h=T_all - U, n_u=U,
                tr=(t0[u[0]], t0[u[i1 - 1]]),
                va=(t0[u[i1]], t0[u[i2 - 1]]),
                te=(t0[u[i2]], t0[u[-1]]))
    return ok, trm, vam, tem, u, span


def live_cols(y0, trm):
    """剔掉**训练期退化**的站，返回 ``(cols, smean)``。

    只用**训练期**统计量判定（碰 valid/test 就是拿评测数据选站 = 泄漏）。
    两类垃圾站，共同点是「训练期的统计量根本描述不了这个站自己」：

    1. **死站**：训练期均值 < 全城活跃站中位数的 ``DEGEN_REL``。归一化分母
       ``CRPS_j/smean_j`` 会爆炸——实测 JHB 曾出现 31 站里 1 站贡献 80.77、
       而全城中位仅 0.269，整个城的指标被它劫持。
    2. **常数平台**：训练期波动 / 训练期均值 < ``DEGEN_CV``，之后才活。
       StandardScaler 的分母(scale)因此极小，测试期的正常偏差被放大上百倍：
       MEL 站 57 训练期恒为 200.00±1.07，一个站就吃掉全城 82% 的验证损失
       （JHB 站 22 占 95%），early-stop / 选点完全被它决定。

    模型侧 ``api/dataset/common.py:split_cross_validation`` 用同一口径同一阈值
    （常量 ``DEGEN_REL``/``DEGEN_CV`` 两边各有一份，``script/check_split_alignment.py``
    断言两边站集逐列相同）。改动这里必须同步改那边，否则站集不同、数字不能同表。
    """
    tr = y0[trm]                                         # 停摆行是 NaN，自动跳过
    smean = np.nanmean(tr, 0)
    sstd = np.nanstd(tr, 0)
    live = np.isfinite(smean) & np.isfinite(sstd) & (smean > 0)
    med = np.median(smean[live]) if live.any() else 0.0
    live &= (smean >= DEGEN_REL * med) & (sstd >= DEGEN_CV * np.maximum(smean, 1e-12))
    cols = np.where(live)[0]
    return cols, smean[cols]


# ------------------------------------------------------------- 主流程
def run_city(city):
    y0, t0 = load(city)
    T_all, N0 = y0.shape
    hod = t0.hour.values

    # ---- 划分：停摆行（>80% 站同时为零）已在 load() 里置 NaN，这里直接丢弃。
    # 剩下的可用行按**时间先后计数**切 2/1/3（不是按日历天，停摆那几周不占额度）。
    # 窗口归属只看**目标时刻** t+H 落在哪一段；锚点只要有效即可（可以落在更早的
    # 一段——"24h 前"本来就该往前看）。跨停摆缺口、锚点无效的窗口一律排除。
    ok, trm, vam, tem, u, span = split_rows(y0, t0)

    # 训练段全零/全 NaN 的站，归一化分母无意义，整列剔除。
    # 注：残差池**不需要**逐站归一化——抽样抽的是整条向量，站 j 只会拿到
    #     站 j 自己的残差，不存在跨站宽度泄漏。（先前写过的"先除 smean 再乘
    #     smean"是恒等变换，已删。）
    cols, smean = live_cols(y0, trm)
    y = y0[:, cols]
    N = len(cols)

    # 训练段 vs 测试段的**量级比**：归一化用的是训练段均值，若两段水平差很多
    # （AMS 1.53、SPO 1.25），比值会被系统性抬高/压低，跨城比较时必须知道。
    lvl = np.nansum(np.nanmean(y0[tem], 0)) / np.nansum(smean)

    npar = np.arange(N)                                  # 列索引，配合 fancy indexing
    res = {"city": city, "n_station": N, "n_dropped": N0 - N,
           "T_all": T_all, "n_u": span["n_u"], "n_train_row": int(trm.sum()),
           "n_test_row": int(tem.sum()), "lvl_ratio": float(lvl), "span": span}
    for H in HORIZONS:
        lag = LAG * int(np.ceil(H / LAG))

        # ---- 训练池：目标 s=t'+H 落在 train 段、锚点 a=t'+H-lag 有效
        tv = np.arange(0, T_all - H)
        s_idx, a_idx = tv + H, tv + H - lag
        m = (a_idx >= 0) & trm[s_idx] & ok[a_idx]
        tv, s_idx, a_idx = tv[m], s_idx[m], a_idx[m]
        pool_r = y[s_idx] - y[a_idx]                     # [n_pool, N] 整条残差向量
        pool_hod = hod[s_idx]                            # 按目标小时分层

        # ---- 测试起点：目标落在 test 段、锚点有效
        te = np.arange(0, T_all - H)
        ts, ta = te + H, te + H - lag
        m = (ta >= 0) & tem[ts] & ok[ta]
        te, obs, fc = te[m], y[ts[m]], y[ta[m]]

        # ---- 逐起点抽样
        rng = np.random.default_rng(0)
        crps_marg, crps_sum, norm_marg = [], [], []
        pit = {k: [] for k in ("picp50", "mpiw50", "picp80", "mpiw80",
                               "picp90", "mpiw90")}
        crps_point, crps_analog, crps_marg_ref = [], [], []
        crps_marg_ind, crps_sum_ind, crps_sum_marg = [], [], []
        per_st, nacc = np.zeros(N), 0                    # 逐站 CRPS 累加

        # 参照池（都是"训练期同小时"，区别在于是整行还是逐站）：
        #   marg[hh]  : 该小时的整行 [n,N] —— 列 j 拿到的是"全城所有站"在该小时的
        #               取值。含义 = "知道小时、不知道站"的参照。别名叫 analog 会
        #               误导（站间规模差几个数量级时它天然很烂，AMS 高达 7.9）。
        #   col[hh]   : 逐站压缩 [nmax,N] —— 列 j 顶部是站 j 自己在该小时的有限值，
        #               n_j 为有效个数。这才是季节朴素点预测的正面参照：
        #               "知道站、知道小时，但不知道是哪一天"。
        marg_pool, col_pool, col_n = {}, {}, {}
        tr_rows, tr_hod = y[trm], hod[trm]
        for hh in range(24):
            rows = tr_rows[tr_hod == hh]
            fin = np.isfinite(rows)
            marg_pool[hh] = rows[fin.all(1)]
            # 按 ~fin 稳定排序 -> 有限值被顶到各列上方
            order = np.argsort(~fin, axis=0, kind="stable")
            C = np.take_along_axis(rows, order, 0)
            C[~np.take_along_axis(fin, order, 0)] = np.nan
            col_pool[hh], col_n[hh] = C, fin.sum(0)

        for i in range(len(te)):
            hh = hod[te[i] + H]
            idx = np.where(pool_hod == hh)[0]
            if len(idx) == 0:
                continue
            pick = rng.choice(idx, min(M_SAMPLE, len(idx)), replace=True)
            # 整条残差向量直接相加 -> [m, N] 整场样本（站间同源结构保留）
            smp = fc[i][None, :] + pool_r[pick]

            # ---- 独立采样对照：边际完全一样，只把站间同源关系拆散 ----
            # 每个样本的每一列各自独立地在该小时的池里重抽。边际分布因此与
            # main 逐列同分布（crps_marg_ind ≈ crps_marg 是自检），但聚合量的
            # 方差被压小 -> crps_sum_ind 会虚低。两者之比 = "忽略站间关系"的代价。
            sub = pool_r[idx]
            sel = rng.integers(0, len(sub), size=(M_SAMPLE, N))
            smp_ind = fc[i][None, :] + sub[sel, npar]
            crps_marg_ind.append(crps_fair(smp_ind, obs[i]).mean())
            crps_sum_ind.append(float(crps_fair(smp_ind.sum(1), obs[i].sum())))

            cs = crps_fair(smp, obs[i])                  # [N] 逐站 CRPS
            per_st += np.nan_to_num(cs)                  # 逐站累加，供稳健归一化
            nacc += 1
            crps_marg.append(cs.mean())
            norm_marg.append(np.nanmean(cs / smean))
            # CRPS_sum：先跨站求和（保持同源），再算 CRPS
            crps_sum.append(float(crps_fair(smp.sum(1), obs[i].sum())))
            crps_point.append(float(np.abs(fc[i] - obs[i]).mean()))
            ap = marg_pool[hh]
            if len(ap):
                asel = ap[rng.choice(len(ap), M_SAMPLE, replace=True)]
                crps_marg_ref.append(float(crps_fair(asel, obs[i]).mean()))
                # 整行抽样本身**保留**了该小时的经验站间结构 -> 无学习但联合的
                # CRPS_sum 参照。注意：analog 与 marg 的**边际**逐列同分布
                # （analog 只是在同一批值上重新打乱），所以它们的边际 CRPS 必然
                # 相等，唯一区别在 CRPS_sum 上。
                crps_sum_marg.append(float(crps_fair(asel.sum(1), obs[i].sum())))
            # 逐站 analog：列 j 只在站 j 自己的同小时值里重采样
            C, nj = col_pool[hh], col_n[hh]
            okj = np.where(nj >= 2)[0]                   # 公平 CRPS 需要 >=2 个样本
            if len(okj):
                sel = rng.integers(0, nj[okj], size=(M_SAMPLE, len(okj)))
                crps_analog.append(float(crps_fair(C[sel, okj[None, :]],
                                                   obs[i][okj]).mean()))
            for k, v in pit_metrics(smp, obs[i]).items():
                pit[k].append(v)

        # ---- 三种归一化。mean-of-ratios 会被"训练均值≈0 的近死站"劫持
        # （CRPS_j/smean_j 分母极小 -> 单站可把全城均值抬高 10 倍，实测 JHB
        # 31 站里 1 站贡献 80.77 / 中位 0.269）。故以**和中位**为准。
        ps = per_st / max(nacc, 1) / smean              # 逐站 CRPS均值 / 训练均值
        res[H] = dict(
            n_origin=len(crps_marg),
            crps_norm_mean=float(np.nanmean(ps)),        # 旧口径，留作对照
            crps_norm_med=float(np.nanmedian(ps)),       # 稳健：逐站比值的中位
            crps_norm_sum=float(np.mean(per_st / max(nacc, 1)) / np.nanmean(smean)),
            n_thin=int(np.sum(smean < 1.0)),             # smean<1 的站数（分母病态）
            crps_marg=float(np.mean(crps_marg)),
            crps_sum=float(np.mean(crps_sum)),
            crps_sum_norm=float(np.mean(crps_sum) / smean.sum()),
            crps_marg_norm=float(np.mean(norm_marg)),
            crps_point=float(np.mean(crps_point)),
            crps_analog=float(np.mean(crps_analog)) if crps_analog else np.nan,
            crps_marg_ref=float(np.mean(crps_marg_ref)) if crps_marg_ref else np.nan,
            # 独立采样对照（边际同分布，站间关系被拆散）
            crps_marg_ind=float(np.mean(crps_marg_ind)) if crps_marg_ind else np.nan,
            crps_sum_ind=float(np.mean(crps_sum_ind)) if crps_sum_ind else np.nan,
            crps_sum_marg=float(np.mean(crps_sum_marg)) if crps_sum_marg else np.nan,
            **{k: float(np.mean(v)) for k, v in pit.items()},
        )
        print(f"[{city} H={H}] 完成 {len(crps_marg)} 个起点", flush=True)
    return res


def main():
    allres = []
    for c in CITIES:
        allres.append(run_city(c))
        json.dump(allres, open(f"{OUT}/metrics.json", "w"), ensure_ascii=False,
                  indent=1, default=str)          # span 里有 Timestamp

    lines = []
    P = lines.append
    P("=" * 100)
    P("概率预测基线  CRPS / CRPS_sum / PICP / MPIW")
    P("  point  = 点预测本身(退化分布的 CRPS，即 MAE；点预测很好时可小于 main，不是下界)")
    P("  analog = 逐站同小时历史值(知道站+小时，不知哪一天)   main = seasonal naive m=24 + 残差场 bootstrap")
    P("  marg   = 同小时整行(知道小时、不知站；站间规模差异大时天然很烂)")
    P("  crpsΣ  = Σ_j CRPS_j / Σ_j smean_j（按站规模加权；跨城可比，**主看这个**）")
    P("  crps中位 = 逐站 CRPS_j/smean_j 的中位数（稳健；与 crpsΣ 同向才可信）")
    P("  （旧的 crps_norm=均值(CRPS_j/smean_j) 已被近死站劫持，见文末说明，不再作头条）")
    P("  crps_sum_norm = CRPS_sum 除以全城训练期均值之和（跨城可比）")
    P("  残差池：训练期残差 y(锚点+H)-y(锚点)，按目标小时分层，整条向量抽样 -> CRPS_sum 不被拆散")
    P("  划分：停摆行(>80%站同时为零)直接丢弃，剩余可用行按计数切 train2/valid1/test3")
    P("=" * 100)

    # ---- 划分边界（与 horizon 无关，只打一次）
    P("\n--- 时间划分：停摆丢弃后在剩余可用时间上按计数切 2/1/3 ---")
    P(f"{'城':<5}{'总长':>6}{'丢停摆':>8}{'可用':>7}"
      f"{'train':>19}{'valid':>19}{'test':>19}{'train行':>8}{'test行':>8}{'测/训量级':>10}")
    P("-" * 116)
    for r in allres:
        s = r["span"]
        f = lambda a, b: (f"{a.strftime('%m-%d')}~{b.strftime('%m-%d')}")
        P(f"{r['city']:<5}{r['T_all']:>6}{s['drop_h']:>8}{s['n_u']:>7}"
          f"{f(*s['tr']):>19}{f(*s['va']):>19}{f(*s['te']):>19}"
          f"{r['n_train_row']:>8}{r['n_test_row']:>8}{r['lvl_ratio']:>10.2f}")
    P("\n  '测/训量级' = 测试期各站均值之和 / 训练期各站均值之和。偏离 1 越远，用训练均值做分母的")
    P("  归一化指标就越失真（AMS 1.53、SPO 1.25；其余 0.87~1.03）。")
    P(f"\n  停摆行 = 该小时 >80% 的站同时为零；可用行按时间先后计数切 2/1/3（停摆几周不占额度）。")
    P(f"  窗口归属只看目标时刻 t+H 落在哪一段；锚点只需有效，跨停摆缺口的窗口一律排除。")

    for H in HORIZONS:
        P(f"\n--- pre_len = {H} ---")
        P(f"{'城':<5}{'站数':>6}{'起点':>7}{'crpsΣ':>9}{'crps中位':>10}{'crpssum_n':>10}{'crps':>9}"
          f"{'point':>8}{'analog':>8}{'死站':>6}"
          f"{'PICP50':>8}{'PICP80':>8}{'PICP90':>8}{'MPIW50':>9}")
        P("-" * 108)
        for r in allres:
            d = r[H]
            P(f"{r['city']:<5}{r['n_station']:>6}{d['n_origin']:>7}"
              f"{d['crps_norm_sum']:>9.4f}{d['crps_norm_med']:>10.4f}{d['crps_sum_norm']:>10.4f}"
              f"{d['crps_marg']:>9.2f}"
              f"{d['crps_point']:>8.2f}{d['crps_analog']:>8.2f}{d['n_thin']:>6}"
              f"{d['picp50']:>8.1%}{d['picp80']:>8.1%}{d['picp90']:>8.1%}{d['mpiw50']:>9.2f}")
    P("\n  '死站' = 训练期均值 smean<1 的站数；这些站的分母病态，是旧 crps_norm 爆炸的来源。")

    # ---- 联合 vs 独立采样：CRPS_sum 的"站间关系"代价 ----
    P("\n" + "=" * 105)
    P("CRPS_sum 三参照：main(联合·有学习) / marg(联合·无学习) / indep(独立·拆散站间关系)")
    P("  自检：main 与 indep 的 crps(边际) 应几乎相等（两者边际逐列同分布）；不等则实现有误。")
    P(f"{'城':<6}{'pre':>5}{'crps主':>9}{'crps独':>9}"
      f"{'sum_marg':>11}{'sum_main':>11}{'sum_indep':>11}{'main/indep':>12}{'main/marg':>11}")
    P("-" * 105)
    for H in HORIZONS:
        for r in allres:
            d = r[H]
            ri = d["crps_sum"] / d["crps_sum_ind"] if d["crps_sum_ind"] else np.nan
            rm = d["crps_sum"] / d["crps_sum_marg"] if d["crps_sum_marg"] else np.nan
            P(f"{r['city']:<6}{H:>5}{d['crps_marg']:>9.2f}{d['crps_marg_ind']:>9.2f}"
              f"{d['crps_sum_marg']:>11.1f}{d['crps_sum']:>11.1f}{d['crps_sum_ind']:>11.1f}"
              f"{ri:>12.2f}{rm:>11.2f}")
    P("\n读法：indep 的边际与 main 相同，只把站间同源关系拆散 -> 聚合方差被压小。")
    P("      'main/indep' 就是'忽略站间关系'在 CRPS_sum 上的代价（>>1 = 关系重要）。")
    P("      marg 是'无学习但联合'的参照：main/marg <1 才说明残差场 bootstrap 真的有用。")
    txt = "\n".join(lines)
    print("\n" + txt)
    open(f"{OUT}/report.txt", "w").write(txt)
    print(f"\n已保存 {OUT}/report.txt  和  metrics.json")


if __name__ == "__main__":
    main()
