# -*- coding: utf-8 -*-
"""诊断：AMS 的 PICP50=90.4% 过度覆盖从哪来。

背景
----
`prob_baseline.py` 的主基线（seasonal naive m=24 + 残差场 bootstrap）在 AMS 上
出现明显过度离散：crps 0.41 > point(MAE) 0.32，PICP50 = 90.4%（名义 50%）。
先前的"逐站归一化"修法是恒等变换（同一 smean 先除后乘），且没必要——残差
是整条**向量**抽样的，站 j 只会拿到站 j 自己的残差，不存在跨站宽度泄漏。

所以真正的问题不是"池子混了站"，而是：**池子里的残差分布 ≠ 测试期的残差分布**。
本脚本量化两件事：

  Q1 池 vs 测试的残差宽度直接对比（逐站 IQR / MAD 比值）
  Q2 残差分布是否**状态依赖**：站是否处在"≥24h 恒定平台"内，是观测得到的历史
     状态（因果可用）。若平台态下残差窄、非平台态下残差宽，而池子把两者混合，
     则季节朴素用无条件池 -> 平台态被过宽覆盖 -> 正是 AMS 的症状（AMS 64.9%
     的站-小时在平台里）。

用法：python3 script/prob_baseline_diag.py
"""
import numpy as np
import pandas as pd

BASE = "/home/data/yzzpz/Charged"
CITIES = ["AMS", "SZH", "MEL", "JHB"]
FEATURE = "volume"
H = 24
LAG = 24
TRAIN_FRAC = 0.7
OUTAGE_FRAC = 0.8
PLATEAU_LEN = 24


def load(city):
    v = pd.read_csv(f"{BASE}/data/{city}_remove_const/{FEATURE}.csv", index_col=0)
    v.index = pd.to_datetime(v.index)
    y = v.values.astype(np.float64)
    y[(y == 0).mean(1) > OUTAGE_FRAC] = np.nan
    return y, v.index


def runlen_ending(y):
    """每个 (时刻, 站) 处，往前连续相等的长度（仅有限值）。"""
    T, N = y.shape
    rl = np.ones((T, N), dtype=np.int32)
    fin = np.isfinite(y)
    for i in range(1, T):
        same = (y[i] == y[i - 1]) & fin[i] & fin[i - 1]
        rl[i] = np.where(same, rl[i - 1] + 1, 1)
    rl[~fin] = 0
    return rl


def run(city):
    y, t = load(city)
    T, N0 = y.shape
    ntr = int(T * TRAIN_FRAC)
    hod = t.hour.values
    smean = np.nanmean(y[:ntr], 0)
    cols = np.where(np.isfinite(smean) & (smean > 0))[0]
    y = y[:, cols]
    N = len(cols)
    rl = runlen_ending(y)
    plat = rl >= PLATEAU_LEN                     # [T, N] 因果可算

    lag = LAG * int(np.ceil(H / LAG))

    # ---- 训练池
    tv = np.arange(0, ntr - H)
    s_idx, a_idx = tv + H, tv + H - lag
    ok = a_idx >= 0
    tv, s_idx, a_idx = tv[ok], s_idx[ok], a_idx[ok]
    keep = np.isfinite(y[s_idx]).all(1) & np.isfinite(y[a_idx]).all(1)
    tv, s_idx, a_idx = tv[keep], s_idx[keep], a_idx[keep]
    pool_r = y[s_idx] - y[a_idx]                  # [P, N]
    pool_plat = plat[a_idx]                       # [P, N] 锚点处是否在平台里（因果）

    # ---- 测试
    te = np.arange(ntr, T - H)
    obs, fc = y[te + H], y[te + H - lag]
    valid = np.isfinite(obs).all(1) & np.isfinite(fc).all(1)
    te, obs, fc = te[valid], obs[valid], fc[valid]
    res = obs - fc                                # [Q, N] 真实残差
    te_plat = plat[te]                            # [Q, N] 测试起点锚点是否在平台里

    # ---- Q1 宽度对比（逐站）
    def iqr(x, ax=0):
        return np.nanpercentile(x, 75, axis=ax) - np.nanpercentile(x, 25, axis=ax)

    pool_iqr, test_iqr = iqr(pool_r), iqr(res)
    pool_mad = np.nanmedian(np.abs(pool_r - np.nanmedian(pool_r, 0)), 0)
    test_mad = np.nanmedian(np.abs(res - np.nanmedian(res, 0)), 0)
    okj = np.isfinite(pool_iqr) & np.isfinite(test_iqr) & (test_iqr > 1e-9)
    r_iqr = pool_iqr[okj] / test_iqr[okj]

    # ---- Q2 覆盖：逐站用池的 [q25,q75] 当 50% 区间（不抽样，等价于 m->inf）
    q25 = np.nanpercentile(pool_r, 25, 0)
    q75 = np.nanpercentile(pool_r, 75, 0)
    fin = np.isfinite(res)                        # [Q, N]
    cov = ((res >= q25) & (res <= q75))[fin]      # 只用有限的站-起点对
    pmask = te_plat[fin]

    def cond_quant(st, mask_src):
        """池限定在 mask_src==st 的起点，返回逐站 [q25,q75]。"""
        q = np.full((2, N), np.nan)
        for j in range(N):
            sel = pool_r[mask_src[:, j] == st, j]
            if len(sel) > 20:
                q[:, j] = np.percentile(sel, [25, 75])
        return q

    # 锚点态条件（锚点状态因果可知）
    out = {}
    for st, name in ((True, "锚点平台"), (False, "锚点非平台")):
        q = cond_quant(st, pool_plat)
        ccov = ((res >= q[0]) & (res <= q[1]))[fin]
        m = pmask == st
        out[name] = (float(m.mean()), float(ccov[m].mean()) if m.sum() else np.nan)

    # 目标态条件（oracle：展望 t+H 处的状态，现实不可知，只作"若有完美状态
    # 转移预测器能到多少"的上界）
    pool_plat_tgt, tgt_plat = plat[s_idx], plat[te + H]
    out_t, qT = {}, {}
    for st, name in ((True, "目标平台"), (False, "目标非平台")):
        q = cond_quant(st, pool_plat_tgt)
        qT[name] = q
        ccov = ((res >= q[0]) & (res <= q[1]))[fin]
        m = (tgt_plat[fin] == st)
        out_t[name] = (float(m.mean()), float(ccov[m].mean()) if m.sum() else np.nan)
    # 逐对选对应状态的池 -> 完美状态知识下的覆盖率
    lo = np.where(tgt_plat, qT["目标平台"][0], qT["目标非平台"][0])
    hi = np.where(tgt_plat, qT["目标平台"][1], qT["目标非平台"][1])
    cov_oracle = float(((res >= lo) & (res <= hi))[fin].mean())

    # ---- Q3 池残差宽度的时间轨迹（把池按起点时间切四段）+ 逐站配对对比
    edges = np.linspace(0, len(pool_r), 5).astype(int)
    seg = []
    for k in range(4):
        blk, idx_blk = pool_r[edges[k]:edges[k + 1]], tv[edges[k]:edges[k + 1]]
        seg.append((int(idx_blk[0]), int(idx_blk[-1]),
                    float(np.nanmedian(iqr(blk)))))
    pairs = np.isfinite(pool_iqr) & np.isfinite(test_iqr)
    worse = float((test_iqr[pairs] < pool_iqr[pairs]).mean())

    # ---- Q4 "常数站"占比：IQR==0 <=> 该站至少一半时间不动。
    # Q1 的 okj 过滤（只留 test_iqr>1e-9 的站）恰好把不动站全剔了，会系统性
    # 抬高池 IQR 的中位数。这里把"全部站"与"okj 站"两个中位数并列，看差异有多大。
    zp = lambda v: float(np.nanmean(v <= 1e-9))
    const_pool, const_test = zp(pool_iqr), zp(test_iqr)

    # ---- Q5 聚合方差：联合(经验) vs 独立(Σ各站方差)
    # 池每行都是全有限（keep 要求 all-finite），可直接算。
    # var_ratio = Var(Σ_j r_j) / Σ_j Var(r_j) = 1 + (N-1)·ρ̄  —— 站间同源关系把
    # 聚合不确定性放大多少倍。这直接预测 CRPS_sum 的"联合/独立"比值：
    # ratio≈1 => 站间关系对聚合量几乎无影响。
    # 必须**按目标小时分层**再算——抽样器就是按小时分层的，混着算会把日周期
    # 这个最大的共同因子算进去，虚高。逐小时算方差再平均。
    pool_hod = hod[s_idx]
    vj_l, vi_l = [], []
    for hh in range(24):
        sel = pool_hod == hh
        if sel.sum() < 5:
            continue
        Rh = pool_r[sel] - pool_r[sel].mean(0)
        vj_l.append(float(np.var(Rh.sum(1))))
        vi_l.append(float(np.sum(np.var(Rh, 0))))
    v_joint, v_indep =float(np.mean(vj_l)), float(np.mean(vi_l))
    var_ratio = v_joint / v_indep if v_indep > 0 else np.nan
    rho_bar = (var_ratio - 1.0) / (N - 1) if N > 1 else np.nan

    # ---- Q6 聚合残差 Σ_j r_j 的池 vs 测试：标准差与偏置
    # 池的聚合 std 就是联合采样给聚合量的宽度。若测试期的聚合 std 远小于它，
    # 则**任何**基于池的采样器（含联合）对聚合量都过宽，联合采样会因过宽而
    # 受罚，反而让"恰好更窄"的独立采样打平 —— 这解释了 main/indep≈1。
    agg_pool, agg_test = pool_r.sum(1), res.sum(1)
    std_agg_pool = float(np.std(agg_pool))
    std_agg_test = float(np.std(agg_test))
    bias_agg_test = float(np.mean(agg_test))
    # 聚合点预测的 MAE：mean|Σy - Σŷ|。若它 ≈ 实测 CRPS_sum，说明 CRPS_sum 几乎
    # 全由点预测的聚合误差决定，分布（以及联合/独立的差别）贡献可忽略。
    agg_point_mae = float(np.mean(np.abs(agg_test)))
    # 相对量：除以全城均值之和，跨城可比
    scale = float(np.nansum(np.nanmean(y[:ntr], 0)))
    agg_ratio = std_agg_pool / std_agg_test if std_agg_test > 0 else np.nan

    return dict(city=city, N=N, n_pool=len(tv), n_test=len(te),
                plat_frac_test=float(pmask.mean()),
                pool_iqr_med=float(np.nanmedian(pool_iqr[okj])),
                test_iqr_med=float(np.nanmedian(test_iqr[okj])),
                ratio_med=float(np.nanmedian(r_iqr)),
                ratio_p90=float(np.nanpercentile(r_iqr, 90)),
                cov_uncond=float(cov.mean()),
                plat_frac_pool=float(pool_plat.mean()),
                cov_anchor=out,
                cov_tgt=out_t,
                cov_oracle=cov_oracle,
                tgt_plat_frac_test=float(tgt_plat[fin].mean()),
                seg=seg, frac_test_narrower=worse,
                const_pool=const_pool, const_test=const_test,
                var_ratio=var_ratio, rho_bar=rho_bar,
                std_agg_pool=std_agg_pool, std_agg_test=std_agg_test,
                agg_ratio=agg_ratio, bias_agg_test=bias_agg_test, scale=scale,
                agg_point_mae=agg_point_mae,
                test_iqr=test_iqr, pool_iqr=pool_iqr,
                t0=float(t[0].value), dt_ns=float((t[1] - t[0]).value))


if __name__ == "__main__":
    rows = [run(c) for c in CITIES]
    print("\n" + "=" * 108)
    print(f"Q1 池残差 vs 测试残差宽度（H={H}，逐站 IQR）")
    print(f"{'城':<6}{'站':>6}{'池样本':>8}{'测试':>7}{'池IQR中位':>11}{'测IQR中位':>11}"
          f"{'池/测':>9}{'比值p90':>9}{'无条件覆盖':>11}")
    print("-" * 108)
    for r in rows:
        print(f"{r['city']:<6}{r['N']:>6}{r['n_pool']:>8}{r['n_test']:>7}"
              f"{r['pool_iqr_med']:>11.3f}{r['test_iqr_med']:>11.3f}"
              f"{r['ratio_med']:>9.2f}{r['ratio_p90']:>9.2f}{r['cov_uncond']:>11.1%}")

    print("\n" + "=" * 118)
    print(f"Q2 状态依赖：平台态 = 该处已有 {PLATEAU_LEN}h+ 未变。锚点态因果可知；目标态是 t+H 处，不可知")
    print(f"{'城':<6}{'池锚点平台率':>12}{'测锚点平台率':>12}"
          f"{'锚点态条件':>20}{'目标态条件(oracle)':>22}{'逐对oracle':>11}")
    print("-" * 118)
    for r in rows:
        (pf, cp), (nf, cn) = r["cov_anchor"]["锚点平台"], r["cov_anchor"]["锚点非平台"]
        (tf, tp), (uf, up) = r["cov_tgt"]["目标平台"], r["cov_tgt"]["目标非平台"]
        print(f"{r['city']:<6}{r['plat_frac_pool']:>12.1%}{r['plat_frac_test']:>12.1%}"
              f"{cp:>9.1%}/{cn:<9.1%}{tp:>10.1%}/{up:<10.1%}{r['cov_oracle']:>11.1%}")
    print("\n" + "=" * 118)
    print("Q3 池残差宽度的时间轨迹（池按起点时间四等分，每段中位逐站 IQR）+ 与测试期逐站配对")
    print(f"{'城':<6}{'段1':>26}{'段2':>20}{'段3':>20}{'段4':>20}{'测试段':>20}{'测<池占比':>11}")
    print("-" * 118)
    for r in rows:
        cells = []
        for (i0, i1, v) in r["seg"]:
            d0 = (pd.Timestamp(r["t0"]) +
                  pd.Timedelta(int(i0 * r["dt_ns"]), unit="ns")).strftime("%m-%d")
            cells.append(f"{d0}~{v:6.2f}")
        print(f"{r['city']:<6}" + "".join(f"{c:>20}" for c in cells) +
              f"{np.nanmedian(r['test_iqr']):>20.2f}{r['frac_test_narrower']:>11.1%}")
    print("\n读法：若段4（最靠近测试期）已经和'测试段'接近，而段1 明显更大 -> 池被**早期瞬态**")
    print("      拉宽（AMS 停摆恢复期）。若四段都差不多大、却都远大于测试段 -> 是平稳的窗口")
    print("      差异，不是瞬态。'测<池占比' = 有多少比例的站，测试期残差比池子更窄。")

    print("\n" + "=" * 118)
    print("Q4 '不动站'占比（逐站残差 IQR==0 <=> 该站至少一半时间不变）与 okj 过滤的偏倚")
    print(f"{'城':<6}{'池不动站':>10}{'测不动站':>10}{'池IQR中位(全站)':>17}{'测IQR中位(全站)':>17}"
          f"{'池IQR中位(okj)':>16}{'测IQR中位(okj)':>16}")
    print("-" * 118)
    for r in rows:
        print(f"{r['city']:<6}{r['const_pool']:>10.1%}{r['const_test']:>10.1%}"
              f"{np.nanmedian(r['pool_iqr']):>17.3f}{np.nanmedian(r['test_iqr']):>17.3f}"
              f"{r['pool_iqr_med']:>16.3f}{r['test_iqr_med']:>16.3f}")
    print("\n读法：Q1 的 '池/测' 比值只在 '测试期会动' 的站子集上算。若两城中位 IQR 差得很多，")
    print("      说明 Q1 那个 2 倍多的比值是**选择出来的**，不代表全城普遍现象。")

    print("\n" + "=" * 118)
    print("Q5 残差聚合方差膨胀：Var(Σ r_j) / Σ Var(r_j) = 1 + (N-1)·ρ̄")
    print(f"{'城':<6}{'站':>6}{'联合/独立方差比':>18}{'平均站间相关ρ̄':>18}{'预测 CRPS_sum 比值':>20}")
    print("-" * 118)
    for r in rows:
        print(f"{r['city']:<6}{r['N']:>6}{r['var_ratio']:>18.3f}{r['rho_bar']:>18.5f}"
              f"{r['var_ratio']:>20.3f}")
    print("\n读法：比值=1+(N-1)ρ̄。**注意**：这只看池内的相关强度，不代表 CRPS_sum 一定有区分度——")
    print("      若测试期聚合波动远小于池期，联合采样反而过宽受罚，见 Q6。")

    print("\n" + "=" * 118)
    print("Q6 聚合残差 Σ_j r_j：池(采样器给聚合的宽度) vs 测试(真相)")
    print(f"{'城':<6}{'池σ':>12}{'测试σ':>12}{'池σ/测σ':>10}{'测试聚合偏置':>14}"
          f"{'聚合点MAE':>12}{'点MAE/测σ':>11}{'池σ/均值之和':>15}")
    print("-" * 118)
    for r in rows:
        print(f"{r['city']:<6}{r['std_agg_pool']:>12.1f}{r['std_agg_test']:>12.1f}"
              f"{r['agg_ratio']:>10.2f}{r['bias_agg_test']:>14.1f}"
              f"{r['agg_point_mae']:>12.1f}"
              f"{r['agg_point_mae']/r['std_agg_test']:>11.2f}"
              f"{r['std_agg_pool']/r['scale']:>15.4f}")
    print("\n读法：'池σ/测σ' >> 1 表示池期聚合波动远大于测试期 -> 联合采样给聚合量过宽（AMS 32.7×）。")
    print("      '聚合点MAE/测σ' 接近或超过 1 时，CRPS_sum 几乎全由**点预测的聚合误差**决定，")
    print("      分布形状（以及联合/独立的差别）贡献可忽略 —— 这是 main/indep≈1 最可能的成因。")
    print("\n" + "=" * 118)
    print("\n读法：名义 50%。'锚点态条件' = 只能用历史状态分层；'目标态条件' = 假设我们知道")
    print("      24h 后那个站是否还在平台里（现实中不可知，是上界）；'逐对 oracle' = 每对按各自")
    print("      目标态查表。若 锚点态 修不好、目标态 能修好 -> 残差宽度由**未来状态**决定，")
    print("      只能靠预测状态转移来缩小，不是靠读当前状态。")
