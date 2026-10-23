# -*- coding: utf-8 -*-
"""量 Kairos-L1 残差场（测试期）的跨站结构 —— 决定 CRPS_sum 该用哪种耦合。

测什么
------
把 L1 点预测在测试期真实犯的错取出来，r[窗, 站] = 实测 - 预测，然后问三件事：

  1. **逐对相关 ρ̄**：同一时刻，各站的误差是不是"一起犯"？（对残差逐站标准化
     后再算相关，消掉站与站之间量级差异的影响）
  2. **求和后的波动放大倍数**：Var(Σ_站 r) / Σ_站 Var(r)。各站互不相干时这个数
     =1；完全同进同退时 =站数。
  3. **直接打分**：拿这份残差场做 CRPS_sum，整条向量抽（保留跨站结构）
     vs 逐站独立抽（拆散跨站结构），差多少。

为什么这件事决定了做法③的主指标能不能用
----------------------------------------
分位头只给**每站各自的**分布，"各站怎么拼成一个场景"没有被参数化，必须人为指定。
两个极端是：独立耦合 -> Var(Σ)=ΣVar（波动最小），完全同单调 -> 波动最大（=站数倍）。
真值在哪，**由这份实测残差场说了算**，不由我们的默认值说了算。

读数规则
--------
* 若实测放大倍数接近 1，且"整条抽 vs 独立抽"的 CRPS_sum 差很小
  -> 独立耦合就是对的，做法③ 的 crpsΣsum 可用。
* 若实测放大倍数明显 >1
  -> 独立耦合在**系统性低估**聚合波动，做法③ 的 crpsΣsum 会虚低（显得比实际好），
    必须换成"按实测相关结构来拼"，否则那个数不能拿去和做法①比。

注意比对的基准不同
------------------
`prob_baseline.py` 上"联合 vs 独立"几乎无差（main/indep 0.93~1.04），但那是
**季节朴素**残差的跨站结构。这里是**模型残差**的跨站结构，两者可以不同：
模型把共同的日/周节律吃掉了，剩下的可能更接近独立；也可能相反。必须实测。

用法：python3 script/prob_resid_corr.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prob_baseline as pb                                    # noqa: E402

BASE = pb.BASE
CITIES = ["LOA", "MEL", "JHB", "AMS", "SPO", "SZH"]
SEQ_L = 72
HORIZONS = [24, 48, 72]
M_SAMPLE = 200
OUT = f"{BASE}/result/prob_resid_corr"


def load_pred(city):
    """返回 (pred[W, H, N], starts[W])；与 pb 的 live_cols 站序一致（已由
    script/check_split_alignment.py 断言过）。"""
    d = f"{BASE}/result/kairos_l1/{city}"
    P = np.load(f"{d}/predict_plain_pre72_fold6.npy")
    s = np.load(f"{d}/starts_plain_pre72_fold6.npy")
    W = len(s)
    N = P.shape[0] // W
    assert N * W == P.shape[0], f"{city}: {P.shape} 不能被窗口数 {W} 整除"
    assert P.shape[1] == SEQ_L
    return P.reshape(W, N, SEQ_L).transpose(0, 2, 1), s, N


def run_city(city):
    y0, t0 = pb.load(city)
    ok, trm, vam, tem, u, span = pb.split_rows(y0, t0)
    cols, smean = pb.live_cols(y0, trm)

    pred, starts, N = load_pred(city)
    assert N == len(cols), f"{city}: 预测站数 {N} != live_cols {len(cols)}"
    W = len(starts)

    # 目标时刻 = 窗口起点 + SEQ_L + h；整段都在 test 段内（由 starts 的构造保证）
    ti = starts[:, None] + SEQ_L + np.arange(SEQ_L)[None, :]
    obs = y0[ti][:, :, cols]                                  # [W, H, N]
    assert np.all(np.isfinite(obs)), f"{city}: 目标区有非有限值"
    r = obs - pred                                            # [W, H, N] 残差场

    hod = t0.hour.values
    res = {"city": city, "n_station": N, "n_win": W,
           "mae_recomputed": float(np.abs(r).mean())}
    rm = json.load(open(f"{BASE}/result/kairos_l1/{city}/"
                        f"metrics_plain_pre72_fold6.json"))
    res["mae_stored"] = rm["MAE"]

    for H in HORIZONS:
        x = r[:, H - 1, :]                                    # [W, N]
        # ---- 1/2. 标准化后逐对相关；标准化消掉站间量级差，剩下的才是"一起犯"
        mu, sd = x.mean(0), x.std(0)
        sd = np.where(sd < 1e-12, 1e-12, sd)
        xs = (x - mu) / sd
        C = (xs.T @ xs) / W
        off = C[~np.eye(N, dtype=bool)]
        rho_bar = float(off.mean())

        # Var比：unstandardized（决定真实求和波动）与 standardized（纯相关效应）
        ratio_raw = float(np.var(x.sum(1)) / np.var(x, axis=0).sum())
        ratio_std = float(np.var(xs.sum(1)) / N)
        # 相关矩阵的主特征值占比：若一个大特征值吃掉很多，说明是"共同因子"结构
        ev = np.linalg.eigvalsh(C)[::-1]
        top_frac = float(ev[0] / N)

        # ---- 3. 直接打分：整条向量抽 vs 逐站独立抽
        # 池 = 同"目标小时"的其它测试窗口的残差向量（与 pb 的季节朴素池同构，
        # 只是这里换成模型残差）。站间同源结构在整条抽样里原样保留。
        rng = np.random.default_rng(0)
        tgt_h = hod[starts + SEQ_L + H - 1]
        npar = np.arange(N)
        cs_j, cs_i = [], []
        pool_n = []
        for i in range(W):
            idx = np.where(tgt_h == tgt_h[i])[0]
            idx = idx[idx != i]                               # 别把自己当自己的池
            if len(idx) == 0:
                continue
            pool_n.append(len(idx))
            pick = rng.choice(idx, min(M_SAMPLE, len(idx)), replace=True)
            sub = r[pick, H - 1, :]                           # [m, N] 整条残差向量
            smp_j = pred[i, H - 1][None, :] + sub
            sel = rng.integers(0, len(idx), size=(M_SAMPLE, N))
            smp_i = pred[i, H - 1][None, :] + sub[sel, npar]
            cs_j.append(float(pb.crps_fair(smp_j.sum(1), obs[i, H - 1].sum())))
            cs_i.append(float(pb.crps_fair(smp_i.sum(1), obs[i, H - 1].sum())))

        used = len(cs_j)
        res[H] = dict(
            rho_bar=rho_bar, ratio_raw=ratio_raw, ratio_std=ratio_std,
            top_eig_frac=top_frac, n_used=used, n_skip=W - used,
            pool_med=int(np.median(pool_n)) if pool_n else 0,
            cs_sum_joint=float(np.mean(cs_j)) if cs_j else np.nan,
            cs_sum_indep=float(np.mean(cs_i)) if cs_i else np.nan,
            ratio_joint_indep=(float(np.mean(cs_j) / np.mean(cs_i))
                               if cs_i and np.mean(cs_i) else np.nan),
        )
        print(f"[{city} H={H}] ρ̄={rho_bar:+.4f} Var比 raw={ratio_raw:.2f} "
              f"std={ratio_std:.2f} (独立=1, 同单调={N}) "
              f"CRPS_sum 整条/独立={res[H]['ratio_joint_indep']:.3f}", flush=True)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    allres = []
    for c in CITIES:
        allres.append(run_city(c))
        json.dump(allres, open(f"{OUT}/metrics.json", "w"),
                  ensure_ascii=False, indent=1)
    print(f"\n已保存 {OUT}/metrics.json")


if __name__ == "__main__":
    main()
