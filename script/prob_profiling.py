# -*- coding: utf-8 -*-
"""概率预测前置画像：零结构 + seasonal naive 残差。

回答两件事：
  A. 零结构 —— 零是随机的还是成块的？零率随昼夜/周内怎么变？条件零概率？
  B. seasonal naive 残差 —— 不确定性随 horizon × hour-of-day 怎么分布？
     残差是什么分布？在"预测为零"和"预测为正"处条件异方差有多大差别？

seasonal naive 用标准形式（Hyndman）：ŷ(t+h|t) = y(t + h - m(k+1)), k = floor((h-1)/m)
只有这个形式误差才随 h 增长；若用"目标时刻往前 m 小时"则残差分布与 h 无关。

用法：python3 script/prob_profiling.py
输出：result/prob_profiling/*.png  + 控制台文本报告
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/data/yzzpz/Charged"
CITIES = ["LOA", "MEL", "JHB", "AMS", "SPO", "SZH"]
FEATURE = "volume"
HORIZONS = [24, 48, 72]          # 用户定的三档
HMAX = 72
SEASONAL = [24, 168]             # 日周期 / 周周期
OUT = f"{BASE}/result/prob_profiling"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- 载入
OUTAGE_FRAC = 0.8      # 某小时 >80% 的站同时为零 => 判为整城停摆（数据中断），置 NaN


def load(city):
    p = f"{BASE}/data/{city}_remove_const"
    v = pd.read_csv(f"{p}/{FEATURE}.csv", index_col=0)
    v.index = pd.to_datetime(v.index)
    y = v.values.astype(np.float64)
    y = mask_outage(y, city)          # 停摆行 -> NaN，不参与任何统计
    return y, v.index, list(v.columns)


def mask_outage(y, city, verbose=True):
    """整城停摆检测。

    CHARGED 里 AMS 2023-04-01~05-16 存在数据中断：20.5% 的小时里 >90% 的站
    同时为 0，这 901 小时贡献了 AMS 全部零值的 84%。把这些小时当"空闲"会同时
    污染零结构（伪长游程）和残差（伪大误差），因此整行置 NaN 剔除。
    """
    frac = (y == 0).mean(1)
    bad = frac > OUTAGE_FRAC
    if verbose and bad.any():
        print(f"  [outage] {city}: {bad.sum()} 小时 ({bad.mean():.1%}) 判为停摆，已剔除")
    y = y.copy()
    y[bad] = np.nan
    return y


# ================================================================ A 零结构
def zero_structure(y, t, city):
    """返回零结构的各项统计（停摆行已为 NaN，全程按有效值归一）"""
    T, N = y.shape
    fin = np.isfinite(y)
    Z = (y == 0) & fin                                  # NaN 既不是零也不算有效
    out = {"city": city, "n_station": N, "n_hour": T,
           "n_valid_hour": int(fin.all(1).sum()),
           "overall": Z.sum() / max(fin.sum(), 1)}

    # --- 昼夜零率
    hod = t.hour.values
    out["hod_zero"] = np.array(
        [Z[hod == h].sum() / max(fin[hod == h].sum(), 1) for h in range(24)])
    out["hod_vol"] = np.array(
        [np.nanmean(y[hod == h]) if fin[hod == h].any() else np.nan
         for h in range(24)])

    # --- 周内零率
    dow = t.dayofweek.values
    out["dow_zero"] = np.array(
        [Z[dow == d].sum() / max(fin[dow == d].sum(), 1) for d in range(7)])

    # --- 分站零率
    n_fin = fin.sum(0)
    per_station = np.where(n_fin > 0, Z.sum(0) / np.maximum(n_fin, 1), np.nan)
    ok = n_fin > 0
    out["station_zero"] = per_station[ok]
    out["station_pct"] = np.percentile(per_station[ok], [10, 50, 90])
    out["n_always_zero"] = int((per_station[ok] > 0.999).sum())
    out["n_never_zero"] = int((per_station[ok] < 0.001).sum())

    # --- 零游程（沿时间轴，逐站；NaN 天然断开游程，因为 Z 在 NaN 处为 False）
    run_lengths, run_per_station, zero_in_long = [], np.zeros(N, int), 0
    for j in range(N):
        z = Z[:, j].astype(np.int8)
        if not z.any():
            continue
        d = np.diff(np.concatenate([[0], z, [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        L = ends - starts
        run_lengths.append(L)
        run_per_station[j] = len(L)
        zero_in_long += int(L[L >= 24].sum())   # 落在 ≥24h 连续空闲里的零小时数
    run_lengths = np.concatenate(run_lengths) if run_lengths else np.array([0])
    out["runs_per_station"] = run_per_station.mean()
    out["run_mean_len"] = run_lengths.mean()
    out["run_p50"] = np.percentile(run_lengths, 50)
    out["run_p90"] = np.percentile(run_lengths, 90)
    out["run_max"] = run_lengths.max()
    out["frac_zero_in_runs_ge24"] = zero_in_long / max(Z.sum(), 1)

    # --- 条件零概率：起点为零 vs 起点非零（两端都必须有效）
    out["cond"] = {}
    for h in [1, 6, 12, 24, 48, 72]:
        a, b = y[:-h], y[h:]
        both = np.isfinite(a) & np.isfinite(b)
        za, zb = (a == 0), (b == 0)
        m0 = za & both
        m1 = (~za) & both
        out["cond"][h] = (
            zb[m0].mean() if m0.any() else np.nan,
            zb[m1].mean() if m1.any() else np.nan,
        )
    return out


# ===================================================== B seasonal naive 残差
def seasonal_residuals(y, t, m, rows_per_h=40, cap=2_000_000):
    """标准 seasonal naive。

    归一化：逐站除以该站全期均值，使不同规模的站可比。

    池化分布只保留**抽样**（每个 horizon 随机取 rows_per_h 行），不保留全量——
    AMS 有 1385 站，全量存 72 个 horizon x 2 个周期要十几 GB。热力图用全量，
    分布统计用抽样（两者用途不同，抽样对直方图/分位数完全够用）。
    """
    T, N = y.shape
    rng = np.random.default_rng(0)
    denom = np.nanmean(y, 0)                         # 停摆行已 NaN，不影响分母
    denom[~np.isfinite(denom) | (denom <= 0)] = np.nan   # 全零站除外
    res_norm = np.full((HMAX, 24), np.nan)           # 热力图：horizon x hour-of-day（全量）
    res_norm_cnt = np.zeros((HMAX, 24), int)
    smp = {"pooled": [], "pred0": [], "predpos": []}  # 抽样缓冲
    n_smp = 0
    for h in range(1, HMAX + 1):
        lag = m * ((h - 1) // m + 1)
        t0 = max(0, lag - h)
        t1 = T - h
        if t0 >= t1:
            continue
        ts = np.arange(t0, t1)
        pred = y[ts + h - lag]                       # ŷ(t+h|t)
        true = y[ts + h]
        r = (true - pred) / denom                    # 逐站归一化
        hour = (ts + h) % 24
        for hh in range(24):                         # 热力图用全量，不抽样
            sel = hour == hh
            if sel.any():
                v = r[sel]
                v = v[np.isfinite(v)]
                if v.size:
                    res_norm[h - 1, hh] = np.std(v)
                    res_norm_cnt[h - 1, hh] = v.size
        # ---- 池化分布：抽样，且总量封顶
        if n_smp < cap:
            rows = rng.choice(len(ts), min(len(ts), rows_per_h), replace=False)
            rr, pp = r[rows], pred[rows]
            fin = np.isfinite(rr)
            smp["pooled"].append(rr[fin])
            if (pp[fin] == 0).any():
                smp["pred0"].append(rr[fin][pp[fin] == 0])
            if (pp[fin] != 0).any():
                smp["predpos"].append(rr[fin][pp[fin] != 0])
            n_smp += fin.sum()
    cat = lambda k: np.concatenate(smp[k]) if smp[k] else np.array([])
    return dict(heat=res_norm, cnt=res_norm_cnt,
                pooled=cat("pooled"), pred0=cat("pred0"), predpos=cat("predpos"))


def describe(r):
    if r.size == 0:
        return dict(n=0)
    return dict(
        n=r.size, mean=r.mean(), std=r.std(),
        p05=np.percentile(r, 5), p25=np.percentile(r, 25), p50=np.percentile(r, 50),
        p75=np.percentile(r, 75), p95=np.percentile(r, 95),
        iqr=np.percentile(r, 75) - np.percentile(r, 25),
        skew=float(pd.Series(r).skew()), kurt=float(pd.Series(r).kurt()),
        frac_exact0=float((r == 0).mean()),
    )


# ================================================================ main
def main():
    ZS, RS = {}, {}
    for c in CITIES:
        y, t, cols = load(c)
        ZS[c] = zero_structure(y, t, c)
        RS[c] = {m: seasonal_residuals(y, t, m) for m in SEASONAL}
        print(f"[load] {c}: {y.shape}", flush=True)

    # ------------------------------------------------ 文本报告 A
    print("\n" + "=" * 96)
    print("A. 零结构")
    print("=" * 96)
    print(f"{'城':<5}{'站数':>6}{'总零率':>9}{'分站零率 P10/50/90':>26}"
          f"{'全零站':>8}{'无零站':>8}{'游程均值':>10}{'≥24h 零占比':>13}")
    print("-" * 96)
    for c in CITIES:
        z = ZS[c]
        p = z["station_pct"]
        print(f"{c:<5}{z['n_station']:>6}{z['overall']:>9.1%}"
              f"{f'{p[0]:.1%}/{p[1]:.1%}/{p[2]:.1%}':>26}"
              f"{z['n_always_zero']:>8}{z['n_never_zero']:>8}"
              f"{z['run_mean_len']:>10.1f}{z['frac_zero_in_runs_ge24']:>13.1%}")

    print("\n昼夜零率（0-23 时）：")
    for c in CITIES:
        v = ZS[c]["hod_zero"]
        print(f"  {c:<5} 谷 {v.argmin():>2}时 {v.min():>5.1%}   峰 {v.argmax():>2}时 {v.max():>5.1%}   "
              f"峰谷差 {v.max()-v.min():>5.1%}")

    print("\n条件零概率  P(y[t+h]=0 | y[t]=0)  vs  P(y[t+h]=0 | y[t]>0)：")
    print(f"  {'城':<5}" + "".join(f"{f'h={h}':>16}" for h in [1, 6, 12, 24, 48, 72]))
    for c in CITIES:
        row = "".join(f"{f'{a:.1%}/{b:.1%}':>16}" for a, b in
                      (ZS[c]["cond"][h] for h in [1, 6, 12, 24, 48, 72]))
        print(f"  {c:<5}{row}")

    # ------------------------------------------------ 文本报告 B
    print("\n" + "=" * 96)
    print("B. seasonal naive 残差（逐站归一化：残差 / 该站全期均值）")
    print("   注：方差/分位数/分布形状来自抽样（每 horizon 40 行，总量封顶 200 万）；")
    print("       h=24/48/72 那三列是热力图的全量均值。")
    print("=" * 96)
    for m in SEASONAL:
        tag = "日周期 m=24" if m == 24 else "周周期 m=168"
        print(f"\n--- {tag} ---")
        print(f"{'城':<5}{'std':>9}{'IQR':>9}{'P5':>9}{'P95':>9}{'偏度':>9}{'峰度':>9}"
              f"{'残差恰为0':>11}{'h=24 std':>10}{'h=48 std':>10}{'h=72 std':>10}")
        print("-" * 96)
        for c in CITIES:
            R = RS[c][m]
            d = describe(R["pooled"])
            hs = [R["heat"][h - 1].mean() for h in HORIZONS]
            print(f"{c:<5}{d['std']:>9.3f}{d['iqr']:>9.3f}{d['p05']:>9.3f}{d['p95']:>9.3f}"
                  f"{d['skew']:>9.2f}{d['kurt']:>9.2f}{d['frac_exact0']:>11.1%}"
                  f"{hs[0]:>10.3f}{hs[1]:>10.3f}{hs[2]:>10.3f}")

        print(f"\n  条件异方差（按 seasonal naive 预测值是否为零分组）：")
        print(f"  {'城':<5}{'预测=0 的残差 std':>20}{'预测>0 的残差 std':>20}{'比值':>9}"
              f"{'预测=0 占比':>13}")
        for c in CITIES:
            R = RS[c][m]
            a, b = R["pred0"], R["predpos"]
            sa = a.std() if a.size else np.nan
            sb = b.std() if b.size else np.nan
            frac = a.size / max(a.size + b.size, 1)
            print(f"{c:<5}{sa:>20.3f}{sb:>20.3f}{sa/max(sb,1e-9):>9.2f}{frac:>13.1%}")

    # ------------------------------------------------ 图
    plt.rcParams.update({"font.size": 9, "axes.grid": True,
                         "grid.alpha": .25, "figure.dpi": 130})
    C = plt.cm.tab10(np.linspace(0, 1, 10))

    # Fig 1: 昼夜零率 + 分站零率分布
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, c in enumerate(CITIES):
        ax[0].plot(range(24), ZS[c]["hod_zero"] * 100, marker="o", ms=3,
                   lw=1.4, color=C[i], label=c)
    ax[0].set(xlabel="Hour of day", ylabel="Zero fraction (%)",
              title="Zero rate by hour of day", xticks=range(0, 24, 3))
    ax[0].legend(ncol=2, fontsize=8, frameon=False)
    bp = ax[1].boxplot([ZS[c]["station_zero"] * 100 for c in CITIES],
                       tick_labels=CITIES, showfliers=False, patch_artist=True, widths=.55)
    for p in bp["boxes"]:
        p.set(facecolor="#dfe6ee", edgecolor="#5b6b80")
    ax[1].set(ylabel="Zero fraction (%)", title="Per-station zero rate")
    fig.tight_layout(); fig.savefig(f"{OUT}/A_zero_structure.png"); plt.close(fig)

    # Fig 2: horizon x hour-of-day 热力图（6 城 x 2 周期）
    for m in SEASONAL:
        fig, axs = plt.subplots(2, 3, figsize=(13, 6.2), sharex=True, sharey=True)
        vmax = np.nanpercentile([RS[c][m]["heat"] for c in CITIES], 98)
        for ax_, c in zip(axs.ravel(), CITIES):
            im = ax_.imshow(RS[c][m]["heat"], aspect="auto", origin="lower",
                            cmap="magma", vmin=0, vmax=vmax,
                            extent=[-.5, 23.5, .5, HMAX + .5])
            ax_.set_title(f"{c}   (m={m})", fontsize=10)
            ax_.set_yticks([1, 24, 48, 72]); ax_.grid(False)
        for ax_ in axs[1]:
            ax_.set_xlabel("Hour of day")
        for ax_ in axs[:, 0]:
            ax_.set_ylabel("Horizon h (hours)")
        fig.colorbar(im, ax=axs, shrink=.75, label="std of normalized residual")
        fig.suptitle(f"Seasonal-naive residual spread by horizon x hour-of-day "
                     f"(residual / station mean)", y=.98)
        fig.savefig(f"{OUT}/B_heatmap_m{m}.png", bbox_inches="tight"); plt.close(fig)

    # Fig 3: 残差分布（池化）
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for m, ls in zip(SEASONAL, ["-", "--"]):
        for i, c in enumerate(CITIES):
            r = RS[c][m]["pooled"]
            ax[0].hist(np.clip(r, -2, 4), bins=120, histtype="step", lw=1.2,
                       color=C[i], ls=ls, alpha=.85,
                       label=f"{c} (m={m})" if m == 24 else None)
    ax[0].set(xlabel="Normalized residual (clipped to [-2,4])", ylabel="count",
              title="Pooled residual distribution")
    ax[0].legend(fontsize=8, frameon=False)
    for m, ls in zip(SEASONAL, ["-", "--"]):
        for i, c in enumerate(CITIES):
            r = RS[c][m]["pred0"]
            ax[1].hist(np.clip(r, -2, 4), bins=120, histtype="step", lw=1.2,
                       color=C[i], ls=ls, alpha=.85,
                       label=f"{c} (m={m})" if m == 24 else None)
    ax[1].set(xlabel="Normalized residual (clipped to [-2,4])", ylabel="count",
              title="Residual where seasonal naive predicts 0")
    ax[1].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/C_residual_dist.png"); plt.close(fig)

    print(f"\n图已保存到 {OUT}/")
    print("  A_zero_structure.png  零结构（昼夜 + 分站分布）")
    print("  B_heatmap_m24.png / B_heatmap_m168.png  horizon x hour-of-day 热力图")
    print("  C_residual_dist.png   残差分布（池化 + 预测为零处）")


if __name__ == "__main__":
    main()
