# -*- coding: utf-8 -*-
"""把「同一个格子上，各个方法给出的那堆数长什么样」画出来。

为什么要有这张图
----------------
指标（CRPS / PICP）是一堆平均数，看不出**形状**。而我们现在卡住的问题恰恰是形状：
做法④ 给出的 24 个数，覆盖率只有 0.64（该 0.90），但把整堆数整体放大之后覆盖率上去了、
打分反而更差 —— 说明不是"整体太窄"，是形状不对。形状不对有至少三种可能（整体偏移 /
该散开的地方挤成一坨 / 尾巴太短），**光看指标分不开，得看图。**

这张图做什么
------------
在测试段里**随机抽若干个格子**（格子 = 某个测试窗口 × 某个站 × 某个提前量），
每个格子一个小面板，里面横着画三行：

    ① 点预测 + 残差池抽样      （蓝）
    ③ 分位头                   （橙）
    ④ 整块扩散                 （青）

每行上：小点 = 该方法给出的样本（一堆"可能的答案"），细横线 = 这堆数的 10%~90% 范围，
粗横线 = 25%~75%，白竖杠 = 中位数。**黑竖线 = 真值**（事后才知道的那个唯一答案）。

看什么：黑线是不是落在三行的范围中间？三行的范围一样宽吗？④ 那 24 个点是散成一个包，
还是挤成一根（或者分成好几簇）？真值是 0 的那几格，④ 有没有干脆给一串 0？

口径说明（别当成严谨指标读）
--------------------------
* 样本数**三行不一样**：①③ 各抽 200 条（它们本来就便宜），④ 只有 24 条
  （扩散采样贵，评测里就是 24 条）。所以 ④ 的点看起来稀，那是采样数少，不是分布更窄。
* 抽的是**测试段**的格子（与评测同一批窗口、同一批真值），不是训练段。
* 站是随机抽的，各站量级差很多，**每个面板自己一个横轴刻度**，面板之间不比分度。
* ④ 用的是**真门卫**（`--mask clf`，与正式表一致）。真值是 0 的格子上 ④ 会被门卫
  判死、24 条样本全是 0 —— 那不是 bug，是做法④ 的行为本身，正是要看的东西。
  想看"只论画师"就加 `--mask oracle`（拿真值当门卫，作弊口径，只能当上限）。
"""
import argparse
import os
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

# 中文字形：默认的 DejaVu Sans 一个汉字都没有，全画成豆腐块（第一次跑就踩了）。
# Noto Sans CJK 已装在系统里且被 matplotlib 登记，直接挂到 family 列表最前面；
# Droid Sans Fallback 兜底（简体覆盖更全），DejaVu 留着管数学符号。
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import prob_baseline as pb                                           # noqa: E402
import kairos_prob_eval as ke                                        # noqa: E402

# 配色：dataviz 参考调色板的前三槽（蓝/橙/青），已过校验器（CVD ΔE 9.2 ≥ 8）。
# 青对浅底对比度 2.74 < 3:1，触发"relief 规则"——所以每一行**必须**直接写出方法名，
# 不许只靠颜色区分身份。这件事在 _draw_panel 里做了。
C1, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a"
C_TRUE = "#0b0b0b"
# 点预测是**参考线**（三个方法都在它上面加东西），不是第四个方法 —— 所以给中性灰、
# 不占第四个分类色槽，且用虚线跟实线的真值分开。
CPRED = "#8f8d86"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"
INK2 = "#52514e"


def collect(city, j, n_cell, seed, device, mask_mode="clf", ts=None):
    """抽格子并算齐三个方法在每个格子上的样本。返回 list[dict]。

    ``ts=(start, span)`` 时不再随机抽，改成取**连续一段**测试窗口 —— 这是画
    "一段连续时间的真值 vs 预测区间"用的：同一个 horizon 下，窗口序号每 +1，
    目标时刻也正好 +1 小时，所以连起来就是一条连续曲线（发报时刻总在目标前
    SEQ_L 小时，是"滚动的一天前预报"）。中间若撞上停摆缺口，窗口会跳号，
    画的时候按真实时间戳落点，缺口会自己显出来。
    """
    OUT_ROOT = os.environ.get("KAIROS_CKPT_ROOT", f"{BASE}/result/kairos_l1")
    rng = np.random.default_rng(seed)

    ds, raw, starts = ke.load_city(city)
    N = raw.shape[1]
    y0, t0 = pb.load(city)
    ok, trm, _, _, _, _ = pb.split_rows(y0, t0)
    cols, smean = pb.live_cols(y0, trm)
    assert len(cols) == N, f"{city} 站数不一致：Kairos {N} vs 基线 {len(cols)}"
    hod = t0.hour.values
    y = y0[:, cols]                                   # [T, N] 原始量纲

    s_te = np.asarray(starts['te']) + ke.SEQ_L + j    # 各测试窗口在该 horizon 的目标行号
    s_va = np.asarray(starts['va']) + ke.SEQ_L + j
    lag = pb.LAG * int(np.ceil((j + 1) / pb.LAG))
    m_te, m_va = ok[s_te - lag], ok[s_va - lag]       # 锚点可用（与 bootstrap_variants 同口径）
    st_te, st_va = np.asarray(starts['te']), np.asarray(starts['va'])

    # ---- 主干点预测：① 只要 horizon j 那一列；④ 要整块 72 步 ----
    km = ke.build_kairos(city, 'plain', OUT_ROOT, device, 99)
    if km is None:
        raise SystemExit(f"{city} 没有 kairos plain 权重（{OUT_ROOT}）")
    Pte = ke.point_pass(km, raw, st_te, [j], device)[j]     # [M,N]
    Pva = ke.point_pass(km, raw, st_va, [j], device)[j]     # [Mv,N]

    # 抽格子：随机抽 or 取连续一段。两种都只在"锚点可用"的测试窗口里取。
    cand = np.where(m_te)[0]
    if ts is not None:
        start, span = ts
        sel = np.arange(start, min(start + span, len(m_te)))
        sel = sel[m_te[sel]]
    else:
        sel = np.sort(rng.choice(cand, min(n_cell, len(cand)), replace=False))
    P72 = ke.point_pass(km, raw, st_te[sel], list(range(ke.SEQ_L)), device)
    P72 = np.stack([P72[h] for h in range(ke.SEQ_L)], axis=2)   # [n_cell,N,72]
    del km
    torch.cuda.empty_cache()

    # ---- ① roll28：**整条残差向量**抽样（与 bootstrap_variants 的命名分支逐字同口径）----
    res_h = np.concatenate([y[s_va[m_va]] - Pva[m_va], y[s_te[m_te]] - Pte[m_te]], 0)
    hour_h = np.concatenate([hod[s_va[m_va]], hod[s_te[m_te]]])
    s_h = np.concatenate([s_va[m_va], s_te[m_te]])

    def samp1(i):
        m = ((hour_h == hod[s_te[i]]) & (s_h < s_te[i])
             & (s_h >= s_te[i] - ke.ROLL_DAYS * 24))
        idx = np.where(m)[0]
        if len(idx) == 0:
            return None
        pick = rng.choice(idx, min(ke.M_SAMPLE, len(idx)), replace=True)
        return Pte[i][None, :] + res_h[pick]                   # [m, N] 整条向量

    # ---- ③ 分位头：抽中的窗口做一次前向，拿 99 个分位数再插值成样本 ----
    qm = ke.build_kairos(city, 'quant', OUT_ROOT, device, 99)
    if qm is None:
        raise SystemExit(f"{city} 没有 kairos quant 权重")
    qs = np.arange(1, 100) / 100
    u3 = ke.strat_u(np.random.default_rng(0), 200, N, shared=False)
    off = np.arange(ke.SEQ_L)[None, :]
    X = raw[st_te[sel][:, None] + off]
    X = np.transpose(X, (0, 2, 1)).reshape(-1, ke.SEQ_L).astype(np.float32)
    P3 = ke._forward(qm, X, device).reshape(len(sel), N, ke.SEQ_L, len(qs))
    del qm
    torch.cuda.empty_cache()
    qv = P3[:, :, j, :]                                        # [n_cell, N, Q]

    # ---- ④ 整块扩散：与评测完全同一套条件打包与采样口径 ----
    core = ke.load_csdi(city, device)
    if core is None:
        raise SystemExit(f"{city} 没有 CSDI 权重")
    from api.model.csdi import base_cond, pack_cond, hist_feats
    hid, inv = core['hid'], np.argsort(core['hid'])
    m_t = torch.from_numpy(np.asarray(core['bias'])[hid]).float().to(device)
    s_t = torch.from_numpy(np.asarray(core['scale'])[hid]).float().to(device)
    ca, cb = core['cal']
    gen = torch.Generator(device=device).manual_seed(0)

    hist_on = np.transpose((y[st_te[:, None] + off] > 0), (0, 2, 1)).astype(np.float32)
    hod72 = np.stack([hod[st_te + ke.SEQ_L + h] for h in range(ke.SEQ_L)],
                     axis=1).astype(np.float32)                # [M,72]

    out = []
    for c, i in enumerate(sel):
        # —— ④：单窗走一遍采样链（与 csdi_pass 逐行对应，只是 b=1）——
        with torch.no_grad():
            pr = torch.from_numpy(np.ascontiguousarray(P72[c][hid])).float().to(device)[None]
            ho = torch.from_numpy(np.ascontiguousarray(hist_on[i][hid])).float().to(device)[None]
            a_c, o_c = hist_feats(ho)
            hd = torch.from_numpy(hod72[i]).float().to(device)[None, None, :]
            base = base_cond(pr, s_t, hd, a_c, o_c, N, ke.SEQ_L)
            p_on = torch.sigmoid(ca * core['ono'](base) + cb)
            if mask_mode == 'oracle':
                # 作弊口径：拿真值 on/off 当门卫。**是整块 [站 × 72]**，不是目标那一个小时
                # —— 扩散一次生成 72 小时，掩码自然也是 72 小时的（csdi_pass 里同样口径）。
                # 掩码要在**空间序**上（卷积在站轴做），而 y 是原始列序，所以按 hid 重排。
                blk = y[st_te[i] + ke.SEQ_L + np.arange(ke.SEQ_L)][:, hid].T   # [N,72]
                m = torch.from_numpy(np.ascontiguousarray(blk > 0)) \
                    .float().to(device)[None, None, :, :]
                assert tuple(m.shape) == tuple(p_on.shape), (m.shape, p_on.shape)
            else:
                m = (torch.rand(p_on.shape, device=device, generator=gen) < p_on).float()
            cond = pack_cond(base, p_on, N, ke.SEQ_L, core.get('cstats'))
            smp = core['dif'].sample(core['net'], cond, m, torch.zeros_like(m),
                                     n_samples=ke.CSDI_NSAMPLE, generator=gen,
                                     stride=ke.CSDI_STRIDE)
            ysim = pr[None, :, None] + m_t[None, None, :, None] \
                + s_t[None, None, :, None] * smp
            ysim = (ysim * m[None]).cpu().numpy()[:, :, 0][:, :, inv, :]   # [S,1,N,72]
        # m 是 [1,1,N,72]，所以"判活"要取目标那一个小时、只数站轴；直接 m.sum()
        # 会把 72 个钟头全加进去，打出"判活 2021/33 格"这种数（第一次跑就踩了）。
        # when 存成 datetime64（npz 里能直接存，且 matplotlib 认它）；存 pandas 的
        # Timestamp 会变成 object 数组，allow_pickle=False 读不回来。
        out.append(dict(win=int(i), when=np.datetime64(t0[s_te[i]]),
                        truth=y[s_te[i]], pred=P72[c][:, j].copy(),
                        s1=samp1(i),
                        s3=ke.quant_to_samples_ind(qv[c], qs, u3),
                        s4=ysim[:, 0, :, j], gon=int(m[:, :, :, j].sum())))
        if ts is None or (c + 1) % 40 == 0:
            print(f"  格子 {c + 1}/{len(sel)}：窗口 {i} 抽完"
                  f"（H={j + 1} 这一小时，门卫判活 {int(m[:, :, :, j].sum())}/{N} 站）",
                  flush=True)
    del core
    torch.cuda.empty_cache()
    return out


def diagnose(cells, lo_q=10, hi_q=90):
    """把 12 个面板的"看一眼"变成能数出来的数：**漏在哪一头**。

    为什么必须数：图上"真值跑到范围外"有两种完全不同的病，治法相反 ——

    * **两头都漏**（真值 < q10 和 > q90 都多）→ 区间太窄，是"宽度"问题；
    * **只漏一头**（比如全都漏高）→ 区间不窄，是**整体偏低**，是"位置/偏差"问题，
      这时候去加宽度是白费，该修的是中心。

    另外单列两件只有 ④ 才有的事：门卫把真值 > 0 的格子**判死**（24 条样本全 0）
    的比例 —— 这笔账在边际指标里看不出来，但它一次就是真值大小的全额损失。
    """
    print(f"\n[diag] {len(cells)} 个抽中的窗口（H 由 --horizon 定），"
          f"逐格统计「真值相对样本的位置」；名义上 10~90 区间该罩住 80%", flush=True)
    rows = []
    def dead_cells(cell):
        """门卫判死的格：24 条样本一条都不动（g 恒 0，输出被乘成 0）。"""
        s4 = cell['s4']
        return (s4.max(0) == 0) & (s4.min(0) == 0)

    for tag, key, col in (("① 点+残差池", 's1', C1), ("③ 分位头", 's3', C3),
                          ("④ 整块扩散", 's4', C4)):
        # 第三行口径是回答"该不该继续投扩散"的关键：**把门卫判死的格剔掉**，
        # 剩下的就是画师自己的表现（这些格上门卫 m≡1，与满分门卫那格逐字等价，
        # 却不用重跑一遍 oracle 采样）。
        for lab, sel in (("全部格", None), ("只真值>0的格", 0.0),
                         ("只真值>0且没被判死", "alive")):
            n = below = above = 0
            bias = []
            for cell in cells:
                v = cell.get(key)
                if v is None:
                    continue
                if sel == "alive":
                    keep = (cell['truth'] > 0) & (~dead_cells(cell))
                elif sel is not None:
                    keep = cell['truth'] > sel
                else:
                    keep = None
                if keep is not None:
                    v, tr = v[:, keep], cell['truth'][keep]
                else:
                    tr = cell['truth']
                if tr.size == 0:
                    continue
                q10 = np.quantile(v, lo_q / 100, axis=0)
                q90 = np.quantile(v, hi_q / 100, axis=0)
                n += tr.size
                below += int((tr < q10).sum())
                above += int((tr > q90).sum())
                bias.append(np.median(v, 0) - tr)
            if not n:
                continue
            b = np.concatenate(bias)
            rows.append((f"{tag} · {lab}", n, below / n, above / n, n and (n - below - above) / n,
                         float(np.median(b)), col))

    print(f"  {'方法 · 口径':<22}{'格数':>7}{'真值<10%':>10}{'真值>90%':>10}"
          f"{'落在里面':>10}{'样本中位数-真值':>16}", flush=True)
    for name, n, b, a, inside, med, _ in rows:
        print(f"  {name:<22}{n:>7d}{b:>9.1%}{a:>10.1%}{inside:>10.1%}{med:>16.2f}",
              flush=True)

    # ④ 独有的账：门卫判死但真值 > 0 —— 边际指标完全看不见这一笔
    dead_tot = dead_true = 0
    for cell in cells:
        s4, tr = cell['s4'], cell['truth']
        # "判死" = 24 条样本一条都不动（门卫给的 g 恒 0，输出被乘成 0）
        dead = (s4.max(0) == 0) & (s4.min(0) == 0)
        dead_tot += int(dead.sum())
        dead_true += int((dead & (tr > 0)).sum())
    print(f"  └ ④ 门卫判死的格子：{dead_tot} 个，其中**真值明明 > 0** 的 {dead_true} 个"
          f"（占判死的 {dead_true / max(1, dead_tot):.0%}）—— 边际 CRPS 看不见这笔",
          flush=True)
    print("  怎么读：两列『漏』都不小 = 太窄（宽度问题）；只有一列大 = 整体偏（位置问题）。",
          flush=True)

    # ---- 覆盖曲线：区分「整条曲线等比缩水」与「尾巴特别薄」----
    # 只按一个全局尺度放大（实验C 的 k 扫描）能救的，必须是**形状对、尺度小**：
    # 那种情况下 50/80/90 三档的缺口应该差不多大。若缺口随档位递增（50 档还行、
    # 90 档差一大截），说明中间够胖、**尾巴太薄** —— 这时候放大 k 是把中间的肥肉
    # 再摊薄，覆盖上去了但打分更差，正是 k 扫描看到的现象。
    print(f"\n[diag] 覆盖曲线（同一批 {len(cells)} 窗，只真值>0 且门卫没判死的格）"
          f"——名义值 50/80/90，看缺口怎么随档位变", flush=True)
    print(f"  {'方法':<12}{'名义 50%':>12}{'名义 80%':>12}{'名义 90%':>12}"
          f"   （括号里 = 实测 − 名义，负得越多越窄）", flush=True)
    for tag, key in (("① 点+残差池", 's1'), ("③ 分位头", 's3'), ("④ 整块扩散", 's4')):
        hit = {50: 0.0, 80: 0.0, 90: 0.0}
        nwin = ncell = 0
        for cell in cells:
            v = cell.get(key)
            if v is None:
                continue
            keep = (cell['truth'] > 0) & (~dead_cells(cell))
            if not keep.any():
                continue
            v, tr = v[:, keep], cell['truth'][keep]
            nwin += 1
            ncell += tr.size
            for lv in hit:
                lo = np.quantile(v, (1 - lv / 100) / 2, axis=0)
                hi = np.quantile(v, 1 - (1 - lv / 100) / 2, axis=0)
                hit[lv] += float(((tr >= lo) & (tr <= hi)).mean())
        if not nwin:
            continue
        # 逐窗比例求和后除以**参与统计的窗口数**（不是总窗数）—— 少除/多除一次
        # 就是那个把 PICP 打 2125 倍的老坑。
        parts = "".join(f"{hit[lv] / nwin:>8.1%}({hit[lv] / nwin - lv / 100:+.0%})"
                        for lv in (50, 80, 90))
        print(f"  {tag:<12}{parts}   窗口 {nwin} / 格 {ncell}", flush=True)
    return rows


def draw_timeseries(cells, sid, city, horizon, out, mask_note=""):
    """连续一段窗口：x = 时间，三行分别是 ①②④ 的预测区间，黑线是真值。

    和格子图的分工：格子图回答"某一格上这堆数长什么样"；这张回答"**沿着时间**
    它的形状对不对" —— 该平的平不平、该起的峰起没起、门卫判死的那些小时是不是
    正好压在真值有车的地方。

    每个点都是"提前 horizon 小时发的报"（窗口每前进 1 小时，目标也前进 1 小时），
    所以这画的是一条**滚动的一天前预报**。
    """
    t = np.array([c['when'] for c in cells])
    tr = np.array([c['truth'][sid] for c in cells], float)
    # 主干点预测（PatchTST/Kairos）：三个方法都是"在它上面加东西"，所以三行都画同一条，
    # 一眼能看出各方法的样本堆是围着它、还是整体偏出去。
    pd_ = np.array([c['pred'][sid] if c.get('pred') is not None else np.nan
                    for c in cells], float)
    rows = [("① 点预测 + 残差池", 's1', C1), ("③ 分位头", 's3', C3),
            ("④ 整块扩散", 's4', C4)]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 8.2), facecolor=SURFACE)
    for ax, (name, key, col) in zip(axes, rows):
        med = np.array([np.median(c[key][:, sid]) if c.get(key) is not None else np.nan
                        for c in cells], float)
        lo10 = np.array([np.quantile(c[key][:, sid], .10) if c.get(key) is not None else np.nan
                         for c in cells], float)
        hi90 = np.array([np.quantile(c[key][:, sid], .90) if c.get(key) is not None else np.nan
                         for c in cells], float)
        lo25 = np.array([np.quantile(c[key][:, sid], .25) if c.get(key) is not None else np.nan
                         for c in cells], float)
        hi75 = np.array([np.quantile(c[key][:, sid], .75) if c.get(key) is not None else np.nan
                         for c in cells], float)
        ax.fill_between(t, lo10, hi90, color=col, alpha=0.22, lw=0, label="10~90% 区间")
        ax.fill_between(t, lo25, hi75, color=col, alpha=0.42, lw=0, label="25~75% 区间")
        ax.plot(t, med, color=col, lw=1.6, label="样本中位数")
        ax.plot(t, pd_, color=CPRED, lw=1.3, ls=(0, (5, 3)),
                label="点预测（PatchTST 主干）", zorder=5)
        ax.plot(t, tr, color=C_TRUE, lw=1.4, label="真值", zorder=6)
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_ylabel(name, fontsize=9, color=INK2)
        ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(labelsize=8, colors=INK2)
        ax.legend(loc="upper right", fontsize=7.5, frameon=False, ncol=5)
    axes[-1].set_xlabel("目标时刻（每个点都是提前 %d 小时发的报）" % horizon,
                        fontsize=9, color=INK2)
    fig.suptitle(f"{city} 站序 {sid} · 连续 {len(cells)} 小时 · 提前 {horizon} 小时"
                 f"（真值=黑实线、点预测=灰虚线，两条在三行里都重复画；"
                 f"站序是 Hilbert 排位，不是原始站号）{mask_note}",
                 fontsize=10, color=C_TRUE)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"已存 {out}", flush=True)


def _draw_panel(ax, cell, sid, title):
    """一个面板 = 一个格子。三行分别画 ①③④ 的样本堆。"""
    truth = float(cell['truth'][sid])
    rows = [("① 点+残差池", cell['s1'][:, sid] if cell['s1'] is not None else None, C1),
            ("③ 分位头", cell['s3'][:, sid], C3),
            ("④ 整块扩散", cell['s4'][:, sid], C4)]
    vals = [np.asarray([truth])] + [np.asarray(v) for _, v, _ in rows if v is not None]
    lo = min(np.percentile(v, 1) for v in vals)
    hi = max(np.percentile(v, 99) for v in vals)
    pad = 0.10 * max(hi - lo, 1e-6)
    ax.set_xlim(lo - pad, hi + pad)

    for r, (name, v, col) in enumerate(rows):
        yc = -r
        ax.axhline(yc, color=GRID, lw=0.6, zorder=0)
        if v is None:
            ax.text(0.5, yc, "（本窗无可用池）", transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=6, color=INK2)
            continue
        v = np.asarray(v, float)
        q = np.quantile(v, [0.10, 0.25, 0.50, 0.75, 0.90])
        ax.plot([q[0], q[4]], [yc, yc], color=col, lw=1.0, alpha=0.85,
                solid_capstyle="butt", zorder=2)
        ax.plot([q[1], q[3]], [yc, yc], color=col, lw=4.0, alpha=0.95,
                solid_capstyle="butt", zorder=3)
        ax.plot([q[2]], [yc], marker="|", color=SURFACE, ms=6, mew=1.6, zorder=4)
        # 原始样本点：④ 只有 24 个，点画大一点也好认
        jit = (np.random.default_rng(0).random(len(v)) - 0.5) * 0.30
        ax.scatter(v, yc + jit, s=3.0 if len(v) > 50 else 7.0, color=col,
                   alpha=0.45, linewidths=0, zorder=1)
        # 直接标方法名 —— 配色校验器对青色的 relief 要求（不许只靠颜色区分身份）
        ax.text(0.006, yc, name, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.5, color=INK2)

    ax.axvline(truth, color=C_TRUE, lw=1.6, zorder=5)
    ax.plot([truth], [0.58], marker="v", color=C_TRUE, ms=5.0, clip_on=False, zorder=6)
    ax.set_ylim(-2.66, 0.78)
    ax.set_yticks([])
    ax.set_title(title, fontsize=7.5, color=C_TRUE, pad=3)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", labelsize=6.5, colors=INK2, length=2, pad=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="SPO")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--n_cell", type=int, default=12,
                    help="抽多少个窗口。图只画前 --max_panels 个，但统计表用全部。")
    ap.add_argument("--max_panels", type=int, default=12,
                    help="图上最多画几个面板（多了排不下）。统计表不受此限。")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ts_span", type=int, default=0,
                    help=">0 就换成「连续一段」模式：取这么多个连续测试窗口，画一条"
                         "滚动预报曲线（时间序列图），而不是随机格子的面板图。")
    ap.add_argument("--ts_start", type=int, default=0, help="连续段从第几个测试窗口开始。")
    ap.add_argument("--station", type=int, default=-1,
                    help="连续段用哪个站。默认 -1 = 自动挑该段里真值为 0 的小时最多的站。")
    ap.add_argument("--mask", choices=("clf", "oracle"), default="clf")
    ap.add_argument("--min_truth", type=float, default=-1.0,
                    help="只保留真值 ≥ 该值的站。默认 -1 = 不筛（零格也照画，"
                         "那正是'④ 把格子判死、24 条样本全 0'的现场）。")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None,
                    help="样本缓存 npz。有就复用，省掉重算（改图不用重跑模型）。")
    ap.add_argument("--no_cache", action="store_true")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    j = args.horizon - 1
    ts = (args.ts_start, args.ts_span) if args.ts_span > 0 else None
    tag_ts = f"_ts{args.ts_start}+{args.ts_span}" if ts else f"_n{args.n_cell}"
    cache = args.cache or (f"{BASE}/result/figs/.cache_{args.city}_H{args.horizon}"
                           f"_{args.mask}_s{args.seed}{tag_ts}.npz")
    if os.path.exists(cache) and not args.no_cache:
        print(f"[{args.city}] 复用样本缓存 {cache}", flush=True)
        z = np.load(cache, allow_pickle=False)
        cells = [dict(win=int(w),
                      # when 是后加的字段，老缓存里没有 —— 缺了就留 None，别整个炸掉
                      when=z[f"when_{i}"][()] if f"when_{i}" in z.files else None,
                      truth=z[f"truth_{i}"],
                      s1=z[f"s1_{i}"] if f"s1_{i}" in z.files else None,
                      s3=z[f"s3_{i}"], s4=z[f"s4_{i}"])
                 for i, w in enumerate(z['wins'])]
    else:
        print(f"[{args.city}] {'连续段 ' + str(ts) if ts else '抽 %d 个格子' % args.n_cell}"
              f" @ H={args.horizon}（seed={args.seed}, mask={args.mask}）", flush=True)
        cells = collect(args.city, j, args.n_cell, args.seed, device, args.mask, ts)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, wins=np.array([c['win'] for c in cells]),
                            **{f"when_{i}": c['when'] for i, c in enumerate(cells)}, **{
            **{f"truth_{i}": c['truth'] for i, c in enumerate(cells)},
            **{f"s3_{i}": c['s3'] for i, c in enumerate(cells)},
            **{f"s4_{i}": c['s4'] for i, c in enumerate(cells)},
            **({f"s1_{i}": c['s1'] for i, c in enumerate(cells)
                if c['s1'] is not None}),
        })
        print(f"样本已缓存 {cache}", flush=True)

    # 连续段模式：只画时间序列图，跳过"随机格子"那套面板和统计表
    if ts:
        N = cells[0]['truth'].shape[0]
        tr_all = np.stack([c['truth'] for c in cells])              # [K, N]
        sid = args.station
        if sid < 0:
            # 自动挑站：先在"该段平均量级不低于全城中位"的站里挑，再取零小时最多的那个。
            # 直接取零最多的会挑到几乎不动的死站，那种曲线看不出东西；这个口径保证
            # 挑到的是"平时有生意、但也经常空着"的站 —— 门卫和画师都在这里分胜负。
            means = tr_all.mean(0)
            cand_s = np.where(means >= np.median(means))[0]
            sid = int(cand_s[np.argmax((tr_all[:, cand_s] == 0).sum(0))])
        print(f"[{args.city}] 连续段共 {len(cells)} 小时，画第 {sid} 号站"
              f"（该段真值：中位 {np.median(tr_all[:, sid]):.2f}，"
              f"零 {int((tr_all[:, sid] == 0).sum())} 小时，"
              f"最大 {tr_all[:, sid].max():.2f}）", flush=True)
        out = args.out or (f"{BASE}/result/figs/prob_series_{args.city}_H{args.horizon}"
                           f"_st{sid}{'' if args.mask == 'clf' else '_oracle'}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        note = "" if args.mask == "clf" else " · 满分门卫（作弊口径，只看画师）"
        draw_timeseries(cells, sid, args.city, args.horizon, out, note)
        return

    diagnose(cells)

    shown = cells[:args.max_panels]
    rng = np.random.default_rng(args.seed + 1)
    ncol, nrow = 3, int(np.ceil(len(shown) / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 1.75 * nrow),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax, cell in zip(axes, shown):
        N = cell['truth'].shape[0]
        pool = np.where(cell['truth'] >= args.min_truth)[0]
        sid = int(rng.choice(pool if len(pool) else np.arange(N)))
        _draw_panel(ax, cell, sid,
                    f"测试窗口 {cell['win']} · 第 {sid} 号站 · 真值 {cell['truth'][sid]:.1f}")
    for ax in axes[len(shown):]:
        ax.axis("off")
    tag = "" if args.mask == "clf" else "（满分门卫 · 作弊口径，只看画师）"
    fig.suptitle(f"{args.city}  H={args.horizon}：每个格子 = 一堆「可能的答案」 vs 那一个真值"
                 f"　——　随机抽 {len(shown)} 格{tag}",
                 fontsize=10.5, color=C_TRUE, y=0.998)
    fig.text(0.012, 0.004,
             "小点 = 样本；细线 = 10~90% 范围；粗线 = 25~75%；白竖杠 = 中位数；"
             "黑竖线 = 真值（▼ 标出）。  ①③ 各 200 条样本，④ 只有 24 条（评测口径）——"
             "④ 的点稀是采样数少，不代表更窄。",
             fontsize=7.0, color=INK2)
    fig.tight_layout(rect=(0, 0.024, 1, 0.984))
    out = args.out or (f"{BASE}/result/figs/prob_shape_{args.city}_H{args.horizon}"
                       f"{'' if args.mask == 'clf' else '_' + args.mask}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"已存 {out}", flush=True)


if __name__ == "__main__":
    main()
