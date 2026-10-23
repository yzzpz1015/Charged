# -*- coding: utf-8 -*-
"""③ 分位头 vs ⑤ 吸收态扩散：同一批测试格子的**预测分布形状**对比。

为什么只画这两个
----------------
指标上已经确认：⑤ 在有电格上 ≈ ③，差距 99% 落在真零格。也就是说问题不在
"连续值画得准不准"，而在"这格到底是不是 0"以及 0 处那团质量。这张图就是为了
把这两件事分开看：

* ③（橙）：99 个分位点插值出的 200 条样本。大量分位点同时等于 0 时，会看到
  一根竖在 0 上的柱子 —— 那就是分位头对"零原子"的表达。
* ⑤（青）：吸收态扩散的 100 条场景样本。末步按 p₀ 抽伯努利，所以同一条
  采样链里既有 0、也有正值；0 的比例就是模型给出的 P(y=0)。

每个面板 = 一个测试窗口 × 一个站 × 提前量 H。黑竖线 = 事后真值；横条 =
10%–90% / 25%–75% 分位范围，白杠 = 中位数。

口径
----
* 只抽 **clean 掩码内**的格子（与正式评测同一批），站集/窗口与评测一致。
* ⑤ 用 100 条样本（评测用 24 条只是为了省时间；这里要多看一点形状）。
* ③ 的 200 条样本是**边际**抽样（各站独立），与评测里的 crpsΣ 口径一致。

用法::

    python script/prob_viz_35.py --city MEL --device 0
    python script/prob_viz_35.py --city MEL --device 0 --n_zero 4 --n_pos 8 --seed 3
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

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import kairos_prob_eval as ke                                        # noqa: E402

C3, C5 = "#eb6834", "#1baf7a"        # 橙 / 青（与 prob_viz 的 ③④ 同槽位）
C_TRUE = "#0b0b0b"
GRID = "#e6e5e1"
INK2 = "#52514e"
SURFACE = "#fcfcfb"


def draw_panel(ax, title, truth, s3, s5):
    """一个格子：上行 ③、下行 ⑤。"""
    vals = [np.asarray([truth])] + [np.asarray(v) for v in (s3, s5) if v is not None]
    lo = min(np.percentile(v, 1) for v in vals)
    hi = max(np.percentile(v, 99) for v in vals)
    pad = 0.10 * max(hi - lo, 1e-6)
    ax.set_xlim(min(lo - pad, -pad), hi + pad)

    rows = [("③ 分位头", s3, C3), ("⑤ zidiff", s5, C5)]
    for r, (name, v, col) in enumerate(rows):
        yc = -r
        ax.axhline(yc, color=GRID, lw=0.6, zorder=0)
        if v is None or len(v) == 0:
            continue
        v = np.asarray(v, float)
        q = np.quantile(v, [0.10, 0.25, 0.50, 0.75, 0.90])
        ax.plot([q[0], q[4]], [yc, yc], color=col, lw=1.0, alpha=0.85,
                solid_capstyle="butt", zorder=2)
        ax.plot([q[1], q[3]], [yc, yc], color=col, lw=4.0, alpha=0.95,
                solid_capstyle="butt", zorder=3)
        ax.plot([q[2]], [yc], marker="|", color=SURFACE, ms=6, mew=1.6, zorder=4)
        jit = (np.random.default_rng(0).random(len(v)) - 0.5) * 0.30
        ax.scatter(v, yc + jit, s=3.0 if len(v) > 50 else 7.0, color=col,
                   alpha=0.45, linewidths=0, zorder=1)
        zr = float(np.mean(v == 0))
        ax.text(0.006, yc, f"{name}  零值率 {zr:.2f}", transform=ax.get_yaxis_transform(),
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
    ap.add_argument("--city", default="MEL")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--n_zero", type=int, default=6)
    ap.add_argument("--n_pos", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_s3", type=int, default=200)
    ap.add_argument("--n_s5", type=int, default=100)
    ap.add_argument("--out_root", default=os.path.join(BASE, "result", "kairos_l1_clean"))
    ap.add_argument("--csdi_root", default=os.path.join(BASE, "result", "csdi_clean"))
    args = ap.parse_args()

    ke.CSDI_ROOT = args.csdi_root
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    j = args.horizon - 1

    city = args.city
    ds, raw, starts = ke.load_city(city)
    N = raw.shape[1]
    st_te = np.asarray(starts["te"])
    keep = ke.clean_keep(raw, st_te)                    # [M,N]，与评测同一掩码
    y = raw
    off = np.arange(ke.SEQ_L)[None, :]

    # ---- 选格子：零格与正格分开抽；正格按真值分位数取，覆盖小/中/大 ----
    ytgt = y[st_te + ke.SEQ_L + j]                      # [M,N]
    zeros = np.argwhere(keep & (ytgt == 0))
    pos = np.argwhere(keep & (ytgt > 0))
    if len(zeros) == 0 or len(pos) == 0:
        raise SystemExit("该 horizon 没有可用的零格/正格")
    zi = zeros[rng.choice(len(zeros), min(args.n_zero, len(zeros)), replace=False)]
    pv = ytgt[pos[:, 0], pos[:, 1]]
    qs_pick = np.quantile(pv, np.linspace(0.05, 0.95, args.n_pos))
    pi = []
    used = set()
    for q in qs_pick:
        d = np.abs(pv - q)
        order = np.argsort(d)
        for k in order:
            if k not in used:
                used.add(int(k)); pi.append(pos[k]); break
    cells_sel = np.concatenate([zi, np.asarray(pi)], 0)
    Msel = len(cells_sel)
    win_idx = cells_sel[:, 0]
    st_idx = cells_sel[:, 1]
    print(f"[{city}] 选 {len(zi)} 个零格 + {len(pi)} 个正格，共 {Msel} 格；"
          f"窗口范围 {win_idx.min()}~{win_idx.max()}", flush=True)

    uniq_w = np.unique(win_idx)
    wmap = {w: k for k, w in enumerate(uniq_w)}
    st_all = st_te[uniq_w]
    km = ke.build_kairos(city, "plain", args.out_root, device, 99)
    if km is None:
        raise SystemExit(f"{city} 没有 plain 权重（{args.out_root}）")
    P72 = ke.point_pass(km, raw, st_all, list(range(ke.SEQ_L)), device)
    P72 = np.stack([P72[h] for h in range(ke.SEQ_L)], axis=2)   # [nW,N,72]
    del km
    torch.cuda.empty_cache()

    # ---- ③ 分位头 ----
    qm = ke.build_kairos(city, "quant", args.out_root, device, 99)
    if qm is None:
        raise SystemExit(f"{city} 没有 quant 权重")
    qs = np.arange(1, 100) / 100
    X = raw[st_all[:, None] + off]
    X = np.transpose(X, (0, 2, 1)).reshape(-1, ke.SEQ_L).astype(np.float32)
    P3 = ke._forward(qm, X, device).reshape(len(st_all), N, ke.SEQ_L, len(qs))
    del qm
    torch.cuda.empty_cache()
    u3 = ke.strat_u(np.random.default_rng(0), args.n_s3, N, shared=False)

    # ---- ⑤ zidiff ----
    core = ke.load_zidiff(city, device)
    if core is None:
        raise SystemExit(f"{city} 没有 zidiff 权重")
    from api.model.csdi import base_cond, hist_feats
    hid, inv = core["hid"], np.argsort(core["hid"])
    m_t = torch.from_numpy(np.asarray(core["bias"])[hid]).float().to(device)
    s_t = torch.from_numpy(np.asarray(core["scale"])[hid]).float().to(device)
    gen = torch.Generator(device=device).manual_seed(0)
    hist_on = np.transpose((y[st_te[:, None] + off] > 0), (0, 2, 1)).astype(np.float32)
    hod = ds.time.hour.values
    hod72 = np.stack([hod[st_te + ke.SEQ_L + h] for h in range(ke.SEQ_L)],
                     axis=1).astype(np.float32)

    ncol = 3
    nrow = int(np.ceil(Msel / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 1.55 * nrow + 0.6),
                             squeeze=False, facecolor=SURFACE)
    summ = []
    for c in range(nrow * ncol):
        ax = axes[c // ncol][c % ncol]
        if c >= Msel:
            ax.axis("off"); continue
        i, sid = int(win_idx[c]), int(st_idx[c])
        w = wmap[i]
        with torch.no_grad():
            pr = torch.from_numpy(np.ascontiguousarray(P72[w][hid])).float().to(device)[None]
            ho = torch.from_numpy(np.ascontiguousarray(hist_on[i][hid])).float().to(device)[None]
            a_c, o_c = hist_feats(ho)
            hd = torch.from_numpy(hod72[i]).float().to(device)[None, None, :]
            cond = base_cond(pr, s_t, hd, a_c, o_c, N, ke.SEQ_L)
            if core.get("cstats") is not None:
                cm, cs = core["cstats"]
                cond = (cond - cm) / cs
            smp = core["dif"].sample(core["net"], cond, n_samples=args.n_s5,
                                     generator=gen, stride=ke.CSDI_STRIDE)
            ysim = pr[None, :, None] + m_t[None, None, :, None] \
                + s_t[None, None, :, None] * smp
            ysim = ysim.cpu().numpy()[:, :, 0][:, :, inv, :]       # [S,1,N,72]
        s5 = ysim[:, 0, sid, j]
        s3 = ke.quant_to_samples_ind(P3[w][:, j, :][None, sid], qs, u3[:, sid: sid + 1])[:, 0]
        truth = float(ytgt[i, sid])
        title = f"窗{i} 站{sid}　真值 {truth:.2f}"
        draw_panel(ax, title, truth, s3, s5)
        summ.append(dict(win=i, st=sid, truth=truth,
                         z3=float(np.mean(s3 == 0)), z5=float(np.mean(s5 == 0)),
                         med3=float(np.median(s3)), med5=float(np.median(s5)),
                         iqr3=float(np.quantile(s3, .75) - np.quantile(s3, .25)),
                         iqr5=float(np.quantile(s5, .75) - np.quantile(s5, .25))))

    fig.suptitle(f"{city}　③ 分位头 vs ⑤ 吸收态扩散（H={args.horizon}，clean 口径）",
                 fontsize=10, color=C_TRUE, y=1.0)
    fig.tight_layout()
    out = os.path.join(BASE, "result", "figs",
                       f"prob_shape_35_{city}_H{args.horizon}_seed{args.seed}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    print(f"已保存 {os.path.relpath(out, BASE)}", flush=True)

    # ---- 形状摘要（把图里看到的数变成可核对的表）----
    print(f"\n{'窗/站':>10}{'真值':>8}{'③零值率':>9}{'⑤零值率':>9}"
          f"{'③中位':>9}{'⑤中位':>9}{'③IQR':>9}{'⑤IQR':>9}", flush=True)
    for d in summ:
        print(f"{d['win']:>4}/{d['st']:<5}{d['truth']:>8.2f}{d['z3']:>9.2f}{d['z5']:>9.2f}"
              f"{d['med3']:>9.2f}{d['med5']:>9.2f}{d['iqr3']:>9.2f}{d['iqr5']:>9.2f}",
              flush=True)


if __name__ == "__main__":
    main()
