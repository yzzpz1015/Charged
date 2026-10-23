# -*- coding: utf-8 -*-
"""城市级宏观时序：6 个月里总充电量、上报站数、站均充电量怎么变。

回答的问题：某城的训练/验证/测试段是不是“同一种数据”。把总量拆成
“有多少站在上报 × 每站平均多少”，再叠上三段划分的边界，一眼看清
数据是在稳定运行，还是处在平台上线/站冻结的过渡期。

用法：
    python script/viz_city_macro.py --city AMS --out result/figs/ams_macro.png
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib import font_manager as _fm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from api.dataset.common import EVDataset  # noqa: E402

_fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

SURFACE = '#fcfcfb'
INK, INK2 = '#0b0b0b', '#52514e'
GRID, AXIS = '#e6e5e1', '#c3c2b7'
SPLIT_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']   # 与分窗图一致：train/valid/test
PANEL_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']


def daily(s, how='mean'):
    return getattr(s.resample('D'), how)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--city', default='AMS')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or f'{BASE}/result/figs/{args.city.lower()}_macro.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ds = EVDataset(feature='volume', auxiliary='None',
                   data_path=f'{BASE}/data/{args.city}_remove_const/')
    ds.split_cross_validation(fold=6, total_fold=6, train_ratio=0.8, valid_ratio=0.1)
    raw, T, N = ds.feat, ds.time, ds.feat.shape[1]

    tot = pd.Series(raw.sum(1), index=T)
    act = pd.Series((raw > 0).sum(1), index=T)
    per = tot / act.clip(lower=1)
    t0, t1 = T[0], T[-1]
    tr0, tr1 = T[ds.train_start], T[ds.train_end - 1]
    va0, va1 = T[ds.valid_start], T[ds.valid_end - 1]
    te0, te1 = T[ds.test_start], T[ds.test_end - 1]

    fig, axes = plt.subplots(3, 1, figsize=(14, 7.2), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    series = [(tot, '每小时总充电量（全部站求和）', 'kWh/小时'),
              (act, '有充电量的站数（volume > 0）', '站'),
              (per, '站均充电量（总充电量 ÷ 有充电量站数）', 'kWh/小时/站')]
    for k, (ax, (s, title, unit)) in enumerate(zip(axes, series)):
        ax.set_facecolor(SURFACE)
        color = PANEL_COLORS[k]
        sd = daily(s)
        if k == 0:      # 总量画小时线，底层细线看日周期、上层日均看宏观
            ax.plot(s.index, s.values, color=color, lw=0.6, alpha=0.28)
        ax.plot(sd.index, sd.values, color=color, lw=2.0,
                solid_capstyle='round')
        ax.set_title(title, fontsize=10, color=INK, loc='left', pad=3)
        ax.set_ylabel(unit, fontsize=8.5, color=INK2)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(AXIS)
            ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors=INK2, labelsize=8, length=3)

        # 未进入划分的时段（停摆行规则剔除）用灰底；三段用各自颜色的极浅底
        ax.axvspan(t0, tr0, color='#f0efec', zorder=0)
        for (a, b, c) in [(tr0, tr1, SPLIT_COLORS[0]), (va0, va1, SPLIT_COLORS[1]),
                          (te0, te1, SPLIT_COLORS[2])]:
            ax.axvspan(a, b, color=c, alpha=0.07, lw=0, zorder=0)
        if k == 0:
            for x in (tr0, va0, te0):
                ax.axvline(x, color=INK2, lw=0.9, ls=(0, (3, 3)))
            trans = ax.get_xaxis_transform()
            bb = dict(facecolor=SURFACE, edgecolor='none', pad=1.5)
            ax.text(t0 + (tr0 - t0) / 2, 0.05, '未进入划分', transform=trans,
                    ha='center', va='bottom', fontsize=8.5, color=INK2, bbox=bb)
            for x0, x1, lab in [(tr0, tr1, '训练'), (va0, va1, '验证'), (te0, te1, '测试')]:
                ax.text(x0 + (x1 - x0) / 2, 0.05, lab, transform=trans,
                        ha='center', va='bottom', fontsize=9.5, color=INK2,
                        fontweight='bold', bbox=bb)

    # 通用标注：开头 30 天是否明显少于后期；是否有持续 ≥3 天的掉线
    act_d = daily(act)
    early, later = act_d.iloc[:30].mean(), act_d.iloc[30:].median()
    if early < 0.5 * later:
        axes[0].annotate(f'前 30 天均值仅 {early:.0f} 站上报（后期 {later:.0f}）',
                         xy=(t0 + pd.Timedelta(days=14), tot[:24 * 20].mean()),
                         xytext=(t0 + pd.Timedelta(days=32), tot.max() * 0.55),
                         fontsize=9, color=INK2,
                         arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.9))
    fy = axes[1].get_xaxis_transform()   # x=data，y=轴内比例；避免大站/小站坐标越界
    axes[1].annotate(f'共 {N} 站', xy=(t1 - pd.Timedelta(days=12), act.iloc[-24 * 12:].mean()),
                     xytext=(t1 - pd.Timedelta(days=40), 0.10), textcoords=fy,
                     fontsize=9, color=INK2, ha='right',
                     arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.9))
    # 掉线检测：只看进入划分之后，基准用整段中位数。
    # 不能用短窗滚动中位数——掉线持续超过窗口长度时，基准会跟着掉下去，反而检测不到。
    d = act_d[act_d.index >= tr0]
    base = d.median()
    low = (d < 0.85 * base).values
    if low.any():
        edges = np.flatnonzero(np.diff(np.r_[False, low, False]))
        runs = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]
        # 开头 30 天内的“低”是上线爬坡（已由总图上方的标注说明），不重复标。
        runs = [r for r in runs if d.index[r[0]] >= tr0 + pd.Timedelta(days=30)]
        if runs:
            a, b = max(runs, key=lambda r: r[1] - r[0])
        else:
            a = b = 0
        if b - a >= 5:
            mid = a + (b - a) // 2
            axes[1].annotate(f'{d.index[a]:%m/%d}–{d.index[b - 1]:%m/%d} '
                             f'约 {d.iloc[a:b].mean():.0f} 站上报（常态 {base:.0f}）',
                             xy=(d.index[mid], d.iloc[mid]),
                             xytext=(d.index[min(b + 8, len(d) - 1)], 0.30),
                             textcoords=fy,
                             fontsize=9, color=INK2,
                             arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.9))

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    axes[-1].set_xlim(t0, t1)
    fig.suptitle(f'{args.city}：6 个月城市级时序（细线=小时，粗线=日均）',
                 fontsize=14, color=INK, x=0.06, ha='left', y=0.985)
    fig.text(0.06, 0.94,
             '灰底 = 该时段 >80% 的站为 0，按停摆行规则不进划分；竖线为三段分界。',
             fontsize=9.5, color=INK2, ha='left')
    legend = [Line2D([0], [0], color=INK2, lw=0.8, alpha=0.4, label='小时值'),
              Line2D([0], [0], color=INK2, lw=2.0, label='日均')]
    fig.legend(handles=legend, loc='upper right', bbox_to_anchor=(0.985, 0.975),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f'[fig] {out}')


if __name__ == '__main__':
    main()
