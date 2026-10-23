# -*- coding: utf-8 -*-
"""可视化一段数据的「典型窗口」：同一站的一段 72h 历史 + 72h 目标。

用途：解释为什么某城的验证损失/训练损失差很多——把窗口按目标段的波动程度
排序，取各自的分位样本画出来，直接看「水平线」和「真在动的线」长什么样。

用法：
    python script/viz_split_windows.py --city AMS --out result/figs/ams_split_windows.png
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 集群上 matplotlib 把 Noto Sans CJK 的 .ttc 注册成 “JP” 族名（字形覆盖简体中文）。
from matplotlib import font_manager as _fm
_fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from api.dataset.common import EVDataset, SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST  # noqa: E402
from api.utils import straight_stretch_hours  # noqa: E402

SEQ = 72
# 参考调色板前三个分类槽（已用 dataviz 校验脚本以 all-pairs 模式验证通过；
# aqua 对浅色背景对比度 <3:1，故每格都有可见标题/行标签作为 relief）。
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e6e5e1'
AXIS = '#c3c2b7'
SPLITS = [
    ('train', SPLIT_TRAIN, '训练集', '#2a78d6'),
    ('valid', SPLIT_VALID, '验证集', '#eb6834'),
    ('test', SPLIT_TEST, '测试集', '#1baf7a'),
]
QUANTS = [(0.25, '低波动（25 分位）'), (0.50, '典型（中位）'), (0.90, '高波动（90 分位）')]
H = 24


def target_std(raw, starts):
    """每个 (窗口, 站) 的目标段 72h 标准差，[M, N]。不整存 [M,72,N] 以省内存。"""
    s1 = np.zeros((len(starts), raw.shape[1]))
    s2 = np.zeros_like(s1)
    for k in range(SEQ):
        v = raw[starts + SEQ + k]
        s1 += v
        s2 += v * v
    mu = s1 / SEQ
    return np.sqrt(np.maximum(s2 / SEQ - mu * mu, 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--city', default='AMS')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or f'{BASE}/result/figs/{args.city.lower()}_split_typical_windows.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ds = EVDataset(feature='volume', auxiliary='None',
                   data_path=f'{BASE}/data/{args.city}_remove_const/')
    ds.split_cross_validation(fold=6, total_fold=6, train_ratio=0.8, valid_ratio=0.1)
    raw, T = ds.feat, ds.time
    # 每站哪些小时落在合成直线段里（与训练/评测同一实现），用来在图上标注。
    bad_hours = [straight_stretch_hours(raw[:, j]) for j in range(raw.shape[1])]

    fig, axes = plt.subplots(3, 3, figsize=(13.2, 8.6), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    picks = []
    for r, (key, sid, cname, color) in enumerate(SPLITS):
        st = ds.window_starts(sid, SEQ, SEQ)
        sd = target_std(raw, st).ravel()
        for c, (q, qname) in enumerate(QUANTS):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            k = int(np.argmin(np.abs(sd - np.quantile(sd, q))))
            i, j = np.unravel_index(k, (len(st), raw.shape[1]))
            x = raw[st[i]:st[i] + 2 * SEQ, j]
            t0 = str(T[st[i]])[:16]
            picks.append((cname, qname, int(st[i]), int(j), t0,
                          float(sd[k]), float(x[SEQ:].mean())))

            hx = np.arange(2 * SEQ)
            ax.plot(hx[:SEQ], x[:SEQ], color=color, lw=1.6, alpha=0.42,
                    solid_capstyle='round')
            ax.plot(hx[SEQ:], x[SEQ:], color=color, lw=2.0,
                    solid_capstyle='round')
            ax.axvline(SEQ - 0.5, color=AXIS, lw=1.0, ls=(0, (2, 3)))
            ax.grid(axis='y', color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            for s in ('left', 'bottom'):
                ax.spines[s].set_color(AXIS)
                ax.spines[s].set_linewidth(0.8)
            ax.tick_params(colors=INK2, labelsize=8, length=3)
            nbad = int(bad_hours[j][st[i]:st[i] + 2 * SEQ].sum())
            title = (f'{qname} · 目标 std {sd[k]:.2f}\n'
                     f'站列 #{j} · {t0} · 目标均值 {x[SEQ:].mean():.1f}')
            if nbad:
                title += f'\n合成直线段 {nbad}/{2 * SEQ}h'
            ax.set_title(title, fontsize=9, color=INK2, loc='left', pad=4)
            if c == 0:
                ax.set_ylabel(f'{cname}\n每小时充电量（kWh）', fontsize=9.5, color=INK,
                              fontweight='bold')
            if r == 2:
                ax.set_xticks(np.arange(0, 2 * SEQ + 1, H))
                ax.set_xlabel('窗口内小时数（0–72 历史，72–144 目标）',
                              fontsize=9, color=INK2)
            ax.set_xlim(0, 2 * SEQ - 1)

    fig.suptitle(f'{args.city}：训练 / 验证 / 测试三段的典型窗口',
                 fontsize=14, color=INK, x=0.06, ha='left', y=0.985)
    fig.text(0.06, 0.945,
             '把每个 (窗口, 站) 按目标段 72h 的标准差排序，取 25% / 50% / 90% 三个位置的原样序列；'
             '每格纵轴各自独立。',
             fontsize=9.5, color=INK2, ha='left')
    legend = [Line2D([0], [0], color=INK2, lw=1.6, alpha=0.42, label='历史 72h（模型输入）'),
              Line2D([0], [0], color=INK2, lw=2.0, label='目标 72h（要预测）')]
    fig.legend(handles=legend, loc='upper right', bbox_to_anchor=(0.985, 0.975),
               frameon=False, fontsize=9, ncol=1)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f'[fig] {out}')
    for p in picks:
        print('  %-4s %-14s 窗起始=%d 站列=%d %s std=%.3f 均值=%.2f' % p)


if __name__ == '__main__':
    main()
