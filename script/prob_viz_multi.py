# -*- coding: utf-8 -*-
"""批量画「多个城市 × 连续多窗口」：每城自动挑两个站——一个最爱归零的、一个几乎不归零的。

和 ``prob_viz.py`` 的关系：那边一次只画一个城、一个站（要么随机抽格子、要么自动挑一个）。
这里只做编排，采样逻辑一行不改，全部复用 ``collect`` / ``draw_timeseries``。

用法::

    python script/prob_viz_multi.py JHB,MEL,SPO 168 200
    #                           城市列表  跨度 起始窗口号

每城只 ``collect`` 一次（④ 的采样链最贵），两个站共用同一份 cell 缓存 —— 所以
第二个站几乎不要钱。
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prob_viz import BASE, collect, draw_timeseries          # noqa: E402

CITIES = sys.argv[1].split(',')
SPAN = int(sys.argv[2]) if len(sys.argv) > 2 else 168
START = int(sys.argv[3]) if len(sys.argv) > 3 else 200
DEV = int(sys.argv[4]) if len(sys.argv) > 4 else 0
H = 24

device = torch.device(f'cuda:{DEV}' if torch.cuda.is_available() else 'cpu')

for city in CITIES:
    t0 = time.time()
    print(f'[{city}] collect 开始 span={SPAN} start={START} device={device}', flush=True)
    cells = collect(city, H - 1, 0, 7, device, 'clf', (START, SPAN))
    if len(cells) < 48:                                      # 该城测试窗口不够长，跳过而不是崩
        print(f'[{city}] 只取到 {len(cells)} 个小时，太短，跳过（把 START 调小）', flush=True)
        continue
    T = np.stack([c['truth'] for c in cells])                # [span, N]
    zr = (T == 0).mean(0)
    rng_ = T.max(0) - T.min(0)

    spiky = int(np.argmax(zr))                               # 最爱归零的
    cand = np.where(zr <= np.percentile(zr, 10))[0]          # 最不归零的一群里，挑波动最大的
    smooth = int(cand[np.argmax(rng_[cand])])

    print(f'[{city}] 实际段长 {len(cells)} 小时，{T.shape[1]} 站；'
          f'选 站序{spiky}(零值{100*zr[spiky]:.0f}%) 与 站序{smooth}(零值{100*zr[smooth]:.0f}%)',
          flush=True)

    for sid in dict.fromkeys([spiky, smooth]):
        out = (f'{BASE}/result/figs/prob_series_{city}_H{H}'
               f'_s{START}+{SPAN}_st{sid}.png')
        draw_timeseries(cells, sid, city, H, out)
        print(f'[{city}] st{sid}  零值率{100*zr[sid]:>4.0f}%  真值max{T[:, sid].max():>6.1f}'
              f'  →  {os.path.relpath(out, BASE)}', flush=True)

    print(f'[{city}] 完成，用时 {time.time()-t0:.0f}s', flush=True)
