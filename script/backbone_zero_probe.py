# -*- coding: utf-8 -*-
"""主干（Kairos-L1 点预测）在零值上的病：它到底有没有"毛毛雨式误报"。

背景：WWW'26 那篇（Energy-Efficient Training-Free Zero-Inflation Correction）说
TSFM 输出连续、必然在真零处产生小正值（drizzle-like false alarms），并波及非零强度校准。
**这一条在我们身上一直是推断、没有测量**。本脚本量它。

口径与主实验逐字一致：真值取 `prob_baseline.load` + `live_cols`（剔训练期退化站），
窗口取训练时落盘的 `starts_*.npy`（即模型真正预测的那批窗），
行 r = 窗 (r // N)、站 (r % N)。预测取 `predict_plain_pre72_fold6.npy`。

关心的三个数（每个都以城市自己的"典型非零量级"为尺子）：
  1. **误报率** 真零格上 pred 超过若干阈值的比例（阈值用 smean 的倍数，跨城可比）；
  2. **误报大小** 真零格上 pred 的均值 / 真非零格真值的均值——"误报相当于一个真值的几成"；
  3. **是否真会输出 0** 预测里恰好为 0 的比例（连续回归往往一个 0 都给不出）。
"""
import importlib.util
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

_spec = importlib.util.spec_from_file_location(
    "prob_baseline", os.path.join(BASE, "script", "prob_baseline.py"))
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)

CITIES = ["LOA", "JHB", "MEL", "AMS", "SPO", "SZH"]
SEQ_L = 72
PRE = 72
# 误报阈值：以每站训练期均值为尺子，取它的 1% / 5% / 20% 绝对量。
FRACS = [0.01, 0.05, 0.20]


def main():
    print("=" * 110)
    print("主干点预测的零值体检（Kairos-L1 plain，test 窗，pre_len=72）")
    print("=" * 110)
    hdr = (f"\n{'城':<5}{'真零率':>8}{'误报>1%smean':>12}{'>5%':>8}{'>20%':>8}"
           f"{'误报均值':>10}{'真非零均值':>11}{'误报/真值':>10}{'pred恰为0':>10}")
    print(hdr)
    for c in CITIES:
        y0, t0 = pb.load(c)
        ok, trm, vam, tem, u, span = pb.split_rows(y0, t0)
        cols, smean = pb.live_cols(y0, trm)
        y = y0[:, cols]
        N = len(cols)

        d = f"{BASE}/result/kairos_l1/{c}/"
        pred = np.load(d + f"predict_plain_pre{72}_fold6.npy")      # [M*N, 72]
        starts = np.load(d + f"starts_plain_pre{72}_fold6.npy")     # [M]
        M = len(starts)
        assert pred.shape[0] == M * N, (pred.shape, M, N)
        pred = pred.reshape(M, N, PRE)

        hh = np.arange(PRE)[None, :]
        truth = np.transpose(y[starts[:, None] + SEQ_L + hh], (0, 2, 1))   # [M,N,PRE]
        assert truth.shape == pred.shape

        sm_h = np.asarray(smean)[None, :, None]        # [1,N,1]，每站的尺子
        zero = truth == 0
        nz = ~zero
        # 误报：真零格上 pred 超过 smean 的若干比例
        rates = [(pred[zero] > f * np.broadcast_to(sm_h, pred.shape)[zero]).mean()
                 for f in FRACS]
        mz = pred[zero].mean()
        mn = truth[nz].mean()
        print(f"{c:<5}{zero.mean():>8.3f}{rates[0]:>12.3f}{rates[1]:>8.3f}"
              f"{rates[2]:>8.3f}{mz:>10.3f}{mn:>11.3f}{mz / mn:>10.3f}"
              f"{(pred == 0).mean():>10.4f}")
    print("\n列含义：")
    print("  误报>X%mean = 真零格中，pred 超过该站训练期均值 X% 的比例（跨城可比的误报率）")
    print("  误报均值    = 真零格上 pred 的平均值（绝对值，随城市量级变）")
    print("  误报/真值   = 上面那个数 ÷ 真非零格真值的平均值 —— **跨城可比**：")
    print("                0 = 主干在真零处确实说 0；越大 = 误报越像'一个真事件'")
    print("  pred恰为0   = 点预测里恰好等于 0 的比例（连续回归往往一个都不给）")


if __name__ == "__main__":
    main()
