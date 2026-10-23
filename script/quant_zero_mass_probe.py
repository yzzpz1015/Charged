# -*- coding: utf-8 -*-
"""做法③（分位头）的输出层有没有把"零质量"补上——主干那笔账到底要不要修。

上一支探针量到：主干点预测在**六个城市里一个 0 都输出不了**，真零格上的预测值
是典型真值的 15%~77%。但那**未必是病**——点预测是条件均值，零膨胀下条件均值本来就
落在 0 与非零之间。真正的判据是**输出层有没有点质量**：

* 若 ③ 能在"真值常常是 0"的格子上把分位数压到 0（即隐含 P(y=0) 很高），
  那么主干的连续输出**没问题**，零质量由输出层补上了 —— 主干不用修；
* 若 ③ 的隐含 P(y=0) 在所有格子上都≈0，那它跟主干一样给不出零，
  主干/输出层才真有账要算。

量法：分位头输出 99 个均匀分位（1%..99%）。某格上 **q≤0 的分位个数 / 99** 就是该格
隐含的 P(y=0)。把它在"真为零"和"真非零"两批格子上分开放，看它会不会分开。
"""
import importlib.util
import os
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import kairos_prob_eval as ke                                        # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "prob_baseline", os.path.join(BASE, "script", "prob_baseline.py"))
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)

CITIES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["LOA", "JHB", "MEL", "AMS", "SPO", "SZH"]
DEV = torch.device(f"cuda:{sys.argv[2] if len(sys.argv) > 2 else 0}")
NWIN = 300
TOL = 1e-6


def main():
    print("=" * 112)
    print("做法③ 分位头的隐含 P(y=0)：输出层补没补上零质量")
    print("=" * 112)
    print(f"\n{'城':<5}{'真零格隐含P0':>14}{'真非零格隐含P0':>16}{'分离度':>10}"
          f"{'真零格q50=0比':>15}{'真零格q90=0比':>15}{'PICP@50':>10}")
    for c in CITIES:
        ds, raw, starts = ke.load_city(c)
        N = raw.shape[1]
        y0, t0 = pb.load(c)
        ok, trm, vam, tem, u, span = pb.split_rows(y0, t0)
        cols, smean = pb.live_cols(y0, trm)
        y = y0[:, cols]

        model = ke.build_kairos(c, "quant", f"{BASE}/result/kairos_l1", DEV, 99)
        st = starts["te"]
        st = st[::max(1, len(st) // NWIN)][:NWIN]
        M = len(st)
        off = np.arange(ke.SEQ_L)[None, :]

        # 一次前向拿全 72 步的分位数：[M, N, 72, Q]
        Q = {}
        with torch.no_grad():
            for c0 in range(0, M, 8):
                cs = st[c0:c0 + 8]
                X = raw[cs[:, None] + off]
                X = np.transpose(X, (0, 2, 1)).reshape(-1, ke.SEQ_L).astype(np.float32)
                Pr = ke._forward(model, X, DEV).reshape(len(cs), N, ke.SEQ_L, -1)
                for j in (0,):
                    Q[c0] = Pr[:, :, j, :]          # _forward 已经转成 numpy
        P = np.concatenate([Q[k] for k in sorted(Q)], 0)          # [M,N,Q]

        truth = np.transpose(y[st[:, None] + ke.SEQ_L
                                + np.zeros((M, 1), int)], (0, 2, 1))[:, :, 0]  # [M,N]
        z = truth == 0

        nq = P.shape[-1]
        zero_frac = (P <= TOL).mean(-1)                            # [M,N] 隐含 P(y=0)
        p0_z = zero_frac[z].mean()
        p0_nz = zero_frac[~z].mean()
        q50_zero = float((np.abs(P[:, :, nq // 2]) <= TOL)[z].mean())
        q90_zero = float((np.abs(P[:, :, int(nq * 0.9)]) <= TOL)[z].mean())
        picp50 = float(np.mean((truth >= P[:, :, nq // 4])
                               & (truth <= P[:, :, 3 * nq // 4])))
        print(f"{c:<5}{p0_z:>14.3f}{p0_nz:>16.3f}{p0_z - p0_nz:>10.3f}"
              f"{q50_zero:>15.3f}{q90_zero:>15.3f}{picp50:>10.3f}")
    print("\n列含义（在 {NWIN} 个 test 窗、H=1 那一格上量）：")
    print("  真零格隐含P0   = 真值为 0 的格子上，99 个分位里 ≤0 的占了多少 → 该格隐含的 P(y=0)")
    print("  真非零格隐含P0 = 同上，但在真值非零的格子上（这两行的**差**才是分辨力）")
    print("  真零格q50=0比  = 真零格中，中位数分位就压到 0 的比例（0.5 以上 = 确实敢说'这格是 0'）")
    print("  PICP@50       = [q25,q75] 覆盖真值的比例，名义值是 0.50")


if __name__ == "__main__":
    main()
