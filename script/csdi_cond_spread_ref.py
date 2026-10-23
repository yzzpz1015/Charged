# -*- coding: utf-8 -*-
"""真值残差的**条件离散度**参照线：条件本身能把分布压多窄？

为什么必须先量这个再骂模型 —— 我们一直在拿"生成样本的 std"去比"真值残差的 **边际**
std"，但模型学的是 **条件**分布 p(x0 | pred, 尺度, 时段, 活动度, p_on)。条件越有信息，
正确的条件分布就越窄。若不先量出这条参照线，就可能把"模型是对的"误判成"欠离散"。

做法：把测试窗口的标准化残差 z 按 (站, 一天中的小时) 分箱，算**箱内** std 的平均，
再跟全部 z 的 std 比。这个比值就是"只靠 (站,小时) 这张气候学表能把分布压到多窄"，
是任何只看得见 (站,小时) 的模型的**理论上界参照**（真实模型还能用点预测/活动度再压一点，
但 memory 里 prob_cond_probe 已量过：再压只有 11% 的额外方差）。

判据：
  * 参照比值 ≈ 生成比值 → 扩散器的离散度**是对的**，PICP 欠覆盖另有原因（偏向/形状）
  * 参照比值 ≈ 1（条件压不动）而生成只有 0.3 → 扩散器确实欠离散，锅在模型
"""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

CITY = sys.argv[1] if len(sys.argv) > 1 else "JHB"


def main():
    from api.dataset.common import SPLIT_TEST
    import torch
    import kairos_prob_eval as ke

    ds, raw, _ = ke.load_city(CITY)
    N = raw.shape[1]
    DEV = torch.device("cpu")
    ste = ds.window_starts(SPLIT_TEST, ke.SEQ_L, ke.SEQ_L)
    ste = ste[::max(1, len(ste) // 400)][:400]
    H = ke.SEQ_L
    off = np.arange(H)[None, :]
    bb = ke.build_kairos(CITY, "plain", f"{BASE}/result/kairos_l1",
                         torch.device("cuda:0"), 99)
    P = ke.point_pass(bb, raw, ste, list(range(H)), torch.device("cuda:0"))
    pred = np.stack([P[h] for h in range(H)], 2)
    obs = np.transpose(raw[ste[:, None] + H + off], (0, 2, 1))
    pos = obs > 0

    # 用**测试期自己的**均值/尺度做标准化，这样 z 的条件结构不被训练期统计搅进来。
    # （参照线要答的是"条件能压多窄"，与 m/s 的取值无关。）
    r = obs - pred
    m = r.mean(axis=(0, 2))[None, :, None]
    s = np.maximum(r.std(axis=(0, 2)), 1e-3)[None, :, None]
    z = (r - m) / s

    hod = np.stack([ds.time[ste + H + h].hour.values for h in range(H)], 1)      # [M,H]
    print(f"\n{CITY}  真值残差的条件离散度参照（{len(ste)} 窗 × {N} 站 × {H} 步）")
    print("z = (真值 − 点预测) 按测试期自身均值/尺度标准化。")
    print("看的是**真为正**的格子（零格是另一个故事，不掺进来）。\n")
    print(f"{'时段':<8}{'边际 std(z)':>13}{'箱内 std 均值':>15}{'比值':>8}   说明")
    for a, b, nm in ((0, 12, "H1-12"), (12, 24, "H13-24"),
                     (24, 48, "H25-48"), (48, 72, "H49-72")):
        zz = z[:, :, a:b]
        pk = pos[:, :, a:b]
        hh = hod[:, a:b]
        marg = zz[pk].std()
        # 箱 = (站, 小时)，箱内 std 的平均 = 条件 std 的估计
        keep, within = [], 0.0
        for j in range(N):
            for h in range(hh.shape[1]):
                v = zz[:, j, h][pk[:, j, h]]
                if v.size >= 20:
                    within += v.size * v.std()
                    keep.append(v.size)
        within = within / sum(keep)
        print(f"{nm:<8}{marg:>13.3f}{within:>15.3f}{within / marg:>8.2f}"
              f"   {len(keep)} 个箱")
    zz, pk = z, pos
    hh = hod
    marg = zz[pk].std()
    tot = w = 0.0
    for j in range(N):
        for h in range(H):
            v = zz[:, j, h][pk[:, j, h]]
            if v.size >= 20:
                tot += v.size * v.std()
                w += v.size
    print(f"\n**参照线** 整段：边际 std {marg:.3f} → 按 (站,小时) 分箱后 {tot / w:.3f}"
          f"，比值 {tot / w / marg:.2f}")
    print(f"扩散器实测生成/真值 = 0.28（见 csdi_spread_diag）。"
          f"参照线若接近 0.28，说明它**离散度是对的**。")


if __name__ == "__main__":
    main()
