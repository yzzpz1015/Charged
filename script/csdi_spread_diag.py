# -*- coding: utf-8 -*-
"""做法④ 的离散度体检：扩散器生成的样本，够不够散？

为什么单独测这个 —— PICP/MPIW 是唯一两个「只看分布宽度、不看中心」的指标，而报告里
做法④ 的 PICP@50/80/90 = 0.21/0.35/0.42（该是 0.50/0.80/0.90），**严重欠覆盖**；
MPIW@50 = 2.79，比同主干分位头的 4.47 窄了近四成。两个指标一致指向"太窄"。

但"太窄"有两个完全不同的成因，修法相反，所以必须先分开：

  (A) **数值欠离散**：扩散器在"该有电"的格子上生成的正值本身波动太小。判据是
      :math:`\\mathrm{std}_S(z)` —— 扩散输出是标准化残差，训练期它的真值标准差
      按构造就是 1，所以生成样本的 std 应当≈1。小于 1 就是欠离散。
  (B) **零格污染**：p_on 在真零格上中位 0.737，于是约 3/4 的样本在零格生成出一个
      正值 —— 样本分布变成"一堆 0 + 一堆正值"的双峰。这会让 MPIW 变**宽**而不是
      变窄，方向相反。

(A) 要靠改扩散（步数/schedule/训练）；(B) 要靠改 on/off。本脚本把两者拆开量。

用法：``python3 script/csdi_spread_diag.py [city] [device]``
"""
import os
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import kairos_prob_eval as ke                                        # noqa: E402
from api.model.csdi import base_cond, pack_cond, hist_feats           # noqa: E402

CITY = sys.argv[1] if len(sys.argv) > 1 else "JHB"
DEV = torch.device(f"cuda:{sys.argv[2] if len(sys.argv) > 2 else 0}")
NWIN, NS = 40, 24
STRIDES = [int(s) for s in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["50"])]

BUCKETS = [(0, 12, "H1-12"), (12, 24, "H13-24"), (24, 48, "H25-48"), (48, 72, "H49-72")]


def main():
    core = ke.load_csdi(CITY, DEV)
    ds, raw, starts = ke.load_city(CITY)
    N = raw.shape[1]
    hid, H = core["hid"], ke.SEQ_L
    off = np.arange(H)[None, :]
    bb = ke.build_kairos(CITY, "plain", f"{BASE}/result/kairos_l1", DEV, 99)

    from api.dataset.common import SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST
    _sp = (sys.argv[4] if len(sys.argv) > 4 else "te")
    ste = ds.window_starts({"tr": SPLIT_TRAIN, "va": SPLIT_VALID,
                            "te": SPLIT_TEST}[_sp], ke.SEQ_L, ke.SEQ_L)
    ste = ste[::max(1, len(ste) // NWIN)][:NWIN]
    M = len(ste)
    P = ke.point_pass(bb, raw, ste, list(range(H)), DEV)
    pred = np.stack([P[h] for h in range(H)], 2)[:, hid, :]
    obs = np.transpose(raw[ste[:, None] + H + off], (0, 2, 1))[:, hid, :]
    hist = np.transpose((raw[ste[:, None] + np.arange(H)[None, :]] > 0), (0, 2, 1))[:, hid, :]
    hod = np.stack([ds.time[ste + H + h].hour.values for h in range(H)], 1).astype(np.float32)
    pos = obs > 0

    ca, cb = core["cal"]
    pr = torch.from_numpy(np.ascontiguousarray(pred)).float().to(DEV)
    hb = torch.from_numpy(hod).float().to(DEV)[:, None, :]
    a_, o_ = hist_feats(torch.from_numpy(hist).float().to(DEV))
    base = base_cond(pr, torch.from_numpy(np.asarray(core["scale"])[hid]).float().to(DEV),
                     hb, a_, o_, N, H)
    with torch.no_grad():
        p_on = torch.sigmoid(ca * core["ono"](base) + cb)
    cond = pack_cond(base, p_on, N, H, core.get('cstats'))

    # 真值的标准化残差：**同一个 pred、同一个 (m,s)**，所以与生成样本严格同尺度可比。
    # 不拿"训练期 std=1"当参照 —— 测试期的残差本来就可能更大或更小，那会让判据自欺。
    zt = ((obs - pred) - np.asarray(core["bias"])[hid][None, :, None]) \
        / np.asarray(core["scale"])[hid][None, :, None]

    print(f"\n{CITY}  做法④ 离散度体检（{M} 窗 × {NS} 样本，伯努利掩码）")
    print("z = 标准化残差。生成样本与真值 z 同尺度，直接对撞。")
    print("**判据是 pooled/pooled**：生成样本的 pooled std（跨样本×跨格一起算）对真值的 "
          "pooled std。\n曾经这里拿『生成的逐格 std_S』对『真值的 pooled std』——一个是格内、"
          "一个是格内+格间，\n分母天生偏大，读数会系统性低估，已改。")
    print(f"\n{'跳步':<7}{'段':<9}{'生成pooled':>12}{'真值pooled':>12}{'比值':>8}"
          f"{'生成跨样本':>11}{'生成|零格':>11}{'生成格占比':>12}")
    for stride in STRIDES:
        Z = []
        for c0 in range(0, M, 20):
            hi = min(c0 + 20, M)
            with torch.no_grad():
                mm = (torch.rand(p_on[c0:hi].shape, device=DEV,
                                 generator=torch.Generator(device=DEV).manual_seed(c0))
                      < p_on[c0:hi]).float()
                smp = core["dif"].sample(core["net"], cond[c0:hi], mm, torch.zeros_like(mm),
                                         n_samples=NS, stride=stride,
                                         generator=torch.Generator(device=DEV).manual_seed(0))
            Z.append(smp[:, :, 0].cpu().numpy())          # [S,b,N,H] 标准化残差
        z = np.concatenate(Z, 1)                          # [S,M,N,H]
        tag = f"{stride}({len(range(core['dif'].steps - 1, -1, -stride))}步)"
        for a, b, nm in BUCKETS:
            pk, zk = pos[:, :, a:b], z[:, :, :, a:b]      # [M,N,Hb] / [S,M,N,Hb]
            gp = zk[:, pk].std()                          # 生成 pooled：跨样本×跨格
            tp = zt[:, :, a:b][pk].std()                  # 真值 pooled：同一个口径
            wc = zk[:, pk].std(0).mean()                  # 生成跨样本：每格 std_S 再平均
            print(f"{tag if nm == BUCKETS[0][2] else '':<7}{nm:<9}"
                  f"{gp:>12.3f}{tp:>12.3f}{gp / tp:>8.2f}{wc:>11.3f}"
                  f"{zk[:, ~pk].std(0).mean():>11.3f}{(zk > 0).mean():>12.3f}")
        gp, tp = z[:, pos].std(), zt[pos].std()
        print(f"{'':<7}{'**整段**':<9}{gp:>12.3f}{tp:>12.3f}{gp / tp:>8.2f}"
              f"{z[:, pos].std(0).mean():>11.3f}")

    pv = p_on[:, 0].cpu().numpy()
    print(f"\n判据 = 生成pooled / 真值pooled，≈1 才算离散度够。"
          f"若各跳步几乎不变 → 欠离散出在模型侧，与 DDIM 步数无关。")
    print(f"『生成跨样本』单看：它要接近的不是真值 pooled，而是**条件分箱后的**格内 std"
          f"（见 csdi_cond_spread_ref.py，≈0.87×真值pooled）；窄于它 = 样本互相抄。")


if __name__ == "__main__":
    main()
