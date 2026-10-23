# -*- coding: utf-8 -*-
"""训练**吸收态（零膨胀）整块扩散**（Kairos-L1 主干冻结）。

它在比较线上的位置
------------------
做法①：点预测 + 抽历史残差整条向量   -> 联合经验来的，没有参数
做法③：主干 + 分位头                -> 只有边际，站间耦合必须人为补
做法④：主干 + 整块扩散 + **独立 on/off 分类器切硬掩码**
本脚本：主干 + 整块扩散 + **掩码与数值同一个网络联合建模，掩码沿采样链反复重抽**

做法④ 的病理（见 api/model/zidiff.py 开头的长注释）是：掩码在采样前抽一次就冻死，
而 on/off 的可预测性只有 BCE 0.396 对常数基准 0.410 —— 信号几乎为零。本脚本把那个
**硬决定**换成**边生成边决定**。

复用的部分（**刻意一个字都不改**，好让对比只隔离"掩码怎么定"）
--------------------------------------------------------------
* 数据准备、逐站去偏去尺度 z=(obs−pred−bias)/scale、Hilbert 站序、条件打包：直接
  从 train_csdi 导入 :func:`prepare` / :func:`to_grid` / :func:`ece`。
* 扩散调度（steps=1000，β 线性 1e-4→0.02，v-参数化）与做法④ 逐字相同。
* 条件只剩 6 路（``base_cond``）—— **没有 p_on 这一路**，因为本模型里不存在一个
  先于扩散产出的分类器。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import kairos_prob_eval as ke                                        # noqa: E402
from train_csdi import prepare, to_grid, ece                         # noqa: E402
from api.model.csdi import base_cond, hist_feats                     # noqa: E402
from api.model.zidiff import ZiScoreNet, ZiDiffusion                 # noqa: E402
from api.dataset.common import SPLIT_TRAIN, SPLIT_VALID              # noqa: E402

SEQ_L = ke.SEQ_L
OUT = f"{BASE}/result/csdi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=["SPO"])
    ap.add_argument("--device", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=6000,
                    help="与做法④ 同一步数上限，保证六城/多轮之间可比。")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--cell_budget", type=int, default=300_000)
    ap.add_argument("--w_mask", type=float, default=1.0,
                    help="掩码 BCE 的权重。值损失(v-预测 MSE)量级远大于 BCE，"
                         "若把掩码压得太狠，网络会只顾值不管生死；给个显式旋钮。")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT, help="权重输出目录（清理后重训用独立目录）")
    ap.add_argument("--ckpt_root", default=f"{BASE}/result/kairos_l1",
                    help="Kairos 点预测权重根目录；清理重训时指向 result/kairos_l1_clean")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    for city in args.cities:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        t0 = time.time()
        print(f"\n===== {city}（吸收态扩散）=====", flush=True)
        bb = ke.build_kairos(city, "plain", args.ckpt_root, device, 99)
        if bb is None:
            print("  跳过：没有 Kairos-L1 权重")
            continue
        d = prepare(city, bb, device)
        del bb
        torch.cuda.empty_cache()

        N, H, hid = d["N"], SEQ_L, d["hid"]
        bs = max(1, args.cell_budget // (N * H))
        print(f"  批大小 {bs} 窗（{bs * N * H} 格）", flush=True)

        m_t = torch.from_numpy(d["bias"][hid]).float().to(device)[None, :, None]
        s_t = torch.from_numpy(d["scale"][hid]).float().to(device)[None, :, None]
        P = {k: to_grid(d[f"pred_{k}"], hid, device) for k in ("tr", "va")}
        O = {k: to_grid(d[f"obs_{k}"], hid, device) for k in ("tr", "va")}
        Z = {k: (O[k] - P[k] - m_t) / s_t for k in ("tr", "va")}
        G = {k: (O[k] > 0).float() for k in ("tr", "va")}
        K = {k: torch.from_numpy(np.ascontiguousarray(d[f"keep_{k}"][:, hid]))
             .float().to(device)[:, None, :, None] for k in ("tr", "va")}
        HD = {k: torch.from_numpy(d[f"hod_{k}"]).float().to(device) for k in ("tr", "va")}
        HON = {k: torch.from_numpy(np.ascontiguousarray(d[f"hon_{k}"][:, hid, :])).float().to(device)
               for k in ("tr", "va")}
        # 值通道的目标：**非正格恒为 0**（与做法④ 完全一致的口径）。
        ZZ = {k: Z[k] * G[k] for k in ("tr", "va")}
        # 同 train_csdi：验证段 0.25% 的格子来自训练期恒定的站，z 可达 1e2；
        # 裁到 ±10 防止这一小撮不可学的尾巴主导早停（训练段 |z|<10，裁了不变）。
        ZZ = {k: ZZ[k].clamp(-10, 10) for k in ("tr", "va")}

        M_tr = len(d["s"]["tr"])
        spe = max(1, M_tr // bs)
        epochs = args.epochs or max(1, round(args.max_steps / spe))
        M_va = len(d["s"]["va"])
        vi = torch.from_numpy(np.arange(M_va)).to(device)
        vb = bs
        VCAP = max(1, 8_000_000 // (N * H))
        vc = vi[::max(1, len(vi) // VCAP)][:VCAP]

        def cond_of(k, idx):
            a_, o_ = hist_feats(HON[k][idx])
            return base_cond(P[k][idx, 0], s_t,
                             HD[k][idx][:, None, :].expand(-1, N, H), a_, o_, N, H)

        # ---- 条件逐路标准化（只看训练段；与做法④ 同一套做法）----
        # 做法④ 那边 7 路，这里 6 路（没有 p_on）。量纲差两三个数量级，不标准化
        # 会让首层卷积被条件主导、网络不听带噪输入 —— 那个坑做法④ 踩过，直接继承。
        with torch.no_grad():
            i0 = torch.from_numpy(np.random.randint(0, M_tr, min(64, M_tr))).to(device)
            b0 = cond_of("tr", i0)
            cst = (b0.mean(dim=(0, 2, 3), keepdim=True),
                   b0.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6))
        print("  条件六路 (均值, 标准差)："
              + "  ".join(f"{n}=({float(cst[0][0, i]):.2f},{float(cst[1][0, i]):.2f})"
                          for i, n in enumerate(
                              ("pred", "log_scale", "hod_sin", "hod_cos", "act24", "on_now"))),
              flush=True)

        net = ZiScoreNet(cond_ch=6, base=args.base).to(device)
        dif = ZiDiffusion(steps=args.steps).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        npar = sum(p.numel() for p in net.parameters())
        print(f"  参数量 {npar / 1e6:.2f}M（做法④ 是两网并联），扩散步 {args.steps}，"
              f"每 epoch {spe} 步 × {epochs} epoch", flush=True)

        best, best_state = 1e9, None
        for ep in range(epochs):
            net.train()
            tot_v = tot_g = n = 0
            for _ in range(spe):
                idx = torch.from_numpy(np.random.randint(0, M_tr, bs)).to(device)
                cond = (cond_of("tr", idx) - cst[0]) / cst[1]
                l_v, l_g = dif.p_loss(net, ZZ["tr"][idx], G["tr"][idx], cond,
                                      mask=K["tr"][idx])
                loss = l_v + args.w_mask * l_g
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                tot_v += float(l_v)
                tot_g += float(l_g)
                n += 1
            sch.step()

            net.eval()
            sv = sg = cnt = 0.0
            with torch.no_grad():
                for c0 in range(0, len(vi), vb):
                    vs = vi[c0:c0 + vb]
                    cond = (cond_of("va", vs) - cst[0]) / cst[1]
                    mv = K["va"][vs]
                    l_v, l_g = dif.p_loss(net, ZZ["va"][vs], G["va"][vs], cond, mask=mv)
                    nv = float(mv.sum())
                    sv += float(l_v) * nv
                    sg += float(l_g) * nv
                    cnt += nv
                v_v, v_g = sv / cnt, sg / cnt
                # 校准误差：把预测的 P(有电) 分箱，量它与真实频率差多少。做法④ 里
                # 它是选掩码阈值的依据；这里只是体检，不参与早停。
                condv = (cond_of("va", vc) - cst[0]) / cst[1]
                # 校准/正例率必须用**同分布**的带噪输入去探（mask_probe 的文档解释了
                # 为什么不能喂全零），否则报出来的是一组偏低的假数。
                lg = dif.mask_probe(net, ZZ["va"][vc], G["va"][vc], condv)
                ece_v = ece(torch.sigmoid(lg), G["va"][vc])      # 全格体检，不参与早停
                mvc = K["va"][vc].expand_as(lg).bool()
                pos_pred = float(torch.sigmoid(lg)[mvc].mean())
                pos_true = float(G["va"][vc][mvc].mean())
            # 早停综合两个损失：单看扩散损失会把掩码学坏的那一轮挑回来（做法④ 的
            # 分类器正是这样被挑中最烂的 epoch —— 那次教训写在 train_csdi.py 里）。
            score = v_v + args.w_mask * v_g
            if score < best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if ep % 10 == 0 or ep == epochs - 1:
                print(f"  ep{ep:4d} 训练 值{tot_v / n:.4f} 掩码{tot_g / n:.4f} | "
                      f"验证 值{v_v:.4f} 掩码{v_g:.4f} 校准误差 {ece_v:.4f} "
                      f"预测正例率 {pos_pred:.3f}/真值 {pos_true:.3f}"
                      f" ({time.time() - t0:.0f}s)", flush=True)

        net.load_state_dict(best_state)
        torch.save(dict(net=net.state_dict(), hid=hid, scale=d["scale"], bias=d["bias"],
                        base=args.base, steps=args.steps, city=city, n_station=N,
                        pre_len=H, param=ZiScoreNet.PARAM,
                        cond_stats=(cst[0].cpu(), cst[1].cpu())),
                   f"{args.out}/{city}_zidiff.pth")
        json.dump(dict(city=city, n_station=N, best_val=vars(args),
                       best_score=best, args=vars(args)),
                  open(f"{args.out}/{city}_zidiff_train.json", "w"), ensure_ascii=False, indent=1)
        print(f"  [{city}] 已存 {args.out}/{city}_zidiff.pth（{time.time() - t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
