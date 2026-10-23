# -*- coding: utf-8 -*-
"""训练 CSDI 整块扩散头（Kairos-L1 主干冻结）。

它在整条比较线上的位置
----------------------
做法①：Kairos 点预测 + **抽历史残差整条向量**  -> 联合分布是经验来的，没有参数
做法③：Kairos 主干 + 分位头                -> 只有边际，站间耦合必须人为补
本脚本：Kairos 主干 + **整块 [站 × 未来] 扩散** -> 联合分布是学出来的

所以这是第一个「站间关系由模型自己负责」的方法。评测时 CRPS_sum 在它身上重新变成
考模型能力（见 memory quantile-coupling-crps-sum 里留的边界条件）。

数据流（每一步的形状）
----------------------
raw          [T, N]           城市原始用量，列已按 live_cols 剔过恒值站
starts       [M]              窗口起点（全局下标），来自 split_window_starts
pred         [M, N, 72]       冻结主干的点预测（评测时用同一份，见 kairos_prob_eval.point_pass）
obs          [M, N, 72]       真值
resid        [M, N, 72]       = obs - pred
z            [M, N, 72]       = (resid - m_j) / s_j，逐站去偏去尺度，扩散在它上面跑
pos          [M, N, 72] bool  该格是否为正（on/off 的监督信号）

站点轴必须按 Hilbert 空间序排 —— 卷积要在站轴上做「相邻即空间相邻」。列序存进 ckpt，
评测时照它重排，防止两边错位。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "script"))

import kairos_prob_eval as ke                                        # noqa: E402
from api.model.csdi import (ScoreNet, OnOffNet, GaussianDiffusion,   # noqa: E402
                            base_cond, pack_cond, hilbert_order,
                            hist_feats, LS, COND_NAMES)
from api.dataset.common import SPLIT_TRAIN, SPLIT_VALID              # noqa: E402
from api.utils import straight_window_keep                           # noqa: E402

SEQ_L = ke.SEQ_L
OUT = f"{BASE}/result/csdi"


def ece(p, g, nb: int = 15):
    """期望校准误差：把预测概率分箱，量「平均预测」与「实际频率」差多少。

    BCE 只看平均损失，会掩盖"整体均值对、但两端全错"这种情形（旧的分类器正是如此：
    预测正例率 0.841 对真值 0.857 看着没问题，极端自信的格子却占了 1/3 的损失）。
    ECE 直接盯校准，是判断 p_on 能不能当概率用的那个指标 —— 采样时它是拿来做
    伯努利抽样的，所以**校准比准确更重要**。
    """
    e = 0.0
    edges = torch.linspace(0, 1, nb + 1, device=p.device)
    for i in range(nb):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.any():
            e += float(m.sum()) * abs(float(p[m].mean()) - float(g[m].mean()))
    return e / p.numel()


def spatial_perm(city, cols):
    """站点的 Hilbert 空间序，返回作用在 ``ds.feat`` 列上的重排下标。

    站点 id 列名不统一：LOA 叫 ``site_id``，JHB/SZH 叫 ``site``（历史遗留）。
    """
    sites = pd.read_csv(f"{BASE}/data/{city}_remove_const/sites.csv")
    key = "site_id" if "site_id" in sites.columns else "site"
    sites = sites.set_index(key)
    ids = [int(c) for c in cols]
    lon = sites.loc[ids, "longitude"].values
    lat = sites.loc[ids, "latitude"].values
    return hilbert_order(lon, lat)


def prepare(city, model, device):
    """跑一遍冻结主干，把训练/验证段的 pred / obs / 目标小时都备好（留在 CPU）。"""
    ds, raw, _ = ke.load_city(city)
    # 逐站剔除合成直线段（站×窗掩码）。必须在转 float32 之前用原始 float64 判，
    # 否则大数值上的等增量误差会被放大到容差之外。
    s = {"tr": ds.window_starts(SPLIT_TRAIN, SEQ_L, SEQ_L),
         "va": ds.window_starts(SPLIT_VALID, SEQ_L, SEQ_L)}
    keep = {k: straight_window_keep(ds.feat, s[k], SEQ_L, SEQ_L) for k in ("tr", "va")}
    for k in ("tr", "va"):
        print(f"  [{city}] clean {k}: 保留 {100 * keep[k].mean():.1f}% 的站×窗", flush=True)
    raw = np.asarray(raw, np.float32)
    N = raw.shape[1]
    hid = spatial_perm(city, ds.feat_columns)
    off = np.arange(SEQ_L)[None, :]

    def blocks(starts):
        P = ke.point_pass(model, raw, starts, list(range(SEQ_L)), device)
        pred = np.stack([P[j] for j in range(SEQ_L)], axis=2)          # [M,N,72]
        obs = np.transpose(raw[starts[:, None] + SEQ_L + off], (0, 2, 1))
        # 历史窗口的 on/off 图案（**被测目标之外**的真实观测，因果合法）。分类器要靠它
        # 判断"这个站最近活跃吗"—— 这是 on/off 唯一可迁移的特征，点预测里没有。
        hist_on = np.transpose((raw[starts[:, None] + np.arange(SEQ_L)[None, :]] > 0),
                               (0, 2, 1))                              # [M,N,72]
        hod = np.stack([ds.time[starts + SEQ_L + h].hour.values
                        for h in range(SEQ_L)], axis=1).astype(np.float32)   # [M,72]
        return pred, obs, hod, hist_on

    t0 = time.time()
    d = {"N": N, "hid": hid, "s": s, "raw": raw}
    for k in ("tr", "va"):
        pred, obs, hod, hist_on = blocks(s[k])
        d[f"pred_{k}"], d[f"obs_{k}"], d[f"hod_{k}"], d[f"hon_{k}"] = pred, obs, hod, hist_on
    print(f"  [{city}] 备齐条件：训练 {len(s['tr'])} 窗 / 验证 {len(s['va'])} 窗，"
          f"{time.time() - t0:.0f}s，站 {N}", flush=True)

    # 逐站尺度只用**训练期**统计（碰验证/测试就是拿评测数据选尺度 = 泄漏）。
    # 合成直线段的残差是伪数据，均值/方差也只算干净的站×窗。
    d["keep_tr"], d["keep_va"] = keep["tr"], keep["va"]
    r = d["obs_tr"] - d["pred_tr"]
    kp = keep["tr"][:, :, None].astype(np.float32)                     # [M,N,1]
    # 分母必须是**格子数**（窗 × H），不是窗数。kp 只在站轴上取值，
    # 每个保留窗对每站贡献 H 个 horizon 格子；漏乘 H 会把 bias 放大 72 倍、
    # scale 放大 ~√72 倍（再叠 (72·μ)² 项），扩散样本的宽度和中心就全错。
    n = kp.sum(axis=(0, 2)) * SEQ_L                                    # [N] 格子数
    bias = (r * kp).sum(axis=(0, 2)) / np.maximum(n, 1)
    var = (((r - bias[None, :, None]) * kp) ** 2).sum(axis=(0, 2)) / np.maximum(n, 1)
    # 某站在训练段一个干净窗都没有时退回全量统计，避免除零。
    bad = n == 0
    if bad.any():
        bias[bad] = r[:, bad, :].mean(axis=(0, 2))
        var[bad] = r[:, bad, :].std(axis=(0, 2)) ** 2
    d["bias"] = bias                                                   # [N] 原始列序
    # 下限不能贴 0：训练期恒定/全零的站 var≈0，验证期一旦有值，
    # z 会被放大到 1e4 量级并独占平方损失。用「全体逐站残差 std 中位数」的
    # 2% 作下限（至少 1e-2），量纲随城市自适应。
    sd = np.sqrt(var)
    pos = sd[np.isfinite(sd) & (sd > 0)]
    floor = max(1e-2, 0.02 * float(np.median(pos))) if pos.size else 1e-2
    d["scale"] = np.maximum(sd, floor)
    # 落盘前自检：真实逐站残差 std 与存下来的 scale 必须是同一量级。
    true_sd = [r[keep["tr"][:, j], j, :].std() for j in range(N)
               if keep["tr"][:, j].sum() > 0]
    if true_sd:
        print(f"  [{city}] scale 自检：真实残差 std 中位 {np.median(true_sd):.3f} / "
              f"scale 中位 {np.median(d['scale']):.3f} "
              f"(比值 {np.median(d['scale']) / max(np.median(true_sd), 1e-9):.2f})，"
              f"下限 {floor:.3f}", flush=True)
    return d


def to_grid(arr, hid, device):
    """[M,N,H] + 空间序 -> [M,1,N,H] 的 GPU 张量。"""
    return torch.from_numpy(np.ascontiguousarray(arr[:, hid, :])).to(device).unsqueeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=["JHB"])
    ap.add_argument("--device", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=None, help="不给就由 --max_steps 折算")
    ap.add_argument("--max_steps", type=int, default=6000,
                    help="总优化步数上限。站数越多批越小、每 epoch 步数越多，"
                         "用步数而不是 epoch 数才能让六城看到的数据量可比。")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64, help="卷积通道数")
    ap.add_argument("--steps", type=int, default=1000,
                    help="DDPM 步数。**必须与 beta 表配套**：Σβ≈10 才让最后一步接近纯噪声"
                         "（1000 步 × β∈[1e-4,0.02] 正好）。见 api/model/csdi.py 类文档。")
    ap.add_argument("--cell_budget", type=int, default=300_000,
                    help="一个 batch 的格子数上限；批大小 = 它 // (站数*72)")
    ap.add_argument("--val_frac", type=float, default=1.0,
                    help="验证用多少比例的窗口（验证只用于挑 epoch，可抽样省时间）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT, help="权重输出目录（清理后重训用独立目录，别覆盖旧结果）")
    ap.add_argument("--ckpt_root", default=f"{BASE}/result/kairos_l1",
                    help="Kairos 点预测权重根目录；清理重训时指向 result/kairos_l1_clean")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    for city in args.cities:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        t0 = time.time()
        print(f"\n===== {city} =====", flush=True)
        bb = ke.build_kairos(city, "plain", args.ckpt_root, device, 99)
        if bb is None:
            print("  跳过：没有 Kairos-L1 权重"); continue
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
        # m_t/s_t 是 [1,N,1]，对 [M,1,N,H] 右对齐广播成 [M,1,N,H]。
        # **别写成 m_t[..., None]**：那会变成 [1,N,1,1]，站轴与通道轴撞上，
        # 静默broadcast 成 [M,N,N,H]（站数进了通道维），后面才在卷积处报错。
        Z = {k: (O[k] - P[k] - m_t) / s_t for k in ("tr", "va")}
        # 训练段实测 |z| < 10（裁了也不变）；验证段有 0.25% 的格子来自
        # 「训练期恒定、验证期突然有值」的站，z 可达 1e2，平方损失会让这
        # 一小撮不可学的格子主导早停。裁到 ±10 只影响验证损失的体检口径。
        Z = {k: Z[k].clamp(-10, 10) for k in ("tr", "va")}
        G = {k: (O[k] > 0).float() for k in ("tr", "va")}
        # 合成直线段的站×窗掩码（Hilbert 序），[M,1,N,1] 广播到 [B,1,N,H]。
        K = {k: torch.from_numpy(np.ascontiguousarray(d[f"keep_{k}"][:, hid]))
             .float().to(device)[:, None, :, None] for k in ("tr", "va")}
        # 目标小时对所有站相同，存 [M,H]；用的时候再展成 [b,N,H]（stack 不接受广播）。
        HD = {k: torch.from_numpy(d[f"hod_{k}"]).float().to(device) for k in ("tr", "va")}
        # 历史的 on/off 图案，[M,N,72]，已经按空间序重排（跟 P/O 同一套下标）。
        HON = {k: torch.from_numpy(np.ascontiguousarray(d[f"hon_{k}"][:, hid, :])).float().to(device)
               for k in ("tr", "va")}

        M_tr = len(d["s"]["tr"])
        spe = max(1, M_tr // bs)
        epochs = args.epochs or max(1, round(args.max_steps / spe))
        # 验证只用**等间隔**抽样的窗口：取前 N 个会因为验证段有季节漂移而偏，
        # 而挑 epoch 本来就只用来早停，抽样的代价又直接决定训练时长。
        M_va = len(d["s"]["va"])
        vi = np.arange(M_va)
        if args.val_frac < 1.0:
            vi = np.linspace(0, M_va - 1, max(1, int(M_va * args.val_frac))).round().astype(int)
        vi = torch.from_numpy(vi).to(device)
        # 验证**必须分块**：AMS 有 1385 站 × 510 验证窗，一次前向的特征图是
        # [510,64,1385,72] ≈ 13 GB，直接 OOM（后台那个五城任务正是死在 AMS 这一步，
        # LOA/MEL 已经训完存盘）。训练循环本来就被 bs 分块，漏的是这一段。
        vb = max(1, args.cell_budget // (N * H))
        # ECE 与 Platt 校准只吃**均匀抽样的子样本**：后者只拟合 2 个标量，全量既不必要，
        # 又要额外留一份计算图（AMS 上 lv 是 [510,1,1385,72]）。
        VCAP = max(1, 8_000_000 // (N * H))
        vc = vi[::max(1, len(vi) // VCAP)][:VCAP]

        def val_cond(vs):
            a_, o_ = hist_feats(HON["va"][vs])
            return base_cond(P["va"][vs, 0], s_t,
                             HD["va"][vs][:, None, :].expand(-1, N, H), a_, o_, N, H)

        # ---- 条件逐路标准化（只看训练段，与评测共用一个 stats）----
        # 六路量纲差两三个数量级（pred 是原始用电量 8~60，hod_sin 是 [-1,1]，x_t 是 O(1)），
        # 不标准化时去噪网络会被条件淹没、几乎不看输入噪声 —— 采样链于是塌成一条。
        # 统计量从**训练段**随机抽 64 窗算，存进 ckpt，评测侧读同一份（否则两边口径不同）。
        with torch.no_grad():
            i0 = torch.from_numpy(np.random.randint(0, M_tr, min(64, M_tr))).to(device)
            a0, o0 = hist_feats(HON["tr"][i0])
            b0 = base_cond(P["tr"][i0, 0], s_t,
                           HD["tr"][i0][:, None, :].expand(-1, N, H), a0, o0, N, H)
            # p_on 那一路拿**真值掩码**顶替（分类器还没训，拿不到 p_on）。必须逐格取，
            # 不能取"整窗一个标量均值再广播"——那样这一路在 64 个窗上的 std 只有 0.06，
            # 归一化后真实 p_on（逐格 std≈0.3）会被放大到 ±4σ，反而变成最霸道的那个通道，
            # 等于把刚修掉的问题换个通道又装回去。掩码 std≈0.35，是分类器输出的合理上界。
            c0 = pack_cond(b0, G["tr"][i0], N, H)
            cst = (c0.mean(dim=(0, 2, 3), keepdim=True),
                   c0.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6))
        print("  条件各路 (均值, 标准差)："
              + "  ".join(f"{n}=({float(cst[0][0, i]):.2f},{float(cst[1][0, i]):.2f})"
                          for i, n in enumerate(COND_NAMES)), flush=True)

        net = ScoreNet(cond_ch=7, base=args.base).to(device)
        ono = OnOffNet(cond_ch=6, base=args.base).to(device)
        dif = GaussianDiffusion(steps=args.steps).to(device)
        # 两个头**分开优化**：扩散损失的量级远大于 BCE，合在一起做
        # clip_grad_norm_ 会让分类器的梯度被扩散的范数挤掉，等于冻住它。
        opt_n = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        opt_o = torch.optim.AdamW(ono.parameters(), lr=args.lr, weight_decay=1e-4)
        sch_n = torch.optim.lr_scheduler.CosineAnnealingLR(opt_n, epochs)
        sch_o = torch.optim.lr_scheduler.CosineAnnealingLR(opt_o, epochs)
        npar = sum(p.numel() for p in net.parameters()) + sum(p.numel() for p in ono.parameters())
        print(f"  参数量 {npar / 1e6:.2f}M，扩散步 {args.steps}，"
              f"每 epoch {spe} 步 × {epochs} epoch，验证用 {len(vi)} 窗，"
              f"on/off 标签平滑 {LS}", flush=True)

        # **两个头各有各的 best**。原先只按验证扩散损失挑 checkpoint，而分类器的
        # 验证 BCE 从第 0 个 epoch 起就一路往上走（它在训练集上背答案：训练 BCE 0.088
        # / 验证 1.19），于是被挑中的那个 epoch 恰好是分类器最烂的那个 —— 这本身就是 bug。
        best_n, best_n_state = 1e9, None
        best_o, best_o_state = 1e9, None
        for ep in range(epochs):
            net.train(); ono.train()
            tot = n = 0
            for _ in range(spe):
                idx = torch.from_numpy(np.random.randint(0, M_tr, bs)).to(device)
                a_tr, o_tr = hist_feats(HON["tr"][idx])
                base = base_cond(P["tr"][idx, 0], s_t,
                                 HD["tr"][idx][:, None, :].expand(-1, N, H),
                                 a_tr, o_tr, N, H)
                g = G["tr"][idx]
                gs = g * (1 - 2 * LS) + LS          # 0 -> LS，1 -> 1-LS
                mk = K["tr"][idx]
                logit = ono(base)
                bce = F.binary_cross_entropy_with_logits(logit, gs, reduction="none")
                # 同 bias/scale 的分母坑：bce 是 [B,1,N,H]，mk 只在站轴上取值，
                # 分母必须乘 H。漏乘会被放大 72 倍，clip 之后分类器实际步长仍大 ~50 倍，
                # 表现为「验证 BCE 第 0 个 epoch 最好、之后一路变差」。
                bce = (bce * mk).sum() / (mk.sum() * H).clamp_min(1.0)
                cond = pack_cond(base, torch.sigmoid(logit).detach(), N, H, cst)
                zz = Z["tr"][idx] * g                       # 零格子真值恒 0
                # **损失算在全格上**（不是只算 gen_mask=g 的格子）。这一条是误差预算
                # 拆出来的：只训生成格时，分类器一旦误判为"有电"，扩散器在那格吐的是
                # 正值（实测 ≈8 原始单位）而不是 0 —— 因为从没有损失要求它在零格输出 0。
                # 全格参与后，零格的目标本身就是 0，被误生成时也只会吐 ≈0。实测（JHB，
                # 真值掩码 R² 0.5510 vs 全生成 0.3167，差额全部来自那 11% 误生成的零格）。
                loss = dif.p_loss(net, zz, cond, mask=mk)
                (loss + bce).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(ono.parameters(), 1.0)
                opt_n.step(); opt_o.step(); opt_n.zero_grad(); opt_o.zero_grad()
                tot += float(loss + bce); n += 1
            sch_n.step(); sch_o.step()

            net.eval(); ono.eval()
            dpm_sum = bce_sum = psum = cnt = 0.0
            with torch.no_grad():
                for c0 in range(0, len(vi), vb):
                    vs = vi[c0:c0 + vb]
                    bv = val_cond(vs)
                    lv = ono(bv)
                    gv = G["va"][vs]
                    mv = K["va"][vs]
                    # 两个损失都是**元素均值**：按元素数加权累加、最后除总数，与整段一次算等价。
                    # 合成直线段的格子两个损失都不计入（mask 加在分母上）。
                    # mv 是 [B,1,N,1]，广播到 [B,1,N,H]；分母要按格子数（×H）算，
                    # 不能只数 [B,N] —— 否则 BCE/正例率会被放大 H 倍（踩过）。
                    ncell = float(mv.sum()) * H
                    eg = F.binary_cross_entropy_with_logits(lv, gv, reduction="none")
                    bce_sum += float((eg * mv).sum())
                    dpm_sum += float(dif.p_loss(net, Z["va"][vs] * gv,
                                                pack_cond(bv, torch.sigmoid(lv), N, H, cst),
                                                mask=mv)) * ncell
                    psum += float((torch.sigmoid(lv) * mv).sum())
                    cnt += ncell
                # 验证 BCE 用**未平滑**的真值报，才跟常数基准可比（平滑会人为抬高它）。
                bce_v, dpm_v = bce_sum / cnt, dpm_sum / cnt
                bvc = val_cond(vc)
                gvc = G["va"][vc]
                mvc = K["va"][vc]
                pvc = torch.sigmoid(ono(bvc))
                ece_v = ece(pvc, gvc)          # 校准误差仍是全格体检，不参与早停
                gvc_mean = float((gvc * mvc).sum() / (mvc.sum() * H).clamp_min(1.0))
            if dpm_v < best_n:
                best_n = dpm_v
                best_n_state = {k: v.detach().cpu().clone()
                                for k, v in net.state_dict().items()}
            if bce_v < best_o:
                best_o = bce_v
                best_o_state = {k: v.detach().cpu().clone()
                                for k, v in ono.state_dict().items()}
            if ep % 10 == 0 or ep == epochs - 1:
                print(f"  ep{ep:4d} 训练 {tot / n:.4f} | 验证扩散 {dpm_v:.4f} "
                      f"BCE {bce_v:.4f}(最好 {best_o:.4f}) 校准误差 {ece_v:.4f} "
                      f"预测正例率 {psum / cnt:.3f}/真值 "
                      f"{gvc_mean:.3f} ({time.time() - t0:.0f}s)", flush=True)

        net.load_state_dict(best_n_state)
        ono.load_state_dict(best_o_state)

        # ---- Platt 校准（温标 + 偏置）----
        # 分类器挑到最好的 epoch 后仍有系统性偏移：预测正例率明显低于真值（实测 JHB
        # 0.782 vs 0.857）。采样时 p_on 是拿来做**伯努利抽样**的，偏移直接变成
        # "把真为正的格子钉成 0"，而一个 0 的质点在 CRPS 上代价很大。所以这里在
        # **校验段**上拟一对 (a,b) 把 logit 拉回校准 —— 校验段在测试段之前，且只用来
        # 定两个标量，不构成泄漏（与 memory 里"用 valid 做区间校准"是同一条口径）。
        # 只在**子样本** vc 上拟合：参数就 2 个，全量既不必要又会多留一份计算图。
        with torch.no_grad():
            lv = ono(bvc).detach()
            gv = gvc
            mcal = mvc.expand_as(gv).bool()
            lv_f, gv_f = lv[mcal], gv[mcal]      # 只在干净格上拟校准
        ab = torch.tensor([1.0, 0.0], device=device, requires_grad=True)
        opt_c = torch.optim.Adam([ab], lr=0.05)
        for _ in range(400):
            opt_c.zero_grad()
            # a 走 softplus 保证恒正：负的 a 会把 on/off 的顺序整个翻过来。
            F.binary_cross_entropy_with_logits(
                F.softplus(ab[0]) * lv_f + ab[1], gv_f).backward()
            opt_c.step()
        with torch.no_grad():
            scale_a, shift_b = float(F.softplus(ab[0])), float(ab[1])
            pc = torch.sigmoid(scale_a * lv_f + shift_b)
            print(f"  [{city}] Platt 校准 a={scale_a:.3f} b={shift_b:+.3f}；"
                  f"正例率 {float(torch.sigmoid(lv_f).mean()):.3f} → {float(pc.mean()):.3f}"
                  f"（真值 {float(gv_f.mean()):.3f}），验证 BCE {best_o:.4f} → "
                  f"{float(F.binary_cross_entropy(pc, gv_f)):.4f}", flush=True)
        torch.save(dict(net=net.state_dict(), ono=ono.state_dict(), hid=hid,
                        scale=d["scale"], bias=d["bias"], base=args.base,
                        steps=args.steps, city=city, n_station=N, pre_len=H,
                        cal=(scale_a, shift_b), param=ScoreNet.PARAM,
                        cond_stats=(cst[0].cpu(), cst[1].cpu())),
                   f"{args.out}/{city}_csdi.pth")
        json.dump(dict(city=city, n_station=N, best_val_diffusion=best_n,
                       best_val_bce=best_o, cal=[scale_a, shift_b],
                       val_bce_calibrated=float(F.binary_cross_entropy(pc, gv_f)),
                       args=vars(args)),
                  open(f"{args.out}/{city}_train.json", "w"), ensure_ascii=False, indent=1)
        print(f"  [{city}] 已存 {args.out}/{city}_csdi.pth（{time.time() - t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
