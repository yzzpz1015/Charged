# -*- coding: utf-8 -*-
"""GATE experiment: can an END-TO-END fine-tuned satellite encoder predict an
UNSEEN station's demand level, where a frozen encoder cannot?

This is the single remaining cell of the 2x2 (frozen vs end-to-end) x (our data).
Everything so far used frozen DINOv2 embeddings + a linear/nearest-neighbour
readout and got ~0.  Per the data-processing inequality, no amount of clever
fusion downstream of a FROZEN encoder can help, so the only untested hypothesis
is that the signal needs a task-reshaped representation.

K=0 setting (no history at all): the only per-station quantity available is the
image, so the task reduces to predicting the station's MEAN level.  Strict
leave-station-out.

Controls (all same folds / same budget):
  real      : image -> station mean, fine-tuned
  shuffled  : same but targets permuted across stations (null: no image->target
              relation exists, so any R2 > shuffled is real signal)
  frozen    : frozen frozen embedding + ridge (the readout we always used)
  randcode  : a random per-station vector instead of the image (capacity null)

Usage:  python3 script/sat_gate_finetune.py --city LOA --crop 192
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import torchvision.transforms as T

torch.set_num_threads(4)   # 多配置并行时避免线程互抢

BASE = "/home/data/yzzpz/Charged"
DINOV2_REPO = "/home/guest3/.cache/torch/hub/facebookresearch_dinov2_main"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- data
def load_targets(city):
    d = f"{BASE}/data/{city}_remove_const"
    v = pd.read_csv(f"{d}/volume.csv", index_col=0)
    v.index = pd.to_datetime(v.index)
    sites = pd.read_csv(f"{d}/sites.csv")
    sc = "site_id" if "site_id" in sites.columns else "site"
    ids = sites[sc].astype(int).tolist()
    wv = v[[str(s) for s in ids]].resample("W").sum()
    wv = wv[wv.sum(1) > 0]
    y = np.log1p(wv.values).mean(0).astype(np.float32)     # station mean level
    return ids, y


def load_images(city, ids, crop_px):
    """Center-crop each station mosaic to crop_px and cache as uint8 [N,3,H,W]."""
    mdir = f"{BASE}/tiles_dataset/{city}/mosaic/z16"
    out, missing = [], []
    for s in ids:
        p = os.path.join(mdir, f"{s}.0.png")
        if not os.path.exists(p):
            missing.append(s)
            out.append(None)
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size
        c = min(crop_px, w, h)
        im = im.crop(((w - c) // 2, (h - c) // 2, (w - c) // 2 + c, (h - c) // 2 + c))
        out.append(np.asarray(im, dtype=np.uint8))
    keep = [i for i, o in enumerate(out) if o is not None]
    if missing:
        print(f"[warn] {len(missing)} stations without mosaic -> dropped")
    X = np.stack([out[i] for i in keep])
    return torch.from_numpy(X).permute(0, 3, 1, 2).contiguous(), keep


# ---------------------------------------------------------------- model
_INIT_SD = None


def build_model(device, unfreeze_last=None):
    """unfreeze_last: None=全量微调; 0=只训 head; N=只解冻最后 N 个 block+norm"""
    global _INIT_SD
    m = torch.hub.load(DINOV2_REPO, "dinov2_vitb14", source="local", pretrained=True)
    if _INIT_SD is None:
        _INIT_SD = {k: v.detach().clone() for k, v in m.state_dict().items()}
    else:
        m.load_state_dict(_INIT_SD)          # 每折重置到同一初始权重
    if unfreeze_last is not None:
        for p in m.parameters():
            p.requires_grad = False
        if unfreeze_last > 0:
            for blk in m.blocks[-unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in m.norm.parameters():
                p.requires_grad = True
    m.to(device)
    head = nn.Linear(m.num_features, 1).to(device)
    return m, head


def forward_feats(backbone, x, bn_head, head):
    """x: [B,3,H,W] in [0,1]; returns prediction. bn_head stabilises the CLS."""
    f = backbone(x)
    return head(bn_head(f)).squeeze(-1)


# ---------------------------------------------------------------- train
def train_fold(Xtr, ytr, Xva, yva, Xte, yte, device, epochs, seed, lr_bb, lr_head,
               unfreeze_last=None, label=""):
    torch.manual_seed(seed)
    np.random.seed(seed)

    augment = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
    ])
    eval_tf = T.Compose([T.Resize(224)])
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    backbone, head = build_model(device, unfreeze_last)
    bn_head = nn.BatchNorm1d(backbone.num_features).to(device)

    ys = ytr.mean(), ytr.std() + 1e-6                     # standardise target
    ytr_n = (ytr - ys[0]) / ys[1]
    yva_n = (yva - ys[0]) / ys[1]

    bb_params = [p for p in backbone.parameters() if p.requires_grad]
    params = [{"params": bb_params, "lr": lr_bb},
              {"params": head.parameters(), "lr": lr_head},
              {"params": bn_head.parameters(), "lr": lr_head}]
    opt = torch.optim.AdamW(params, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    bs = min(32, len(Xtr))
    best_va, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        backbone.train(); head.train(); bn_head.train()
        perm = np.random.permutation(len(Xtr))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            xb = torch.stack([norm(augment(Xtr[j].float() / 255.0)) for j in idx]).to(device)
            yb = torch.tensor(ytr_n[idx], device=device)
            opt.zero_grad()
            loss = F.mse_loss(forward_feats(backbone, xb, bn_head, head), yb)
            loss.backward()
            opt.step()
        sched.step()
        if ep % 25 == 0:
            print(f"      [{label}] ep{ep:>3} loss={float(loss):.4f} best_val={best_va:.4f}", flush=True)

        backbone.eval(); head.eval(); bn_head.eval()
        with torch.no_grad():
            xv = torch.stack([norm(eval_tf(Xva[j].float() / 255.0)) for j in range(len(Xva))]).to(device)
            pv = forward_feats(backbone, xv, bn_head, head).cpu().numpy()
        vl = float(((pv - yva_n) ** 2).mean())
        if vl < best_va - 1e-5:
            best_va, bad = vl, 0
            best_state = ({k: v.detach().clone() for k, v in backbone.state_dict().items()},
                          {k: v.detach().clone() for k, v in head.state_dict().items()},
                          {k: v.detach().clone() for k, v in bn_head.state_dict().items()})
        else:
            bad += 1
            if bad >= 40:
                break

    backbone.load_state_dict(best_state[0]); head.load_state_dict(best_state[1]); bn_head.load_state_dict(best_state[2])
    backbone.eval(); head.eval(); bn_head.eval()
    with torch.no_grad():
        xt = torch.stack([norm(eval_tf(Xte[j].float() / 255.0)) for j in range(len(Xte))]).to(device)
        pt = forward_feats(backbone, xt, bn_head, head).cpu().numpy()
    return pt * ys[1] + ys[0], ep + 1


# ---------------------------------------------------------------- metrics
def r2(y, yh):
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - yh) ** 2).sum() / ss if ss > 1e-12 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="LOA")
    ap.add_argument("--crop", type=int, default=192, help="中心裁剪像素(=VisionEV 380m 视野)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr_bb", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.15, help="训练站里再留出的早停站")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unfreeze_last", type=int, default=2, help="只解冻最后 N 个 block; 0=只训 head; -1=全量")
    args = ap.parse_args()
    unf = None if args.unfreeze_last < 0 else args.unfreeze_last

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids, y = load_targets(args.city)
    Ximg, keep = load_images(args.city, ids, args.crop)
    ids = [ids[i] for i in keep]; y = y[keep]
    n = len(y)
    print(f"[{args.city}] n={n} stations  crop={args.crop}px  target=log mean weekly volume")
    print(f"  y: mean={y.mean():.3f} std={y.std():.3f}\n")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    res = {k: [] for k in ("real", "shuffled", "frozen")}
    oof = {k: np.full(n, np.nan) for k in ("real", "shuffled", "frozen")}
    for fi, (tr_all, te) in enumerate(kf.split(np.arange(n))):
        rng = np.random.RandomState(fi)
        nval = max(4, int(args.val_frac * len(tr_all)))
        vi = rng.choice(len(tr_all), nval, replace=False)
        va, tr = tr_all[vi], np.setdiff1d(tr_all, tr_all[vi])

        # --- real ---
        pr, ep = train_fold(Ximg[tr], y[tr], Ximg[va], y[va], Ximg[te], y[te],
                            device, args.epochs, args.seed*100+fi, args.lr_bb, args.lr_head, unf, label="real")
        # --- shuffled-label null (same folds, same budget) ---
        ysh = y.copy(); rng.shuffle(ysh)
        ps, _ = train_fold(Ximg[tr], ysh[tr], Ximg[va], ysh[va], Ximg[te], ysh[te],
                           device, args.epochs, args.seed*100+fi, args.lr_bb, args.lr_head, unf, label="shuf")
        # --- frozen + ridge ---
        with torch.no_grad():
            feats = []
            backbone, _ = build_model(device); backbone.eval()
            norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ev = T.Compose([T.Resize(224)])
            for i in range(0, n, 64):
                xb = torch.stack([norm(ev(Ximg[j].float() / 255.0)) for j in range(i, min(i + 64, n))]).to(device)
                feats.append(backbone(xb).cpu().numpy())
        Fz = np.concatenate(feats)
        Fz = Fz / (np.linalg.norm(Fz, axis=1, keepdims=True) + 1e-9)
        r = Ridge(alpha=10.0).fit(Fz[tr], y[tr])
        pf = r.predict(Fz[te])

        for k, p in (("real", pr), ("shuffled", ps), ("frozen", pf)):
            res[k].append(r2(y[te], p)); oof[k][te] = p
        print(f"  fold{fi}  real R2={res['real'][-1]:+.3f}  shuffled R2={res['shuffled'][-1]:+.3f}"
              f"  frozen R2={res['frozen'][-1]:+.3f}   (ep={ep}, n_te={len(te)})")

    print("\n=== 汇总 (leave-station-out, mean over folds) ===")
    for k in ("real", "shuffled", "frozen"):
        v = np.array(res[k])
        print(f"  {k:<9} R2 = {v.mean():+.3f}  ± {v.std():.3f}")
    print("\n=== pooled out-of-fold (全部 221 站) ===")
    for k in ("real", "shuffled", "frozen"):
        p, yy = oof[k], y
        print(f"  {k:<9} R2 = {r2(yy, p):+.3f}   spearman = {spearmanr(p, yy).statistic:+.3f}")
    np.savez(f"{BASE}/sat_gate_{args.city}_c{args.crop}_u{args.unfreeze_last}_s{args.seed}.npz",
             y=y, ids=np.array(ids), **oof)
    print("\n  【判决】real 需同时满足 R2>0 且 R2>frozen，才算端到端挖出了冻结挖不到的站级信号")


if __name__ == "__main__":
    main()
