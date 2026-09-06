# -*- coding: utf-8 -*-
"""Patch-token discrimination check.

Question: is the near-degenerate cross-site cosine (~0.87) a CLS-pooling
artifact, or does the DINOv2 feature space genuinely fail to distinguish these
62 Melbourne charging sites?

Compare cross-site pairwise cosine of several spatial representations:
  CLS (baseline), patch-mean, patch-max, and grid-mean-pooled patch tokens
  (7x7 / 5x5 / 3x3 regions).  Lower cosine = more discriminative.  Then, for
  each, a leave-one-out ridge R^2 against per-site mean load level (busy-ness):
  if a patch representation turns the counterfactual positive, the signal was
  in the spatial layout and CLS destroyed it.
"""
import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from sklearn.linear_model import Ridge

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE = "/home/data/yzzpz/Charged"
CITY = "MEL"
MOSAIC = f"{BASE}/tiles_dataset/{CITY}/mosaic/z16"


def load_model(device):
    m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device).eval()
    return m


def get_tokens(model, files, device):
    tfm = transforms.Compose([
        transforms.Resize(518), transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cls, patches = [], []
    with torch.no_grad():
        for f in files:
            img = Image.open(f"{MOSAIC}/{f}").convert('RGB')
            x = tfm(img).unsqueeze(0).to(device)
            out = model.forward_features(x)          # dict
            cls.append(out['x_norm_clstoken'][0].cpu().numpy())          # [768]
            patches.append(out['x_norm_patchtokens'][0].cpu().numpy())   # [1369, 768]
    return np.stack(cls), np.stack(patches)          # [N,768], [N,1369,768]


def grid_mean(patches, g):
    """Mean-pool 37x37 patch grid to g x g regions -> [N, g*g, 768]."""
    N, P, D = patches.shape
    side = int(round(P ** 0.5))                      # 37
    p = patches.reshape(N, side, side, D)
    # crop to multiple of g
    s = side - side % g
    p = p[:, :s, :s, :]
    h = s // g
    return p.reshape(N, g, h, g, h, D).mean(axis=(2, 4)).reshape(N, g * g, D)


def cosine_spread(reps):
    """reps: [N, D] (already L2-normalized rows). mean/median/min pairwise cosine."""
    C = reps @ reps.T
    np.fill_diagonal(C, np.nan)
    off = C[np.triu_indices(reps.shape[0], 1)]
    return np.nanmean(off), np.nanmedian(off), np.nanmin(off)


def loo_ridge(X, y, alpha=10.0):
    X = np.asarray(X, float); y = np.asarray(y, float)
    N = len(y); r2s = []
    for i in range(N):
        tr = np.arange(N) != i
        m = X[tr].mean(0, keepdims=True); s = X[tr].std(0, keepdims=True); s[s < 1e-9] = 1.0
        r = Ridge(alpha=alpha).fit((X[tr] - m) / s, y[tr])
        r2s.append((y[i] - r.predict((X[i:i + 1] - m) / s)[0]) ** 2)
    ss = float(((y - y.mean()) ** 2).sum())
    return 1 - float(np.sum(r2s)) / ss if ss > 0 else np.nan


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    files = sorted([f for f in __import__('os').listdir(MOSAIC) if f.endswith('.png')],
                   key=lambda x: int(x.split('.')[0]))
    print(f"[load] DINOv2 vitb14 on {device}, {len(files)} mosaics")
    model = load_model(device)
    cls, patches = get_tokens(model, files, device)
    print(f"[tok] CLS {cls.shape}  patches {patches.shape}")

    # load level
    import pandas as pd
    vol = pd.read_csv(f"{BASE}/data/{CITY}_remove_zero/volume.csv", index_col=0).astype(float)
    mu = vol.mean(0).values                       # per-site busy-ness (raw)

    # representations
    reps = {
        "CLS": cls,
        "patch-mean": patches.mean(1),
        "patch-max": patches.max(1),
        "patch-mean+std": np.concatenate([patches.mean(1), patches.std(1)], 1),
    }
    for g in [7, 5, 3]:
        reps[f"grid{g}x{g}"] = grid_mean(patches, g).reshape(patches.shape[0], -1)

    print(f"\n{'repr':<16}{'cos.mean':>10}{'cos.med':>9}{'cos.min':>9}{'LOO R2(mu)':>12}")
    for name, R in reps.items():
        Rl = l2(R)
        cm, cn, cx = cosine_spread(Rl)
        r2 = loo_ridge(R, mu)
        print(f"{name:<16}{cm:>10.4f}{cn:>9.4f}{cx:>9.4f}{r2:>12.3f}")


if __name__ == "__main__":
    main()
