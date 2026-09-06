# -*- coding: utf-8 -*-
"""RemoteCLIP (geo-pretrained) discrimination probe.

Contrasts with DINOv2: is the near-degenerate cross-site cosine a DINOv2
(ImageNet) artifact, or do the Melbourne sites genuinely look alike?  Extract
RemoteCLIP ViT-L-14 image features (CLS + patch tokens) for the same 62
mosaics and report the SAME cosine-spread + load-level LOO R^2 metrics.
"""
import os
import numpy as np
import torch
from PIL import Image, ImageFile
from huggingface_hub import hf_hub_download
from sklearn.linear_model import Ridge

ImageFile.LOAD_TRUNCATED_IMAGES = True
BASE = "/home/data/yzzpz/Charged"
CITY = "MEL"
MOSAIC = f"{BASE}/tiles_dataset/{CITY}/mosaic/z16"
MODEL = "ViT-L-14"


def load_model():
    import open_clip
    local = f"{BASE}/api/model/weights/remoteclip/models--chendelong--RemoteCLIP/" \
            f"snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-{MODEL}.pt"
    if os.path.exists(local) and os.path.getsize(local) > 1.5e9:
        ckpt = local
    else:
        ckpt = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{MODEL}.pt",
                               cache_dir=f"{BASE}/api/model/weights/remoteclip")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model = model.cuda().eval()
    return model, preprocess


def extract(model, preprocess, files, device):
    model.visual.output_tokens = True
    cls, patches = [], []
    with torch.no_grad():
        for f in files:
            img = Image.open(f"{MOSAIC}/{f}").convert('RGB')
            x = preprocess(img).unsqueeze(0).to(device)   # [1,3,224,224]
            pooled, tokens = model.visual(x)              # pooled [1,768] CLS, tokens [1,196,1024] patches
            cls.append(pooled[0].cpu().numpy())
            patches.append(tokens[0].cpu().numpy())       # [N_patch, 1024]
    return np.stack(cls), np.stack(patches)


def grid_mean(patches, g):
    N, P, D = patches.shape
    side = int(round(P ** 0.5))
    p = patches.reshape(N, side, side, D)
    s = side - side % g
    p = p[:, :s, :s, :]
    h = s // g
    return p.reshape(N, g, h, g, h, D).mean(axis=(2, 4)).reshape(N, g * g, D)


def cosine_spread(reps):
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
    files = sorted([f for f in os.listdir(MOSAIC) if f.endswith('.png')],
                   key=lambda x: int(x.split('.')[0]))
    print(f"[load] RemoteCLIP {MODEL} on {device}, {len(files)} mosaics")
    model, preprocess = load_model()
    cls, patches = extract(model, preprocess, files, device)
    print(f"[tok] CLS {cls.shape}  patches {patches.shape}")

    import pandas as pd
    vol = pd.read_csv(f"{BASE}/data/{CITY}_remove_zero/volume.csv", index_col=0).astype(float)
    dur = pd.read_csv(f"{BASE}/data/{CITY}_remove_zero/duration.csv", index_col=0).astype(float)
    sites = pd.read_csv(f"{BASE}/data/{CITY}_remove_zero/sites.csv")
    site_ids = [int(f.split('.')[0]) for f in files]
    assert vol.shape[1] == len(files) == dur.shape[1], "site count mismatch"
    mu_vol = vol.mean(0).values
    mu_dur = dur.mean(0).values
    power = sites.set_index('site').loc[site_ids, 'avg_power'].values

    reps = {
        "CLS": cls,
        "patch-mean": patches.mean(1),
        "grid7x7": grid_mean(patches, 7).reshape(patches.shape[0], -1),
    }
    labels = {
        "vol(~power)": mu_vol,
        "duration": mu_dur,
        "log duration": np.log1p(mu_dur),
        "vol/power": mu_vol / power,
    }

    print(f"\n{'repr':<12}{'cos.mean':>10}{'cos.min':>9}" + "".join(f"{k:>14}" for k in labels))
    for name, R in reps.items():
        cm, _, cx = cosine_spread(l2(R))
        row = f"{name:<12}{cm:>10.4f}{cx:>9.4f}"
        for k, y in labels.items():
            row += f"{loo_ridge(R, y):>14.3f}"
        print(row)

    np.save(f"{BASE}/tiles_dataset/{CITY}/embedding/z16/remoteclip_cls.npy", cls)
    np.save(f"{BASE}/tiles_dataset/{CITY}/embedding/z16/remoteclip_grid7.npy", reps["grid7x7"])
    print(f"\n[saved] remoteclip_cls.npy {cls.shape}  remoteclip_grid7.npy {reps['grid7x7'].shape}")


if __name__ == "__main__":
    main()
