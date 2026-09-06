# -*- coding: utf-8 -*-
"""RemoteCLIP (geo-pretrained, discriminative patch) probe across cities.

The cross-city DINOv2 probe showed station SIZE (area/charger_num) drives
busy-ness (charger_num->duration r 0.35-0.87 in JHB/LOA) but DINOv2 CLS is too
degenerate (cos 0.85) to see size (R2(charger#) < 0).  Here we extract
RemoteCLIP ViT-L-14 patch grid (cos 0.67 in MEL) for LOA (charger_num 1..16,
the diversity test case) and JHB, and ask the decisive question:

    does a discriminative geo-encoder recover {charger_num, area, duration}?

If grid7x7 -> charger_num / area turns positive where DINOv2 CLS was ~0, then
the satellite signal EXISTS and the bottleneck is the DINOv2 CLS encoder, not
the data.  (No downloads: weights + mosaics already local.)
"""
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

ImageFile.LOAD_TRUNCATED_IMAGES = True
BASE = "/home/data/yzzpz/Charged"
CITIES = ["LOA", "JHB"]          # LOA = key diversity case (charger_num 1..16)
MODEL = "ViT-L-14"
PCA_DIM = 15
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def load_model():
    import open_clip
    local = f"{BASE}/api/model/weights/remoteclip/models--chendelong--RemoteCLIP/" \
            f"snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-{MODEL}.pt"
    assert os.path.exists(local), "RemoteCLIP weights missing"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    model.load_state_dict(torch.load(local, map_location="cpu", weights_only=True))
    return model.cuda().eval(), preprocess


def extract(model, preprocess, mosaic, files, device):
    model.visual.output_tokens = True
    cls, patches = [], []
    with torch.no_grad():
        for f in files:
            img = Image.open(f"{mosaic}/{f}").convert('RGB')
            x = preprocess(img).unsqueeze(0).to(device)
            pooled, tokens = model.visual(x)   # pooled [1,768], tokens [1,196,1024]
            cls.append(pooled[0].cpu().numpy())
            patches.append(tokens[0].cpu().numpy())
    return np.stack(cls), np.stack(patches)


def grid_mean(patches, g=7):
    N, P, D = patches.shape
    side = int(round(P ** 0.5))
    p = patches.reshape(N, side, side, D)
    s = side - side % g
    p = p[:, :s, :s, :]
    h = s // g
    return p.reshape(N, g, h, g, h, D).mean(axis=(2, 4)).reshape(N, g * g, D)


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def cosine_spread(reps):
    C = reps @ reps.T
    np.fill_diagonal(C, np.nan)
    off = C[np.triu_indices(reps.shape[0], 1)]
    return np.nanmean(off)


def pca(X, n):
    return X if X.shape[1] <= n else PCA(n_components=n).fit_transform(X)


def loo_ridge(X, y, alphas=ALPHAS):
    X = np.asarray(X, float); y = np.asarray(y, float)
    N = len(y); best_r2 = -1e18
    for a in alphas:
        r2s = []
        for i in range(N):
            tr = np.arange(N) != i
            m = X[tr].mean(0, keepdims=True); s = X[tr].std(0, keepdims=True); s[s < 1e-9] = 1.0
            r = Ridge(alpha=a).fit((X[tr] - m) / s, y[tr])
            r2s.append((y[i] - r.predict((X[i:i + 1] - m) / s)[0]) ** 2)
        ss = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float(np.sum(r2s)) / ss if ss > 0 else np.nan
        if r2 > best_r2:
            best_r2 = r2
    return best_r2


def labels_for(city, fids):
    data = f"{BASE}/data/{city}_remove_zero"
    dur = pd.read_csv(f"{data}/duration.csv", index_col=0)
    vol = pd.read_csv(f"{data}/volume.csv", index_col=0)
    sites = pd.read_csv(f"{data}/sites.csv")
    col = 'site_id' if 'site_id' in sites.columns else 'site'
    s = sites.set_index(col)
    mu_dur = dur[[str(f) for f in fids]].mean(0).values
    mu_vol = vol[[str(f) for f in fids]].mean(0).values
    charger = s.loc[fids, 'charger_num'].astype(float).values
    area = s.loc[fids, 'area'].astype(float).values
    return mu_dur, mu_vol, charger, area


def main():
    device = "cuda:0"
    model, preprocess = load_model()
    for city in CITIES:
        mosaic = f"{BASE}/tiles_dataset/{city}/mosaic/z16"
        files = sorted([f for f in os.listdir(mosaic) if f.endswith('.png')],
                       key=lambda x: int(x.split('.')[0]))
        fids = [int(f.split('.')[0]) for f in files]
        print(f"\n== {city}: {len(fids)} mosaics, extracting RemoteCLIP ...")
        cls, patches = extract(model, preprocess, mosaic, files, device)
        g7 = grid_mean(patches, 7).reshape(len(fids), -1)
        np.save(f"{BASE}/tiles_dataset/{city}/embedding/z16/remoteclip_cls.npy", cls)
        np.save(f"{BASE}/tiles_dataset/{city}/embedding/z16/remoteclip_grid7.npy", g7)

        mu_dur, mu_vol, charger, area = labels_for(city, fids)
        reps = {"CLS": l2(cls), "grid7x7": l2(g7)}
        print(f"  cosine: CLS {cosine_spread(l2(cls)):.3f}  grid7x7 {cosine_spread(l2(g7)):.3f}")
        print(f"  {'repr':<9}{'R2(chg#)':>10}{'R2(area)':>10}{'R2(dur)':>10}{'R2(vol)':>10}{'R2(dur/chg)':>12}")
        for name, R in reps.items():
            X = pca(R, PCA_DIM)
            r_chg = loo_ridge(X, np.log1p(charger))
            r_area = loo_ridge(X, np.log1p(area))
            r_dur = loo_ridge(X, mu_dur)
            r_vol = loo_ridge(X, mu_vol)
            r_durc = loo_ridge(X, mu_dur / charger)
            print(f"  {name:<9}{r_chg:>10.3f}{r_area:>10.3f}{r_dur:>10.3f}{r_vol:>10.3f}{r_durc:>12.3f}")


if __name__ == "__main__":
    main()
