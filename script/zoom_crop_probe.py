# -*- coding: utf-8 -*-
"""Zoom/scale hypothesis probe (free — no download).

Station footprints are 66-170m but the z16 mosaic is 1536px = ~3.7km, so a
station is <0.2% of the tile — a needle in a haystack, invisible to any
encoder.  The tile is CENTERED on the site lon/lat (fixed_square_window), so we
can crop the center of the EXISTING z16 mosaic to simulate a finer zoom, and
ask: does a tighter crop recover the {charger_num, area, duration} signal?

If R2 rises from ~0 (full 1536px) to positive as the crop tightens, the
bottleneck is scale — and re-downloading at z18/z19 is the fix, not a better
encoder.
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
CITY = "LOA"                     # diversity case: charger_num 1..16, area up to 67k m2
MODEL = "ViT-L-14"
PCA_DIM = 15
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def load_model():
    import open_clip
    local = f"{BASE}/api/model/weights/remoteclip/models--chendelong--RemoteCLIP/" \
            f"snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-{MODEL}.pt"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    model.load_state_dict(torch.load(local, map_location="cpu", weights_only=True))
    return model.cuda().eval(), preprocess


def center_crop(img, size):
    w, h = img.size
    l = (w - size) // 2
    t = (h - size) // 2
    return img.crop((l, t, l + size, t + size))


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


def labels(fids):
    data = f"{BASE}/data/{CITY}_remove_zero"
    dur = pd.read_csv(f"{data}/duration.csv", index_col=0)
    vol = pd.read_csv(f"{data}/volume.csv", index_col=0)
    s = pd.read_csv(f"{data}/sites.csv").set_index('site_id')
    mu_dur = dur[[str(f) for f in fids]].mean(0).values
    mu_vol = vol[[str(f) for f in fids]].mean(0).values
    charger = s.loc[fids, 'charger_num'].astype(float).values
    area = s.loc[fids, 'area'].astype(float).values
    return mu_dur, mu_vol, charger, area


def main():
    mosaic = f"{BASE}/tiles_dataset/{CITY}/mosaic/z16"
    files = sorted([f for f in os.listdir(mosaic) if f.endswith('.png')],
                   key=lambda x: int(x.split('.')[0]))
    fids = [int(f.split('.')[0]) for f in files]
    mu_dur, mu_vol, charger, area = labels(fids)

    model, preprocess = load_model()
    model.visual.output_tokens = True

    # crop size -> extracted grid7x7 features
    crops = [1536, 768, 512, 384, 256]
    feat = {}
    for c in crops:
        patches = []
        with torch.no_grad():
            for f in files:
                img = Image.open(f"{mosaic}/{f}").convert('RGB')
                if c < 1536:
                    img = center_crop(img, c)
                x = preprocess(img).unsqueeze(0).cuda()
                _, tok = model.visual(x)     # [1,196,1024]
                patches.append(tok[0].cpu().numpy())
        g7 = grid_mean(np.stack(patches), 7).reshape(len(fids), -1)
        feat[c] = l2(g7)

    print(f"== {CITY}: {len(fids)} sites ==  (charger_num {int(charger.min())}-{int(charger.max())}, "
          f"area {area.min():.0f}-{area.max():.0f} m2)")
    print(f"  crop(px)  ->km    cos(g7)   R2(chg#)  R2(area)  R2(dur)   R2(vol)")
    for c in crops:
        km = c * 2.39 / 1000.0
        X = pca(feat[c], PCA_DIM)
        cos = float(np.nanmean(
            (lambda C: (np.fill_diagonal(C, np.nan), C)[1])(feat[c] @ feat[c].T)
            [np.triu_indices(len(fids), 1)]))
        r_chg = loo_ridge(X, np.log1p(charger))
        r_area = loo_ridge(X, np.log1p(area))
        r_dur = loo_ridge(X, mu_dur)
        r_vol = loo_ridge(X, mu_vol)
        print(f"  {c:>6}    {km:>5.2f}   {cos:>7.3f}   {r_chg:>8.3f} {r_area:>8.3f} {r_dur:>8.3f} {r_vol:>8.3f}")


if __name__ == "__main__":
    main()
