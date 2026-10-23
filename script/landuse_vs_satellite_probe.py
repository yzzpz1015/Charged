# -*- coding: utf-8 -*-
"""对标 VisionEV 的关键对照：**土地利用指标能否替代卫星图？**

VisionEV 的 7 个 LDU one-hot（Institutional/Transportation/Commercial/Residential/
Recreational/Vacant/Industrial）在对非视觉基线是额外给的，而它的"w/o 图像"消融里没有
任何土地利用信息 —— 所以 1.20 RMSE 测的是「图像作为唯一土地利用通道」的价值。

本脚本在 CHARGED 上直接做这个对照：把 768 维卫星图换成 POI 导出的土地利用特征，
看城内与跨城表现是否同量级。

注意：VisionEV 用的是**地块土地利用标签**；这里只有 OSM POI 类型，是「代理的代理」。
且 `other` 占 LOA 53% / MEL 85% / JHB 87%，OSM 覆盖度跨城不对称 —— 跨城 POI 结果需谨慎。

用法: python3 script/landuse_vs_satellite_probe.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

torch.set_num_threads(8)
BASE = "/home/data/yzzpz/Charged"
CITIES = ["LOA", "MEL", "JHB"]
R_EARTH = 6371000.0
CROP = 192                      # ≈380m，VisionEV 的视野
RADII = [200, 400, 800, 1500]   # POI 半径（米）；400 ≈ 匹配 380m 视野
NOISE = {"other", "bench", "waste_basket", "post_box", "toilets", "drinking_water",
         "telephone", "shelter", "recycling", "waste", "vending_machine",
         "bicycle_parking", "waste_disposal", "clock", "fire_hydrant",
         "give_way", "traffic_signals", "crossing", "street_lamp", "grit_bin",
         "parking_space", "parking_entrance"}
SF = "site_id"

# OSM type -> VisionEV 的 LDU 类别（含 Transport；Residential/Vacant 在 POI 里基本无对应）
LU_MAP = {
    "Transportation": {"parking", "fuel", "bus_station", "taxi", "charging_station",
                       "car_rental", "bicycle_rental", "car_wash", "ferry_terminal",
                       "bus_stop", "railway_station", "motorcycle_parking"},
    "Commercial": {"restaurant", "fast_food", "cafe", "bank", "atm", "bar", "pub",
                   "pharmacy", "marketplace", "ice_cream", "cinema", "theatre",
                   "events_venue", "hotel", "shop", "supermarket", "convenience",
                   "bakery", "butcher", "clothes", "hairdresser", "mall",
                   "department_store", "nightclub", "food_court", "beverages"},
    "Institutional": {"school", "kindergarten", "college", "university", "library",
                      "hospital", "clinic", "doctors", "dentist", "community_centre",
                      "social_facility", "childcare", "place_of_worship", "police",
                      "fire_station", "post_office", "townhall", "courthouse",
                      "government", "embassy", "prison"},
    "Recreational": {"fountain", "bbq", "playground", "park", "sports_centre", "pitch",
                     "swimming_pool", "garden", "picnic_table", "dog_park",
                     "fitness_centre", "stadium", "golf_course", "marina", "attraction",
                     "viewpoint", "museum", "zoo", "theme_park", "artwork"},
    "Industrial": {"industrial", "warehouse", "works", "quarry", "logistics",
                   "factory", "depot", "construction", "landfill"},
    "Residential": {"residential", "apartments", "house", "dormitory", "nursing_home"},
}
LU_CLASSES = ["Transportation", "Commercial", "Institutional",
              "Recreational", "Industrial", "Residential"]
TYPE2LU = {t: c for c, ts in LU_MAP.items() for t in ts}


# ------------------------------------------------------------------ 数据
def load_city(city):
    d = f"{BASE}/data/{city}_remove_const"
    v = pd.read_csv(f"{d}/volume.csv", index_col=0)
    v.index = pd.to_datetime(v.index)
    st = pd.read_csv(f"{d}/sites.csv")
    sc = SF if SF in st.columns else "site"
    allids = st[sc].astype(int).tolist()
    mdir = f"{BASE}/tiles_dataset/{city}/mosaic/z16"
    have = set(int(f.split('.')[0]) for f in os.listdir(mdir) if f.endswith('.png'))
    keep = [i for i, s in enumerate(allids) if s in have]
    ids = [allids[i] for i in keep]
    lat = st["latitude"].values.astype(float)[keep]
    lon = st["longitude"].values.astype(float)[keep]
    chg = st["charger_num"].values.astype(float)[keep]
    wv = v[[str(s) for s in ids]].resample("W").sum()
    wv = wv[wv.sum(1) > 0]
    y = np.log1p(wv.values).mean(0)
    poi = pd.read_csv(f"{d}/poi.csv")
    return dict(ids=ids, lat=lat, lon=lon, chg=chg, y=y, poi=poi, mdir=mdir)


def sat_feat(d, crop=CROP):
    mdir = d["mdir"]
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ev = T.Resize(224)
    b = []
    for s in d["ids"]:
        im = Image.open(f"{mdir}/{s}.0.png").convert("RGB")
        w, h = im.size
        c = min(crop, w, h)
        im = im.crop(((w - c) // 2, (h - c) // 2, (w - c) // 2 + c, (h - c) // 2 + c))
        b.append(norm(ev(T.functional.pil_to_tensor(im).float() / 255.0)))
    m = torch.hub.load('/home/guest3/.cache/torch/hub/facebookresearch_dinov2_main',
                       'dinov2_vitb14', source='local', pretrained=True).cuda().eval()
    out = []
    for i in range(0, len(b), 64):
        with torch.no_grad():
            out.append(m(torch.stack(b[i:i + 64]).cuda()).cpu().numpy())
    F = np.concatenate(out)
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


# --------------------------------------------------- POI 计数（球面 KD 树）
def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def poi_counts(d, radius_m, top_cats=None):
    """返回 [n_station, n_feat]。top_cats=None -> 用 6 类粗分土地利用。"""
    poi = d["poi"]
    t = poi["type"].astype(str).values
    plat = poi["latitude"].values.astype(float)
    plon = poi["longitude"].values.astype(float)
    P = _xyz(plat, plon)
    S = _xyz(d["lat"], d["lon"])
    tree = cKDTree(P)
    # 弦长阈值：角度 theta = r/R，弦 = 2 sin(theta/2)
    thr = 2 * np.sin(radius_m / R_EARTH / 2)
    if top_cats is None:
        lab = np.array([TYPE2LU.get(x, "Residential" if x in LU_MAP["Residential"] else "__drop__")
                        for x in t])
        cats = LU_CLASSES
        idx = {c: np.where(lab == c)[0] for c in cats}
    else:
        cats = list(top_cats)
        idx = {c: np.where(t == c)[0] for c in cats}
    out = np.zeros((len(S), len(cats)), float)
    for j, s in enumerate(S):
        nb = tree.query_ball_point(s, thr)
        nb = np.asarray(nb, dtype=int)
        if len(nb) == 0:
            continue
        for k, c in enumerate(cats):
            ii = idx[c]
            if len(ii):
                out[j, k] = np.intersect1d(nb, ii, assume_unique=False).size
    return np.log1p(out), cats


def top_cats_of(d, k=15):
    """跨城可比的通用类别：取三城合并后的高频非噪声类别"""
    return None  # 由 main 统一计算


# ------------------------------------------------------------------ 评估
def per_fold_rho(X, y, nseed=20, alpha=10.0):
    """不逐列标准化！L2 归一化后的 DINOv2 嵌入列方差极小，除以它会把噪声放大百倍，
    alpha=10 随之失效 → 灾难性过拟合（实测 R2 掉到 −1.5）。改用 sklearn Ridge 的
    内部中心化（fit_intercept），与已验证协议一致。"""
    n = len(y)
    rs, r2, rm = [], [], []
    for s in range(nseed):
        p = np.zeros(n)
        for f in np.array_split(np.random.RandomState(s).permutation(n), 5):
            tr = np.setdiff1d(np.arange(n), f)
            p[f] = Ridge(alpha=alpha).fit(X[tr], y[tr]).predict(X[f])
        rs.append(spearmanr(p, y).statistic)
        ss = ((y - y.mean()) ** 2).sum()
        r2.append(1 - ((y - p) ** 2).sum() / ss)
        rm.append(np.sqrt(((y - p) ** 2).mean()))
    return np.mean(rs), np.mean(r2), np.mean(rm)


def main():
    D, SAT = {}, {}
    for c in CITIES:
        D[c] = load_city(c)
        SAT[c] = sat_feat(D[c])
        print(f"[load] {c}: n={len(D[c]['y'])}  sat={SAT[c].shape}", flush=True)

    # 跨城共用的 top-k 类别（三城合并频次，去噪）
    from collections import Counter
    cnt = Counter()
    for c in CITIES:
        vc = D[c]["poi"]["type"].astype(str).value_counts()
        for k, v in vc.items():
            if k not in NOISE:
                cnt[k] += v
    TOP = [k for k, _ in cnt.most_common(15)]
    print(f"[POI] 跨城通用 top-15 类别: {TOP}\n", flush=True)

    for c in CITIES:
        poi = D[c]["poi"]
        n_other = (poi["type"].astype(str) == "other").sum()
        print(f"### {c}  n={len(D[c]['y'])}   POI 'other' 占比 = {n_other/len(poi):.1%}", flush=True)
        _, _, rm0 = per_fold_rho(D[c]["chg"].reshape(-1, 1), D[c]["y"])
        print(f"  {'特征':<30}{'rho':>9}{'R2':>9}{'RMSE':>9}", flush=True)
        print("  " + "-" * 57)
        r, R2, rm = per_fold_rho(D[c]["chg"].reshape(-1, 1), D[c]["y"])
        print(f"  {'仅 charger_num (1维)':<30}{r:>+9.3f}{R2:>+9.3f}{rm:>9.3f}", flush=True)
        r, R2, rm = per_fold_rho(SAT[c], D[c]["y"])
        print(f"  {'仅 卫星图 380m (768维)':<30}{r:>+9.3f}{R2:>+9.3f}{rm:>9.3f}", flush=True)
        for rad in RADII:
            X7, _ = poi_counts(D[c], rad)
            r, R2, rm = per_fold_rho(X7, D[c]["y"])
            print(f"  {f'仅 土地利用6类 @{rad}m':<30}{r:>+9.3f}{R2:>+9.3f}{rm:>9.3f}", flush=True)
        for rad in RADII:
            XT, _ = poi_counts(D[c], rad, top_cats=TOP)
            r, R2, rm = per_fold_rho(XT, D[c]["y"])
            print(f"  {f'仅 POI top15 @{rad}m':<30}{r:>+9.3f}{R2:>+9.3f}{rm:>9.3f}", flush=True)
        X4, _ = poi_counts(D[c], 400)
        r, R2, rm = per_fold_rho(np.concatenate([SAT[c], X4], 1), D[c]["y"])
        print(f"  {'卫星图 + 土地利用@400m':<30}{r:>+9.3f}{R2:>+9.3f}{rm:>9.3f}", flush=True)
        print(flush=True)

    # ---------------- 跨城 ----------------
    print("=== 跨城 LOA -> MEL∪JHB  (目标逐城 z 分数后拼接) ===", flush=True)
    print(f"  {'特征':<30}{'rho':>9}{'p':>9}", flush=True)
    print("  " + "-" * 48)

    def zz(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    ytar = np.concatenate([zz(D["MEL"]["y"]), zz(D["JHB"]["y"])])
    sets = {}
    sets["卫星图 380m"] = (SAT["LOA"],
                          np.concatenate([SAT["MEL"], SAT["JHB"]]))
    for rad in [400, 1500]:
        Xl, _ = poi_counts(D["LOA"], rad)
        Xt, _ = poi_counts(D["MEL"], rad); Xt2, _ = poi_counts(D["JHB"], rad)
        sets[f"土地利用6类 @{rad}m"] = (Xl, np.concatenate([Xt, Xt2]))
    for rad in [400, 1500]:
        Xl, _ = poi_counts(D["LOA"], rad, top_cats=TOP)
        Xt, _ = poi_counts(D["MEL"], rad, top_cats=TOP); Xt2, _ = poi_counts(D["JHB"], rad, top_cats=TOP)
        sets[f"POI top15 @{rad}m"] = (Xl, np.concatenate([Xt, Xt2]))
    sets["charger_num"] = (D["LOA"]["chg"].reshape(-1, 1),
                           np.concatenate([D["MEL"]["chg"], D["JHB"]["chg"]]).reshape(-1, 1))

    for nm, (Xl, Xt) in sets.items():
        yl = zz(D["LOA"]["y"])
        p = Ridge(alpha=10.0).fit(Xl, yl).predict(Xt)
        o = spearmanr(p, ytar).statistic
        rng = np.random.RandomState(0)
        nl = np.array([abs(spearmanr(Ridge(alpha=10.0).fit(Xl, rng.permutation(yl)).predict(Xt),
                                     ytar).statistic) for _ in range(3000)])
        print(f"  {nm:<30}{o:>+9.3f}{(nl >= abs(o)).mean():>9.4f}", flush=True)


if __name__ == "__main__":
    main()
