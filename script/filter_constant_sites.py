#!/usr/bin/env python
"""筛出并剔除「常量」站点，生成 {CITY}_remove_const/ 干净版本。

背景：部分站点的 volume 序列整段是同一个值，没有真实逐小时波动，分两类：
  - 恒非 0（fill）：= total_volume / 8760 的年度化平铺伪影，序列被常量填充；
  - 恒 0（zero）：全时段闲置站，_remove_zero 因 duration/price 非 0 漏网的残留。
本脚本在 _remove_zero 基础上额外剔除这两类「恒值」站，产出 _remove_const，
并写出 data/constant_sites.csv 排除清单（含 kind 字段区分 zero/fill）。

判据：volume.csv 中「唯一取值数 == 1」（无论该值是 0 还是非 0）的站点。
"""
import os
import shutil
import numpy as np
import pandas as pd

CITIES = ["AMS", "JHB", "LOA", "MEL", "SPO", "SZH"]
SRC_SUFFIX = "_remove_zero"
DST_SUFFIX = "_remove_const"
BASE = "data"


def site_col(df: pd.DataFrame) -> str:
    return "site_id" if "site_id" in df.columns else "site"


def main():
    records = []

    for city in CITIES:
        src = f"{BASE}/{city}{SRC_SUFFIX}"
        dst = f"{BASE}/{city}{DST_SUFFIX}"
        vol = pd.read_csv(f"{src}/volume.csv", index_col=0)

        # 1) 识别常量站点（恒 0 与恒非 0 都剔）
        exclude = set()
        for col in vol.columns:
            u = np.unique(vol[col].astype(float).values)
            if len(u) == 1:
                exclude.add(str(col))

        sites = pd.read_csv(f"{src}/sites.csv")
        sc = site_col(sites)
        sites["_sid"] = sites[sc].astype(str)
        for sid in sorted(exclude, key=lambda x: int(x) if x.isdigit() else 0):
            row = sites[sites["_sid"] == sid]
            tv = float(row["total_volume"].iloc[0]) if len(row) else np.nan
            cv = float(vol[sid].iloc[0]) if sid in vol.columns else np.nan
            records.append({"city": city, "site_id": sid, "kind": "zero" if cv == 0 else "fill",
                            "constant_value": cv, "total_volume": tv})

        # 2) 建目标目录，逐文件过滤
        os.makedirs(dst, exist_ok=True)

        # 逐站点列文件：去掉 exclude 列
        for f in ["volume.csv", "duration.csv", "e_price.csv", "s_price.csv"]:
            df = pd.read_csv(f"{src}/{f}", index_col=0)
            keep = [c for c in df.columns if str(c) not in exclude]
            df[keep].to_csv(f"{dst}/{f}")

        # sites.csv / chargers.csv / site_poi_1500m.csv：去掉 exclude 行
        for f in ["sites.csv", "chargers.csv"]:
            df = pd.read_csv(f"{src}/{f}")
            c = site_col(df)
            df = df[~df[c].astype(str).isin(exclude)]
            df.to_csv(f"{dst}/{f}", index=False)

        if os.path.exists(f"{src}/site_poi_1500m.csv"):
            df = pd.read_csv(f"{src}/site_poi_1500m.csv")
            c = site_col(df)
            df = df[~df[c].astype(str).isin(exclude)]
            df.to_csv(f"{dst}/site_poi_1500m.csv", index=False)

        # distance.csv：对称矩阵，去掉 exclude 行 + 列
        dist = pd.read_csv(f"{src}/distance.csv", index_col=0)
        dist.index = dist.index.astype(str)
        dist.columns = dist.columns.astype(str)
        keep_idx = [c for c in dist.index if c not in exclude]
        dist = dist.loc[keep_idx, keep_idx]
        dist.to_csv(f"{dst}/distance.csv")

        # poi.csv / weather.csv：城市级，原样拷贝
        for f in ["poi.csv", "weather.csv"]:
            if os.path.exists(f"{src}/{f}"):
                shutil.copy2(f"{src}/{f}", f"{dst}/{f}")

        # info.csv：更新 total_sites / total_chargers
        info = pd.read_csv(f"{src}/info.csv")
        keep_sites = sites[~sites["_sid"].isin(exclude)]
        info["total_sites"] = len(keep_sites)
        info["total_chargers"] = int(keep_sites["charger_num"].sum())
        info.to_csv(f"{dst}/info.csv", index=False)

        print(f"{city}: 剔除 {len(exclude)} 个常量站，"
              f"剩余 {len(vol.columns) - len(exclude)} 站")

    # 3) 写出排除清单
    out = pd.DataFrame(records)
    out_path = f"{BASE}/constant_sites.csv"
    out.to_csv(out_path, index=False)
    print(f"\n排除清单已写: {out_path} (共 {len(out)} 站)")


if __name__ == "__main__":
    main()
