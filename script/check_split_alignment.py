# -*- coding: utf-8 -*-
"""核对模型侧划分是否与基线 script/prob_baseline.py 完全一致（只读，不训练）。

用这个脚本当"改完划分之后的回归测试"：

  1. 打印每城三段的起止日期 + 行数，与 result/prob_baseline/report.txt 的划分表对照；
  2. 对 seq_l=72 / pre_len ∈ {24,48,72} 枚举窗口，断言三条不变量：
       - 任一窗口整段无停摆行；
       - 任一窗口的目标块整块落在它声明的 split 内；
       - 三个 split 的窗口数都 > 0。
  3. 跨城检查：五个干净城（LOA/MEL/JHB/SPO/SZH）的日期必须彼此一致。

用法：python3 script/check_split_alignment.py [suffix]
      suffix 默认 _remove_const（与基线一致）。
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.dataset.common import EVDataset, SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST
from api.utils import split_window_starts

# 基线以"脚本"方式导入（prob_baseline.py 不在包内），共用划分/剔站定义。
_spec = importlib.util.spec_from_file_location(
    "prob_baseline", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prob_baseline.py"))
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)

CITIES = ["LOA", "MEL", "JHB", "AMS", "SPO", "SZH"]
SEQ_L = 72
PRE_LENS = [24, 48, 72]
NAMES = {SPLIT_TRAIN: "train", SPLIT_VALID: "valid", SPLIT_TEST: "test"}

# 基线 report.txt 里的期望边界（停摆丢弃后在剩余可用时间上按计数切 2/1/3）+
# 剔站后的站数（训练期退化站已剔，见 prob_baseline.live_cols）
EXPECT = {
    "LOA": ("04-01~05-31", "06-01~07-01", "07-01~09-30", 0, 218),
    "MEL": ("04-01~05-31", "05-31~07-01", "07-01~09-30", 19, 56),
    "JHB": ("04-01~05-31", "06-01~07-01", "07-01~09-30", 0, 29),
    "AMS": ("05-07~06-25", "06-26~07-20", "07-20~09-30", 901, 1383),
    "SPO": ("04-01~05-31", "06-01~07-01", "07-01~09-30", 0, 33),
    "SZH": ("04-01~05-31", "06-01~07-01", "07-01~09-30", 0, 843),
}


def check_city(city, suffix):
    ds = EVDataset(feature='volume', auxiliary='None',
                   data_path=f"data/{city}{suffix}/")
    ds.split_cross_validation(fold=6, total_fold=6, train_ratio=0.8, valid_ratio=0.1)
    t = ds.time

    def rng(a, b):                                  # [start, end) -> "mm-dd~mm-dd"
        return f"{t[a].strftime('%m-%d')}~{t[b - 1].strftime('%m-%d')}"

    got = (rng(ds.train_start, ds.train_end),
           rng(ds.valid_start, ds.valid_end),
           rng(ds.test_start, ds.test_end))
    dropped = int(ds.outage_row.sum())

    ok = True
    exp = EXPECT[city]
    if got != exp[:3]:
        print(f"  ✗ {city} 日期不符：期望 {exp[:3]}，实得 {got}")
        ok = False
    if dropped != exp[3]:
        print(f"  ✗ {city} 停摆行数不符：期望 {exp[3]}，实得 {dropped}")
        ok = False
    if ds.feat.shape[1] != exp[4]:
        print(f"  ✗ {city} 剔站后站数不符：期望 {exp[4]}，实得 {ds.feat.shape[1]}")
        ok = False

    # ---- 与基线逐列比对站集（两边各有一份阈值常量，漂了就会在这里炸）
    # 基线的 load() 写死 _remove_const，所以只在同后缀时比对。
    if suffix == "_remove_const":
        y0, t0 = pb.load(city)
        _ok, trm, _vam, _tem, _u, _span = pb.split_rows(y0, t0)
        base_cols, _smean = pb.live_cols(y0, trm)
        _vol_cols = list(pd.read_csv(f"data/{city}{suffix}/volume.csv", index_col=0).columns)
        base_ids = [_vol_cols[i] for i in base_cols]
        if [str(c) for c in ds.feat_columns] != [str(c) for c in base_ids]:
            print(f"  ✗ {city} 与基线的站集/站序不一致"
                  f"（模型 {ds.feat.shape[1]} 列 vs 基线 {len(base_ids)} 列）")
            ok = False

    # ---- 窗口不变量
    line = []
    for pre in PRE_LENS:
        counts = []
        for sid in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST):
            st = split_window_starts(ds.usable, ds.split_of_row, sid, SEQ_L, pre)
            span = SEQ_L + pre
            if len(st):
                # 整段可用
                bad = [i for i in st if not ds.usable[i:i + span].all()]
                assert not bad, f"{city} pre={pre} {NAMES[sid]}: {len(bad)} 个窗口跨停摆缺口"
                # 目标块整块属于本 split
                for i in st:
                    tb = ds.split_of_row[i + SEQ_L: i + span]
                    assert (tb == sid).all(), f"{city} pre={pre} {NAMES[sid]}: 目标块越界"
            else:
                print(f"  ✗ {city} pre={pre} {NAMES[sid]} 窗口数为 0")
                ok = False
            counts.append(len(st))
        line.append(f"pre{pre}: " + "/".join(str(c) for c in counts))

    print(f"{city:<5}站{ds.feat.shape[1]:>5}  丢{dropped:>4}  "
          f"tr {got[0]}  va {got[1]}  te {got[2]}   " + "  ".join(line))
    return ok


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else "_remove_const"
    print(f"data suffix = {suffix}   seq_l = {SEQ_L}")
    print("=" * 118)
    allok = True
    for c in CITIES:
        try:
            allok &= check_city(c, suffix)
        except Exception as e:                       # noqa: BLE001
            print(f"  ✗ {c} 失败：{type(e).__name__}: {e}")
            allok = False
    print("=" * 118)
    print("窗口数 = train/valid/test（模型口径：整段 72h 历史 + 目标块都要可用）")
    print("注意：它与基线 report.txt 的『起点』列**本来就不相等**——基线是单锚点法，"
          "只要锚点那一行可用；模型要整段可用。只有**日期**必须对齐。")
    print("\n" + ("全部通过 ✓" if allok else "有不符项 ✗"))


if __name__ == "__main__":
    main()
