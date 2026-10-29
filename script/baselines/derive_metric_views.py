# -*- coding: utf-8 -*-
"""从 ``metrics/`` 主 JSON 派生 ``metrics_point/`` 与 ``metrics_wis/`` 视图。

新的训练脚本一次评测就会把 CRPS、WIS、点预测指标全部写进主 JSON；旧批次因为
指标是分批补的，才需要 ``BASELINE_TAG=point/wis`` 重评。重跑之后用本脚本按
历史字段集合切视图，避免再采样一遍（对 DiffSTG 尤其重要）。

用法::

    python3 script/baselines/derive_metric_views.py --methods stgcn stgcn_nograph gwnet diffstg tsmixer
"""
import argparse
import glob
import json
import os

POINT_KEYS = {"mae", "rmse", "r2", "n_point_cells"}
WIS_KEYS = {
    "wis", "wis_nz", "wis_zero", "crps_zero_contrib", "crps_pos_contrib",
    "crps_zero_mean", "crps_pos_mean", "zero_cell_frac", "n_zero_cells",
    "n_pos_cells",
}

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "result", "baselines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    n_point = n_wis = 0
    for method in args.methods:
        for src in sorted(glob.glob(os.path.join(ROOT, "metrics", method,
                                                 "*", "H*_seed*.json"))):
            with open(src) as f:
                full = json.load(f)
            rel = os.path.relpath(src, os.path.join(ROOT, "metrics"))
            for view, extra in (("metrics_point", POINT_KEYS),
                                ("metrics_wis", POINT_KEYS | WIS_KEYS)):
                dst = os.path.join(ROOT, view, rel)
                if os.path.exists(dst) and not args.overwrite:
                    raise SystemExit(f"已存在，拒绝覆盖：{dst}（加 --overwrite）")
                # 各方法的超参字段不同（如 DiffSTG 有 ddim_steps/eta、没有
                # epochs_run/lr），因此不能按固定 BASE_KEYS 求交集；点预测视图
                # 保留主 JSON 中除 WIS 外的全部字段，WIS 视图全保留。
                drop = set() if view == "metrics_wis" else (WIS_KEYS - POINT_KEYS)
                payload = {k: v for k, v in full.items() if k not in drop}
                missing = extra - set(payload)
                if missing:
                    raise SystemExit(f"{src} 缺字段 {sorted(missing)}")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "w") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=1)
                if view == "metrics_point":
                    n_point += 1
                else:
                    n_wis += 1
        print(f"[{method}] views derived", flush=True)
    print(f"done: metrics_point {n_point}, metrics_wis {n_wis}")


if __name__ == "__main__":
    main()
