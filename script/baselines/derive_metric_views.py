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

BASE_KEYS = {
    "method", "city", "seed", "horizon", "seq_l", "pre_len", "n_sample",
    "stations", "crps_norm_sum", "crps_sum_norm", "crps_nz", "nz_rate",
    "n_scored_windows", "picp50", "picp80", "picp90", "mpiw50", "mpiw80",
    "mpiw90", "params", "best_valid", "elapsed_s", "n_test_windows",
    "epochs_run", "d_model", "layers", "lr", "knn",
}
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
                keep = BASE_KEYS | extra
                payload = {k: v for k, v in full.items() if k in keep}
                missing = (BASE_KEYS | extra) - set(payload)
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
