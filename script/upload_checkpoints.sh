#!/usr/bin/env bash
# 上传当前概率预测主线的训练权重到 HuggingFace model repo。
#
#   HF_ENDPOINT=https://hf-mirror.com ./script/upload_checkpoints.sh yzzpz/Charged-checkpoints
#
# 只上传 .pth；starts / keep / metrics 等小文件留在本地 result/。
# 显式排除 result/timediff_clean（future-mixup 版采样崩塌，不能用于对比）。
#
# 注意：仓库需先在 https://huggingface.co/new 建好（或用 `hf repo create`）。
# 这里走 HfApi.upload_folder，不会调用 /api/repos/create —— 部分镜像
# （如 hf-mirror.com）对该 POST 会 SSL 断连，而 create_commit 正常。
set -euo pipefail
REPO="${1:-yzzpz/Charged-checkpoints}"
cd "$(dirname "$0")/.."
: "${HF_ENDPOINT:=https://huggingface.co}"
export HF_ENDPOINT

REPO="$REPO" python3 - <<'PY'
import os
from huggingface_hub import HfApi

repo = os.environ["REPO"]
jobs = [
    ("result/kairos_l1_clean", "kairos_l1_clean",
     "Add Kairos L1 clean checkpoints (plain + quant, pre72)"),
    ("result/patchtst_quant_clean", "patchtst_quant_clean",
     "Add PatchTST + quantile-head ablation checkpoints"),
    ("result/csdi_clean", "csdi_clean",
     "Add CSDI and zidiff clean diffusion baselines"),
    ("result/timediff_nomix_clean", "timediff_nomix_clean",
     "Add TimeDiff no-mixup checkpoints (MEL/SPO)"),
]
api = HfApi()
for src, dst, msg in jobs:
    print(f"[upload] {src} -> {dst}", flush=True)
    api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=src,
        path_in_repo=dst,
        allow_patterns=["*.pth"],
        commit_message=msg,
    )
api.upload_file(
    repo_id=repo,
    repo_type="model",
    path_or_fileobj="docs/checkpoints_manifest.md",
    path_in_repo="README.md",
    commit_message="Add checkpoint manifest",
)
PY

echo "[done] https://huggingface.co/$REPO"
