#!/usr/bin/env bash
# 上传当前概率预测主线的训练权重到 HuggingFace model repo。
#
#   HF_ENDPOINT=https://hf-mirror.com ./script/upload_checkpoints.sh yzzpz/Charged-checkpoints
#
# 只上传 .pth；starts / keep / metrics 等小文件留在本地 result/。
# 显式排除 result/timediff_clean（future-mixup 版采样崩塌，不能用于对比）。
set -euo pipefail
REPO="${1:-yzzpz/Charged-checkpoints}"
cd "$(dirname "$0")/.."
: "${HF_ENDPOINT:=https://huggingface.co}"
export HF_ENDPOINT

hf repo create "$REPO" --repo-type model --exist-ok

hf upload "$REPO" result/kairos_l1_clean kairos_l1_clean \
  --repo-type model --include "*.pth" \
  --commit-message "Add Kairos L1 clean checkpoints (plain + quant, pre72)"
hf upload "$REPO" result/patchtst_quant_clean patchtst_quant_clean \
  --repo-type model --include "*.pth" \
  --commit-message "Add PatchTST + quantile-head ablation checkpoints"
hf upload "$REPO" result/csdi_clean csdi_clean \
  --repo-type model --include "*.pth" \
  --commit-message "Add CSDI and zidiff clean diffusion baselines"
hf upload "$REPO" result/timediff_nomix_clean timediff_nomix_clean \
  --repo-type model --include "*.pth" \
  --commit-message "Add TimeDiff no-mixup checkpoints (MEL/SPO)"
hf upload "$REPO" docs/checkpoints_manifest.md README.md \
  --repo-type model --commit-message "Add checkpoint manifest"

echo "[done] https://huggingface.co/$REPO"
