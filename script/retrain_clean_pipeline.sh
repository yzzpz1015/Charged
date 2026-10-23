#!/usr/bin/env bash
# 清理后重训流水线：每城一条链，六城并行。
#   1. 等 plain 权重就绪（本轮已在外面启动）
#   2. quant 分位头
#   3. 做法④ CSDI
#   4. 做法⑤ 吸收态扩散 zidiff
# 每城固定一块 GPU，链内串行、链间并行。所有产物写 *_clean / csdi_clean，
# 不覆盖清理前的旧结果。
set -u
cd "$(dirname "$0")/.."

CITIES=(LOA JHB MEL SPO SZH)
export HF_HUB_OFFLINE=1

run_city() {
  local c=$1 gpu=$2
  local want="result/kairos_l1_clean/$c/plain_pre72_fold6.pth"
  while [ ! -f "$want" ]; do sleep 30; done

  echo "[$(date '+%F %T')] $c quant start (GPU$gpu)" >&2
  python -u script/train_kairos_l1.py --city "$c" --variant quant --pre_len 72 \
    --epochs 30 --device "$gpu" --out result/kairos_l1_clean \
    > "/tmp/clean_quant_$c.log" 2>&1

  echo "[$(date '+%F %T')] $c csdi start (GPU$gpu)" >&2
  python -u script/train_csdi.py --cities "$c" --device "$gpu" \
    --out result/csdi_clean --ckpt_root result/kairos_l1_clean \
    > "/tmp/clean_csdi_$c.log" 2>&1

  echo "[$(date '+%F %T')] $c zidiff start (GPU$gpu)" >&2
  python -u script/train_zidiff.py --cities "$c" --device "$gpu" \
    --out result/csdi_clean --ckpt_root result/kairos_l1_clean \
    > "/tmp/clean_zidiff_$c.log" 2>&1

  echo "[$(date '+%F %T')] $c done" >&2
}

i=0
for c in "${CITIES[@]}"; do
  run_city "$c" "$i" &
  i=$((i + 1))
done
wait
echo "ALL DONE"
