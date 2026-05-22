#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/private/training_tables}"
ACCELERATOR="${ACCELERATOR:-gpu}"

run_column() {
  local column="$1"
  local rs_threshold="$2"

  python scripts/train_stage1_denoise.py \
    --approach approach3 \
    --column "${column}" \
    --joint_train \
    --rs_threshold "${rs_threshold}" \
    --data_dir "${DATA_DIR}" \
    --out_dir "outputs/reproduced/${column}" \
    --split scaffold \
    --seed 2028 \
    --n_seeds 5 \
    --mpnn_hidden 600 \
    --mpnn_depth 4 \
    --ffn_hidden 600 \
    --dropout 0.1 \
    --patience 30 \
    --epochs 200 \
    --accelerator "${ACCELERATOR}"
}

run_column IA 0.5
run_column IB 0.75
run_column IC 1.5
run_column IG 0.75

