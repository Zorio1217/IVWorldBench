#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
: "${MODEL:?export MODEL=my_model}"
: "${DIMENSION:?export DIMENSION=clip_iqa}"
: "${DATA_JSON:?export DATA_JSON=/abs/path/to/dataset.json}"

python "$(dirname "$0")/runner.py" \
  --dataset_json "${DATA_JSON}" \
  --model_name "${MODEL}" \
  --dimension "${DIMENSION}" \
  --device "${DEVICE}" \
  "$@"
