#!/usr/bin/env bash

# PhyWordBench entry: evaluate using dataset JSON and our runner
# Examples:
#   bash /data1/luyt/WorldModelBench/phywordbench/evaluate.sh \
#     --dataset_json /data2/luowei/Videos/condition_to_4D/text-to-any/text_to_3d_dataset.json \
#     --model my_model --device cuda:0
#   bash /data1/luyt/WorldModelBench/phywordbench/evaluate.sh \
#     --dataset_json /data2/.../text_to_3d_dataset.json --model my_model \
#     --dimension complex_plot --prompt "a complex story"

set -euo pipefail

DATASET_JSON=""
MODEL_NAME=""
DIMENSION=""
PROMPT=""
DEVICE=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --dataset_json)
      DATASET_JSON="$2"; shift; shift;;
    --model)
      MODEL_NAME="$2"; shift; shift;;
    --dimension)
      DIMENSION="$2"; shift; shift;;
    --prompt)
      PROMPT="$2"; shift; shift;;
    --device)
      DEVICE="$2"; shift; shift;;
    *)
      echo "Unknown option: $1"; exit 1;;
  esac
done

if [[ -z "$DATASET_JSON" || -z "$MODEL_NAME" ]]; then
  echo "Required: --dataset_json PATH and --model NAME"; exit 1
fi

RUNNER="runner.py"

CMD=(python "$RUNNER" --dataset_json "$DATASET_JSON" --model "$MODEL_NAME" --device "$DEVICE")
if [[ -n "$DIMENSION" ]]; then
  CMD+=(--dimension "$DIMENSION")
fi
if [[ -n "$PROMPT" ]]; then
  CMD+=(--prompt "$PROMPT")
fi

echo "Running: ${CMD[@]}"
"${CMD[@]}"


