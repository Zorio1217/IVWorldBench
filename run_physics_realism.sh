#!/usr/bin/env bash
# Auto-evaluation script for Physics Realism metrics
# Physics Realism: Evaluates physical plausibility of generated 3D/4D content

set -e

# OpenAI API configuration (used by physics_realism dimension)
OPENAI_API_KEY="${OPENAI_API_KEY:-your-api-key-here}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export OPENAI_API_KEY
export OPENAI_BASE_URL

DEVICE="${DEVICE:-cuda:0}"
DATASET_BASE_DIR="${DATASET_BASE_DIR:-/data/videos}"

echo "=========================================="
echo "Starting Physics Realism evaluation on device: $DEVICE"
echo "Dataset base directory: $DATASET_BASE_DIR"
echo "=========================================="

# Video-to-4D models
echo "Evaluating video-to-4D models..."
bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/video-to-any/video-to-4D-physical.json \
  --model output_uniform_ex4d --dimension physics_realism --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/video-to-any/video-to-4D-physical.json \
  --model output_uniform_recam --dimension physics_realism --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/video-to-any/video-to-4D-physical.json \
  --model output_uniform_traj --dimension physics_realism --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/video-to-any/video-to-4D-physical.json \
  --model output_uniform_vista --dimension physics_realism --device $DEVICE

# Text-to-4D models
echo "Evaluating text-to-4D models..."
bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/text-to-any/text_to_4D_physical.json \
  --model dream_in_4D --dimension physics_realism --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/text-to-any/text_to_4D_physical.json \
  --model 4d-fy --dimension physics_realism --device $DEVICE


# Image-to-4D models
echo "Evaluating image-to-4D models..."
bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/image-to-any/image-any-physical-image-to-4D.json \
  --model CamI2V --dimension physics_realism --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/image-to-any/image-any-physical-image-to-4D.json \
  --model Diffusion_as_Shader --dimension physics_realism --device $DEVICE

echo "=========================================="
echo "Physics Realism evaluation completed!"
echo "=========================================="


