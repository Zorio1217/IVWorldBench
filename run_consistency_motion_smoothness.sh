#!/usr/bin/env bash
# Auto-evaluation script for Consistency Motion Smoothness metrics
# Uses VFIMamba frame interpolation to evaluate motion smoothness (SSIM/LPIPS)

set -e

# OpenAI API configuration (used by physics_realism dimension)
OPENAI_API_KEY="${OPENAI_API_KEY:-your-api-key-here}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export OPENAI_API_KEY
export OPENAI_BASE_URL



DEVICE="${DEVICE:-cuda:0}"
DATASET_BASE_DIR="${DATASET_BASE_DIR:-/data/videos/}"

echo "=========================================="
echo "Starting Consistency Motion Smoothness evaluation on device: $DEVICE"
echo "Dataset base directory: $DATASET_BASE_DIR"
echo "=========================================="

# Video-to-4D models
echo "Evaluating video-to-4D models..."
# [DONE] output_uniform_ex4d: 0.8535
# bash evaluate.sh \
#   --dataset_json $DATASET_BASE_DIR/condition_to_4D/video-to-any/video-to-4D-nonphysical.json \
#   --model output_uniform_ex4d --dimension consistency_motion_smoothness --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/video-to-any/video-to-4D-nonphysical.json \
  --model output_uniform_recam --dimension consistency_motion_smoothness --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/video-to-any/video-to-4D-nonphysical.json \
  --model output_uniform_traj --dimension consistency_motion_smoothness --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/video-to-any/video-to-4D-nonphysical.json \
  --model output_uniform_vista --dimension consistency_motion_smoothness --device $DEVICE

# Text-to-4D models
echo "Evaluating text-to-4D models..."
bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/text-to-any/text_to_4D_nonphysical.json \
  --model dream_in_4D --dimension consistency_motion_smoothness --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/text-to-any/text_to_4D_nonphysical.json \
  --model 4d-fy --dimension consistency_motion_smoothness --device $DEVICE


# Image-to-4D models
echo "Evaluating image-to-4D models..."
bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/image-to-any/image-any-nonphysical-scene-image-to-4D.json \
  --model CamI2V --dimension consistency_motion_smoothness --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/image-to-any/image-any-nonphysical-scene-image-to-4D.json \
  --model Diffusion_as_Shader --dimension consistency_motion_smoothness --device $DEVICE

echo "=========================================="
echo "Consistency Motion Smoothness evaluation completed!"
echo "=========================================="
