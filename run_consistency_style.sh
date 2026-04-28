#!/usr/bin/env bash
# Auto-evaluation script for Consistency Style metrics
# Uses VGG19 Gram Matrix to evaluate style consistency across frames

set -e

# OpenAI API configuration (used by physics_realism dimension)
OPENAI_API_KEY="${OPENAI_API_KEY:-your-api-key-here}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export OPENAI_API_KEY
export OPENAI_BASE_URL



DEVICE="${DEVICE:-cuda:0}"
DATASET_BASE_DIR="${DATASET_BASE_DIR:-/data/videos/}"

echo "=========================================="
echo "Starting Consistency Style evaluation on device: $DEVICE"
echo "Dataset base directory: $DATASET_BASE_DIR"
echo "=========================================="

# Video-to-4D models
echo "Evaluating video-to-4D models..."
bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/video-to-any/video-to-4D-nonphysical.json \
  --model output_uniform_ex4d --dimension consistency_style --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/video-to-any/video-to-4D-nonphysical.json \
  --model output_uniform_recam --dimension consistency_style --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/video-to-any/video-to-4D-nonphysical.json \
  --model output_uniform_traj --dimension consistency_style --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/video-to-any/video-to-4D-nonphysical.json \
  --model output_uniform_vista --dimension consistency_style --device $DEVICE

# Text-to-4D models
echo "Evaluating text-to-4D models..."
bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/text-to-any/text_to_4D_nonphysical.json \
  --model dream_in_4D --dimension consistency_style --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/text-to-any/text_to_4D_nonphysical.json \
  --model 4d-fy --dimension consistency_style --device $DEVICE

# Text-to-3D models
echo "Evaluating text-to-3D models..."
bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/text-to-any/text_to_3d_dataset.json \
  --model 4d-fy --dimension consistency_style --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/text-to-any/text_to_3d_dataset.json \
  --model director3d --dimension consistency_style --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/text-to-any/text_to_3d_dataset.json \
  --model step1x-3d --dimension consistency_style --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/text-to-any/text_to_3d_dataset.json \
  --model text2nerf --dimension consistency_style --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/text-to-any/text_to_3d_dataset.json \
  --model wonderjourney --dimension consistency_style --device $DEVICE

# Image-to-3D models (object)
echo "Evaluating image-to-3D models (objects)..."
bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/image-to-any/image-any-nonphysical_object-image-to-3D.json \
  --model SyncDreamer --dimension consistency_style --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/image-to-any/image-any-nonphysical_object-image-to-3D.json \
  --model V3D --dimension consistency_style --device $DEVICE

# Image-to-3D models (scene)
echo "Evaluating image-to-3D models (scenes)..."
bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/image-to-any/image-any-nonphysical-scene-image-to-3D.json \
  --model FlexWorld --dimension consistency_style --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/image-to-any/image-any-nonphysical-scene-image-to-3D.json \
  --model ViewCrafter --dimension consistency_style --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/image-to-any/image-any-nonphysical-scene-image-to-3D.json \
  --model motionctrl --dimension consistency_style --device $DEVICE

# Image-to-4D models
echo "Evaluating image-to-4D models..."
bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/image-to-any/image-any-nonphysical-scene-image-to-4D.json \
  --model CamI2V --dimension consistency_style --device $DEVICE

bash evaluate.sh \
  --dataset_json $DATASET_BASE_DIR/condition_to_4D/image-to-any/image-any-nonphysical-scene-image-to-4D.json \
  --model Diffusion_as_Shader --dimension consistency_style --device $DEVICE

echo "=========================================="
echo "Consistency Style evaluation completed!"
echo "=========================================="
