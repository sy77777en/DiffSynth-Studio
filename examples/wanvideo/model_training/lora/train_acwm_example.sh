#!/usr/bin/env bash
# Example launcher for ACWM training (edit paths for your cluster).
# Train high-noise expert first, then low-noise (see Wan2.2-I2V-A14B.sh for timestep boundaries).

set -e
cd "$(dirname "$0")/../../.."  # DiffSynth-Studio repo root

export TOKENIZERS_PARALLELISM=false

MODEL_ROOT="Wan-AI/Wan2.2-I2V-A14B"
METADATA="./train_metadata_100.json"
ACWM_CFG="./configs/action_conditioning.yaml"
OUT="./models/train/acwm_high_noise"

# Comma-separated weights string — mirror lora/Wan2.2-I2V-A14B.sh high_noise split.
MODEL_SPEC="${MODEL_ROOT}:high_noise_model/diffusion_pytorch_model*.safetensors,${MODEL_ROOT}:models_t5_umt5-xxl-enc-bf16.pth,${MODEL_ROOT}:Wan2.1_VAE.pth"

accelerate launch examples/wanvideo/model_training/train_acwm.py \
  --dataset_base_path . \
  --dataset_metadata_path "${METADATA}" \
  --height 368 \
  --width 640 \
  --num_frames 17 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "${MODEL_SPEC}" \
  --acwm_config "${ACWM_CFG}" \
  --output_path "${OUT}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --lora_base_model dit \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs input_image \
  --learning_rate 1e-4 \
  --action_encoder_lr 1e-4 \
  --num_epochs 1 \
  --use_gradient_checkpointing \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0
