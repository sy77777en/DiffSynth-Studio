#!/usr/bin/env bash
# ACWM LoRA + ActionFFN on Wan2.2-TI2V-5B (single DiT, no high/low split).
# Layout: diffusion_pytorch_model-{01,02,03}-of-00003.safetensors + T5 + Wan2.2_VAE.pth (+ google/umt5 tokenizer).
#
# Usage:
#   bash wan_acwm_finetune_ti2v5b_4gpu.sh
#
# Requires: 4 GPUs (DeepSpeed ZeRO-2, bf16). Tune GRAD_ACCUM / NUM_GPUS if OOM.

set -e
export TOKENIZERS_PARALLELISM=false

# ============================================================================
# >>> EDIT THESE PATHS <<<
# ============================================================================
# Root of Wan2.2-TI2V-5B (contains diffusion shards, T5, VAE, google/umt5-xxl, …)
MODEL_DIR="/net/holy-isilon/ifs/rc_labs/ydu_lab/sycen/code/DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B"

DIFFSYNTH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METADATA_JSON="${DIFFSYNTH_DIR}/subfolder_exp_split/train.json"

LORA_INIT_PATH=""
ACTION_ENCODER_LR="1e-4"
LOG_EVERY_N_STEPS=50
LOG_LOSS_TO_CSV=1

# Optional second-stage-style presets (same checkpoint format as 14B ACWM)
PRESET_LORA_DIT_PATH=""
ACTION_ENCODER_CKPT=""
USE_FREEZE_ACTION_ENCODER=0

# Hardware
NUM_GPUS=1
# Per-process gradient accumulation (effective optimizer steps scale with NUM_GPUS * GRAD_ACCUM)
GRAD_ACCUM=2

# Hyperparameters
LORA_RANK=16
LEARNING_RATE="1e-4"
NUM_EPOCHS=10
DATASET_REPEAT=2
SAVE_STEPS=500

# Feature toggles
ENABLE_TEMPORAL_ADAPTER=0   # 0=baseline, 1=enable temporal attention in DiT blocks
USE_MASKED_TRAJ=1            # 1=use masked traj visual condition, 0=obs-only

# Resolution / frames — TI2V official LoRA example uses 480x832 x 49f; ACWM uses 17 f (1 obs + 16 targets).
TRAIN_HEIGHT=384
TRAIN_WIDTH=640
TRAIN_NUM_FRAMES=17

# Single DiT: full FlowMatch timestep range [0, 1]
MAX_TIMESTEP=1.0
MIN_TIMESTEP=0.0
LORA_BASE_MODEL="dit"
CKPT_PREFIX="pipe.dit."
# ============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_TAG="acwm_ti2v5b_4gpu"
OUTPUT_PATH="${DIFFSYNTH_DIR}/outputs/acwm_ti2v5b_${TRAIN_TAG}_${TIMESTAMP}"
mkdir -p "$OUTPUT_PATH"

echo "========================================="
echo "ACWM — Wan2.2-TI2V-5B — ${NUM_GPUS} GPUs"
echo "========================================="
echo "MODEL_DIR:       $MODEL_DIR"
echo "LoRA base:       $LORA_BASE_MODEL  (strip ckpt prefix: ${CKPT_PREFIX})"
echo "Timestep:        [$MIN_TIMESTEP, $MAX_TIMESTEP]"
echo "Frames:          ${TRAIN_NUM_FRAMES}  (${TRAIN_HEIGHT}x${TRAIN_WIDTH})"
echo "GPUs:            $NUM_GPUS  grad_accum=$GRAD_ACCUM"
echo "Temporal adapter: $ENABLE_TEMPORAL_ADAPTER"
echo "Masked traj:      $USE_MASKED_TRAJ"
echo "Output:          $OUTPUT_PATH"
echo "========================================="

# ----------------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------------
for f in \
  "diffusion_pytorch_model-00001-of-00003.safetensors" \
  "diffusion_pytorch_model-00002-of-00003.safetensors" \
  "diffusion_pytorch_model-00003-of-00003.safetensors" \
  "models_t5_umt5-xxl-enc-bf16.pth" \
  "Wan2.2_VAE.pth"
do
  if [ ! -f "${MODEL_DIR}/${f}" ]; then
    echo "Error: missing ${MODEL_DIR}/${f}"
    exit 1
  fi
done
if [ ! -f "$METADATA_JSON" ]; then
  echo "Error: metadata not found: $METADATA_JSON"
  exit 1
fi
if [ -n "$LORA_INIT_PATH" ] && [ ! -f "$LORA_INIT_PATH" ]; then
  echo "Error: LoRA checkpoint not found: $LORA_INIT_PATH"
  exit 1
fi

# ----------------------------------------------------------------------------
# Accelerate + DeepSpeed ZeRO-2
# ----------------------------------------------------------------------------
ACCEL_CONFIG="/tmp/wan_acwm_ti2v5b_accelerate_$$.yaml"
cat > "$ACCEL_CONFIG" << EOF
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: ${GRAD_ACCUM}
  gradient_clipping: 1.0
  zero_stage: 2
distributed_type: DEEPSPEED
mixed_precision: bf16
num_machines: 1
num_processes: ${NUM_GPUS}
EOF

# Order: DiT shards → T5 → VAE (single wan_video_dit → pipe.dit only; pipe.dit2 stays unused)
MODEL_PATHS_JSON="[
  [
    \"${MODEL_DIR}/diffusion_pytorch_model-00001-of-00003.safetensors\",
    \"${MODEL_DIR}/diffusion_pytorch_model-00002-of-00003.safetensors\",
    \"${MODEL_DIR}/diffusion_pytorch_model-00003-of-00003.safetensors\"
  ],
  \"${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth\",
  \"${MODEL_DIR}/Wan2.2_VAE.pth\"
]"

OPTIONAL_ARGS=()
if [ -n "$LORA_INIT_PATH" ]; then
  OPTIONAL_ARGS+=(--lora_checkpoint "$LORA_INIT_PATH")
fi
if [ -n "$ACTION_ENCODER_LR" ]; then
  OPTIONAL_ARGS+=(--action_encoder_lr "$ACTION_ENCODER_LR")
fi
OPTIONAL_ARGS+=(--log_every_n_steps "$LOG_EVERY_N_STEPS")
if [ "${LOG_LOSS_TO_CSV:-0}" = "1" ]; then
  OPTIONAL_ARGS+=(--log_loss_to_csv)
fi
if [ -n "${PRESET_LORA_DIT_PATH:-}" ]; then
  OPTIONAL_ARGS+=(--preset_lora_path "$PRESET_LORA_DIT_PATH" --preset_lora_model "dit")
fi
if [ -n "${ACTION_ENCODER_CKPT:-}" ]; then
  OPTIONAL_ARGS+=(--action_encoder_checkpoint "$ACTION_ENCODER_CKPT")
fi
if [ "${USE_FREEZE_ACTION_ENCODER:-0}" = "1" ]; then
  OPTIONAL_ARGS+=(--freeze_action_encoder)
fi
if [ "${ENABLE_TEMPORAL_ADAPTER:-0}" = "1" ]; then
  OPTIONAL_ARGS+=(--enable_temporal_adapter)
fi
if [ "${USE_MASKED_TRAJ:-1}" = "0" ]; then
  OPTIONAL_ARGS+=(--no_masked_traj)
fi
# Local tokenizer (recommended if MODEL_DIR contains google/umt5-xxl)
if [ -d "${MODEL_DIR}/google/umt5-xxl" ]; then
  OPTIONAL_ARGS+=(--tokenizer_path "${MODEL_DIR}/google/umt5-xxl")
elif [ -d "${MODEL_DIR}/google" ]; then
  echo "[warn] No ${MODEL_DIR}/google/umt5-xxl — using built-in tokenizer_config from train.py."
fi

cd "$DIFFSYNTH_DIR"

accelerate launch \
  --config_file "$ACCEL_CONFIG" \
  examples/wanvideo/model_training/train_acwm.py \
  --dataset_base_path . \
  --dataset_metadata_path "$METADATA_JSON" \
  --dataset_num_workers 8 \
  --height "$TRAIN_HEIGHT" \
  --width "$TRAIN_WIDTH" \
  --num_frames "$TRAIN_NUM_FRAMES" \
  --dataset_repeat "$DATASET_REPEAT" \
  --model_paths "$MODEL_PATHS_JSON" \
  --learning_rate "$LEARNING_RATE" \
  --num_epochs "$NUM_EPOCHS" \
  --save_steps "$SAVE_STEPS" \
  --remove_prefix_in_ckpt "$CKPT_PREFIX" \
  --output_path "$OUTPUT_PATH" \
  --lora_base_model "$LORA_BASE_MODEL" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --use_gradient_checkpointing_offload \
  --lora_rank "$LORA_RANK" \
  --max_timestep_boundary "$MAX_TIMESTEP" \
  --min_timestep_boundary "$MIN_TIMESTEP" \
  --extra_inputs "input_image" \
  "${OPTIONAL_ARGS[@]}"

echo ""
echo "========================================="
echo "Done. Checkpoints: $OUTPUT_PATH"
echo "========================================="
