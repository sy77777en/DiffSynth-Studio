#!/usr/bin/env bash
set -e
export TOKENIZERS_PARALLELISM=false

# ============================================================================
# Usage:
#   bash wan_acwm_finetune.sh [NUM_GPUS] [GRAD_ACCUM]
# Example:
#   bash wan_acwm_finetune.sh 4 2
#   bash wan_acwm_finetune.sh 1 4
# ============================================================================

NUM_GPUS="${1:-4}"
GRAD_ACCUM="${2:-2}"

# ============================================================================
# Paths
# ============================================================================
MODEL_DIR="/net/holy-isilon/ifs/rc_labs/ydu_lab/sycen/code/DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B"
DIFFSYNTH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ACWM_CONFIG="${DIFFSYNTH_DIR}/configs/action_conditioning.yaml"

# If dataset/model path are already in YAML, you can remove METADATA_JSON/MODEL_DIR usage
# from argparse later. For now keep metadata explicit unless train_acwm.py reads it from YAML.
METADATA_JSON="${DIFFSYNTH_DIR}/train_metadata_100.json"

# ============================================================================
# Hyperparameters
# ============================================================================
LORA_RANK=16
LEARNING_RATE="1e-4"
ACTION_ENCODER_LR="1e-4"
NUM_EPOCHS=10
DATASET_REPEAT=2
SAVE_STEPS=500

ENABLE_TEMPORAL_ADAPTER=0

TRAIN_HEIGHT=368
TRAIN_WIDTH=640
TRAIN_NUM_FRAMES=17

MAX_TIMESTEP=1.0
MIN_TIMESTEP=0.0

LORA_BASE_MODEL="dit"
CKPT_PREFIX="pipe.dit."

LOG_EVERY_N_STEPS=50
LOG_LOSS_TO_CSV=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_TAG="acwm_ti2v5b_${NUM_GPUS}gpu"
OUTPUT_PATH="${DIFFSYNTH_DIR}/outputs/${TRAIN_TAG}_${TIMESTAMP}"
mkdir -p "$OUTPUT_PATH"

echo "========================================="
echo "ACWM Training"
echo "========================================="
echo "GPUs:       $NUM_GPUS"
echo "Grad accum: $GRAD_ACCUM"
echo "Config:     $ACWM_CONFIG"
echo "Metadata:   $METADATA_JSON"
echo "Output:     $OUTPUT_PATH"
echo "========================================="

# ============================================================================
# Sanity checks
# ============================================================================
if [ ! -f "$ACWM_CONFIG" ]; then
  echo "Error: ACWM config not found: $ACWM_CONFIG"
  exit 1
fi

if [ ! -f "$METADATA_JSON" ]; then
  echo "Error: metadata not found: $METADATA_JSON"
  exit 1
fi

# Optional model sanity checks
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

# ============================================================================
# Accelerate config
# ============================================================================
ACCEL_CONFIG="/tmp/wan_acwm_accelerate_${NUM_GPUS}gpu_$$.yaml"

if [ "$NUM_GPUS" -eq 1 ]; then
cat > "$ACCEL_CONFIG" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: NO
mixed_precision: bf16
num_machines: 1
num_processes: 1
EOF
else
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
fi

# ============================================================================
# Model paths
# ============================================================================
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

OPTIONAL_ARGS+=(--action_encoder_lr "$ACTION_ENCODER_LR")
OPTIONAL_ARGS+=(--log_every_n_steps "$LOG_EVERY_N_STEPS")

if [ "${LOG_LOSS_TO_CSV:-0}" = "1" ]; then
  OPTIONAL_ARGS+=(--log_loss_to_csv)
fi

if [ -d "${MODEL_DIR}/google/umt5-xxl" ]; then
  OPTIONAL_ARGS+=(--tokenizer_path "${MODEL_DIR}/google/umt5-xxl")
fi

if [ "${ENABLE_TEMPORAL_ADAPTER:-0}" = "1" ]; then
  OPTIONAL_ARGS+=(--enable_temporal_adapter)
fi

cd "$DIFFSYNTH_DIR"

accelerate launch \
  --config_file "$ACCEL_CONFIG" \
  examples/wanvideo/model_training/train_acwm.py \
  --acwm_config "$ACWM_CONFIG" \
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
echo "Done. Checkpoints: $OUTPUT_PATH"
