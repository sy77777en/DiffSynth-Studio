#!/usr/bin/env bash
# ACWM fine-tuning (train_acwm.py): DiT LoRA + ActionFFNEncoder + ConditionEncoder path.
# Layout mirrors wan_lora_train.sh (modes, DeepSpeed ZeRO-2, sharded DiT, optional LoRA resume).
#
# Usage:
#   bash wan_acwm_finetune.sh              # default high_noise
#   bash wan_acwm_finetune.sh high_noise
#   bash wan_acwm_finetune.sh low_noise
#
# Typical model: Wan2.2-I2V-A14B (needs --extra_inputs input_image). For T2V-only checkpoints,
# adjust MODEL_DIR and paths below (still use train_acwm.py only if your pipeline matches).

set -e
export TOKENIZERS_PARALLELISM=false

# ============================================================================
# >>> EDIT THESE PATHS <<<
# ============================================================================
# Wan checkpoint root (contains high_noise_model/, low_noise_model/, T5, VAE).
# Recommended for ACWM: .../Wan-AI/Wan2.2-I2V-A14B
MODEL_DIR="/net/holy-isilon/ifs/rc_labs/ydu_lab/sycen/code/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B"

# ACWM dataset: JSON list (same format as prepare_training_data / acwm_dataset.py)
METADATA_JSON="/net/holy-isilon/ifs/rc_labs/ydu_lab/sycen/code/syc_test/DiffSynth-Studio/train_metadata_100.json"

# Repo root (DiffSynth-Studio)
DIFFSYNTH_DIR="/net/holy-isilon/ifs/rc_labs/ydu_lab/sycen/code/syc_test/DiffSynth-Studio"

# action_conditioning.yaml (experiments.wan must match --acwm_experiment if set)
ACWM_CONFIG="${DIFFSYNTH_DIR}/configs/action_conditioning.yaml"

# Optional: resume DiT LoRA only (--lora_checkpoint). Leave empty to train from scratch.
LORA_INIT_PATH=""
# Example:
# LORA_INIT_PATH="/path/to/step-2000.safetensors"

# Optional: separate LR for ActionFFNEncoder (omit or empty → same as LEARNING_RATE)
ACTION_ENCODER_LR="1e-4"

# Loss logging (train_acwm.py): print every N steps; append CSV under OUTPUT_PATH
LOG_EVERY_N_STEPS=50
LOG_LOSS_TO_CSV=1   # 1 = enable --log_loss_to_csv
# ============================================================================

MODE="${1:-high_noise}"

# Per-mode timestep range for FlowMatch (same split style as your wan_lora_train.sh T2V script).
# For Wan2.2-I2V-A14B official examples often use max≈0.358 for high_noise — adjust if needed.
if [ "$MODE" == "high_noise" ]; then
    NOISE_SUBDIR="high_noise_model"
    MAX_TIMESTEP=0.417
    MIN_TIMESTEP=0
elif [ "$MODE" == "low_noise" ]; then
    NOISE_SUBDIR="low_noise_model"
    MAX_TIMESTEP=1
    MIN_TIMESTEP=0.417
else
    echo "Usage: bash wan_acwm_finetune.sh [high_noise|low_noise]"
    exit 1
fi

# Shared hyperparameters
LORA_RANK=16
LEARNING_RATE="1e-4"
NUM_EPOCHS=5
DATASET_REPEAT=10
SAVE_STEPS=500
NUM_GPUS=1

# Resolution / temporal length for ACWM (1 obs + 16 targets = 17 frames → T_latent=5)
TRAIN_HEIGHT=368
TRAIN_WIDTH=640
TRAIN_NUM_FRAMES=17

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_TAG="acwm_dit"
OUTPUT_PATH="${DIFFSYNTH_DIR}/outputs/acwm_${MODE}_lr${LEARNING_RATE}_r${LORA_RANK}_${TRAIN_TAG}_${TIMESTAMP}"
mkdir -p "$OUTPUT_PATH"

echo "========================================="
echo "Wan ACWM Fine-tuning (train_acwm.py)"
echo "========================================="
echo "Mode:            $MODE"
echo "Noise subdir:    $NOISE_SUBDIR"
echo "LoRA Rank:       $LORA_RANK"
echo "LR (DiT LoRA):   $LEARNING_RATE"
if [ -n "$ACTION_ENCODER_LR" ]; then
    echo "LR (ActionFFN):  $ACTION_ENCODER_LR"
else
    echo "LR (ActionFFN):  (same as DiT LoRA)"
fi
echo "Epochs:          $NUM_EPOCHS"
echo "Dataset Repeat:  $DATASET_REPEAT"
echo "Save Steps:      $SAVE_STEPS"
echo "Timestep:        [$MIN_TIMESTEP, $MAX_TIMESTEP]"
echo "Frames:          ${TRAIN_NUM_FRAMES}  (${TRAIN_HEIGHT}x${TRAIN_WIDTH})"
echo "Eff. batch (est): $((NUM_GPUS * 4))  (num_processes * grad_accum in accelerate config)"
echo "Output:          $OUTPUT_PATH"
if [ -n "$LORA_INIT_PATH" ]; then
    echo "LoRA init:       $LORA_INIT_PATH"
fi
echo "========================================="

# ============================================================================
# Verify paths
# ============================================================================
NOISE_MODEL_DIR="${MODEL_DIR}/${NOISE_SUBDIR}"
if [ ! -d "$NOISE_MODEL_DIR" ]; then
    echo "Error: ${NOISE_SUBDIR} not found at $NOISE_MODEL_DIR"
    exit 1
fi
for f in "models_t5_umt5-xxl-enc-bf16.pth" "Wan2.1_VAE.pth"; do
    if [ ! -f "${MODEL_DIR}/${f}" ]; then
        echo "Error: ${f} not found in ${MODEL_DIR}"
        exit 1
    fi
done
if [ ! -f "$METADATA_JSON" ]; then
    echo "Error: metadata not found: $METADATA_JSON"
    exit 1
fi
if [ ! -f "$ACWM_CONFIG" ]; then
    echo "Error: ACWM config not found: $ACWM_CONFIG"
    exit 1
fi
if [ -n "$LORA_INIT_PATH" ] && [ ! -f "$LORA_INIT_PATH" ]; then
    echo "Error: LoRA checkpoint not found at $LORA_INIT_PATH"
    exit 1
fi

# First sharded file must exist (adjust pattern if your checkpoint uses different split count)
if [ ! -f "${NOISE_MODEL_DIR}/diffusion_pytorch_model-00001-of-00006.safetensors" ]; then
    echo "Error: expected sharded DiT at ${NOISE_MODEL_DIR}/diffusion_pytorch_model-00001-of-00006.safetensors"
    exit 1
fi

# ============================================================================
# Accelerate + DeepSpeed ZeRO-2 (same pattern as wan_lora_train.sh)
# ============================================================================
ACCEL_CONFIG="/tmp/wan_acwm_accelerate_config_$$.yaml"
cat > "$ACCEL_CONFIG" << EOF
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 4
  gradient_clipping: 1.0
  zero_stage: 2
distributed_type: DEEPSPEED
mixed_precision: bf16
num_machines: 1
num_processes: ${NUM_GPUS}
EOF

# ============================================================================
# Model paths JSON (DiT shards + T5 + VAE) — matches UnifiedDataset-style loading via train.py paths
# ============================================================================
MODEL_PATHS_JSON="[
  [
    \"${NOISE_MODEL_DIR}/diffusion_pytorch_model-00001-of-00006.safetensors\",
    \"${NOISE_MODEL_DIR}/diffusion_pytorch_model-00002-of-00006.safetensors\",
    \"${NOISE_MODEL_DIR}/diffusion_pytorch_model-00003-of-00006.safetensors\",
    \"${NOISE_MODEL_DIR}/diffusion_pytorch_model-00004-of-00006.safetensors\",
    \"${NOISE_MODEL_DIR}/diffusion_pytorch_model-00005-of-00006.safetensors\",
    \"${NOISE_MODEL_DIR}/diffusion_pytorch_model-00006-of-00006.safetensors\"
  ],
  \"${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth\",
  \"${MODEL_DIR}/Wan2.1_VAE.pth\"
]"

# ============================================================================
# Optional args
# ============================================================================
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

# ============================================================================
# Launch
# ============================================================================
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
  --acwm_config "$ACWM_CONFIG" \
  --learning_rate "$LEARNING_RATE" \
  --num_epochs "$NUM_EPOCHS" \
  --save_steps "$SAVE_STEPS" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_PATH" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --use_gradient_checkpointing_offload \
  --lora_rank "$LORA_RANK" \
  --max_timestep_boundary "$MAX_TIMESTEP" \
  --min_timestep_boundary "$MIN_TIMESTEP" \
  --extra_inputs "input_image" \
  "${OPTIONAL_ARGS[@]}"

echo ""
echo "========================================="
echo "ACWM training complete!"
echo "Mode: $MODE"
echo "Checkpoints: $OUTPUT_PATH"
echo "========================================="
