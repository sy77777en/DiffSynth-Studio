"""
ACWM Inference Script

Given a JSON file (same format as training metadata), generates predicted
future frames for each sample using the trained LoRA + ActionFFNEncoder,
with automatic temporal adapter detection from checkpoint.

Usage:
  # Baseline (no temporal adapter, no masked traj)
  python infer_acwm.py \
    --json_path ./subfolder_exp_split/test.json \
    --model_dir ./models/Wan2.2-TI2V-5B \
    --ckpt_path ./outputs/acwm_xxx/epoch_0.safetensors \
    --output_dir ./inference_results/baseline \
    --height 384 --width 640 --num_frames 17

  # With masked traj (requires masked_traj_frames in JSON)
  python infer_acwm.py \
    --json_path ./test.json \
    --model_dir ./models/Wan2.2-TI2V-5B \
    --ckpt_path ./outputs/acwm_xxx/epoch_0.safetensors \
    --output_dir ./inference_results/with_traj \
    --use_masked_traj \
    --height 384 --width 640 --num_frames 17

  # Temporal adapter is auto-detected from checkpoint keys.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from safetensors.torch import save_file
import tempfile

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from safetensors.torch import load_file

from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.core import ModelConfig
from diffsynth.models.wan_video_dit import TemporalAttentionAdapter


# ============================================================================
# ActionFFNEncoder (must match training definition)
# ============================================================================


class ActionFFNEncoder(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int, num_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(action_dim, embed_dim), nn.GELU()]
        for _ in range(max(0, num_layers - 2)):
            layers += [nn.Linear(embed_dim, embed_dim), nn.GELU()]
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mlp(actions))


# ============================================================================
# Visual condition encoding (masked traj)
# ============================================================================


def encode_visual_condition(
    pipe: WanVideoPipeline,
    obs_img: Image.Image,
    masked_traj_imgs: list[Image.Image],
    height: int,
    width: int,
    num_frames: int,
) -> torch.Tensor:
    """Encode obs + masked_traj into visual condition y."""
    device = pipe.device
    dtype = pipe.torch_dtype

    all_frames = [obs_img] + masked_traj_imgs
    frame_tensors = []
    for img in all_frames:
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        frame_tensors.append(t)

    cond_video = torch.stack(frame_tensors, dim=1)  # (3, T, H, W)

    T_cond = cond_video.shape[1]
    if T_cond < num_frames:
        pad = torch.zeros(3, num_frames - T_cond, height, width)
        cond_video = torch.cat([cond_video, pad], dim=1)
    elif T_cond > num_frames:
        cond_video = cond_video[:, :num_frames]

    cond_video = cond_video.to(dtype=dtype, device=device)
    y = pipe.vae.encode([cond_video], device=device)[0]
    y = y.to(dtype=dtype, device=device)

    T_lat = y.shape[1]
    H_lat, W_lat = y.shape[2], y.shape[3]
    n_real_frames = 1 + len(masked_traj_imgs)

    msk = torch.zeros(1, num_frames, H_lat, W_lat, device=device)
    msk[:, :n_real_frames] = 1

    msk = torch.cat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
        dim=1,
    )
    msk = msk.view(1, msk.shape[1] // 4, 4, H_lat, W_lat)
    msk = msk.transpose(1, 2)[0]  # (4, T_lat, H', W')

    y = torch.cat([msk, y])  # (20, T_lat, H', W')
    y = y.unsqueeze(0).to(dtype=dtype, device=device)
    return y


# ============================================================================
# Checkpoint loading
# ============================================================================


def split_checkpoint(state_dict: dict) -> tuple[dict, dict, dict]:
    """Split a unified checkpoint into action_encoder, lora, and temporal_adapter parts."""
    action_encoder_state = {}
    temporal_adapter_state = {}
    lora_state = {}

    for k, v in state_dict.items():
        if k.startswith("action_encoder."):
            new_key = k[len("action_encoder."):]
            action_encoder_state[new_key] = v
        elif "temporal_adapter" in k:
            temporal_adapter_state[k] = v
        else:
            lora_state[k] = v

    return action_encoder_state, lora_state, temporal_adapter_state


def detect_temporal_adapter_layers(state_dict: dict) -> list[int]:
    """Detect which block indices have temporal adapter weights."""
    layers = set()
    for k in state_dict:
        if "temporal_adapter" in k:
            parts = k.split(".")
            block_idx = int(parts[1])
            layers.add(block_idx)
    return sorted(layers)


# ============================================================================
# Pipeline setup
# ============================================================================


def load_pipeline(args) -> tuple[WanVideoPipeline, ActionFFNEncoder, bool]:
    """Load pipeline, apply LoRA + action encoder + optional temporal adapter.

    Returns (pipe, action_encoder, has_temporal_adapter).
    """
    # --- 1. Load base pipeline ---
    model_configs = [
        ModelConfig(path=[
            os.path.join(args.model_dir, "diffusion_pytorch_model-00001-of-00003.safetensors"),
            os.path.join(args.model_dir, "diffusion_pytorch_model-00002-of-00003.safetensors"),
            os.path.join(args.model_dir, "diffusion_pytorch_model-00003-of-00003.safetensors"),
        ]),
        ModelConfig(path=os.path.join(args.model_dir, "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(path=os.path.join(args.model_dir, "Wan2.2_VAE.pth")),
    ]

    tokenizer_path = os.path.join(args.model_dir, "google", "umt5-xxl")
    assert os.path.isdir(tokenizer_path), f"Tokenizer not found: {tokenizer_path}"
    tokenizer_config = ModelConfig(path=tokenizer_path)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    # --- 2. Load and split checkpoint ---
    print(f"Loading checkpoint: {args.ckpt_path}")
    full_state = load_file(args.ckpt_path)
    action_encoder_state, lora_state, temporal_adapter_state = split_checkpoint(full_state)

    has_temporal_adapter = len(temporal_adapter_state) > 0
    print(f"  LoRA keys:              {len(lora_state)}")
    print(f"  Action encoder keys:    {len(action_encoder_state)}")
    print(f"  Temporal adapter keys:  {len(temporal_adapter_state)}")
    print(f"  Temporal adapter:       {'YES' if has_temporal_adapter else 'NO'}")

    # --- 3. Load LoRA into DiT ---
    if lora_state:
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            lora_only_path = tmp.name
    
        save_file(lora_state, lora_only_path)
        pipe.load_lora(pipe.dit, lora_only_path, alpha=1.0)
        print("LoRA loaded with pipe.load_lora()")

    # --- 4. Insert and load temporal adapter ---
    if has_temporal_adapter:
        ta_layers = detect_temporal_adapter_layers(temporal_adapter_state)
        print(f"  Temporal adapter layers: {ta_layers}")
        for i, block in enumerate(pipe.dit.blocks):
            if i in ta_layers:
                block.use_temporal_adapter = True
                block.temporal_adapter = TemporalAttentionAdapter(
                    dim=pipe.dit.dim,
                    num_heads=block.num_heads,
                ).to(device=pipe.device, dtype=pipe.torch_dtype)

        missing, unexpected = pipe.dit.load_state_dict(temporal_adapter_state, strict=False)
        print(f"  Temporal adapter loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    # --- 5. Load action encoder ---
    action_encoder = ActionFFNEncoder(
        action_dim=args.action_dim,
        embed_dim=args.action_embed_dim,
        num_layers=args.action_num_layers,
    )
    if action_encoder_state:
        action_encoder.load_state_dict(action_encoder_state)
        print("  Action encoder loaded")
    else:
        print("  [WARN] No action encoder weights in checkpoint!")
    action_encoder.eval().to(device=pipe.device, dtype=pipe.torch_dtype)

    return pipe, action_encoder, has_temporal_adapter


# ============================================================================
# Inference
# ============================================================================


@torch.no_grad()
def run_inference(
    pipe: WanVideoPipeline,
    action_encoder: ActionFFNEncoder,
    sample: dict,
    args,
) -> list[Image.Image]:
    """Run inference on a single sample."""
    device = pipe.device
    dtype = pipe.torch_dtype

    obs_img = Image.open(sample["obs_frame"]).convert("RGB")

    actions = torch.tensor(sample["actions"][:16], dtype=torch.float32)
    actions = actions.unsqueeze(0).to(device=device, dtype=dtype)
    action_tokens = action_encoder(actions)

    preencoded_visual_latent = None
    skip_condition_vae_encode = False

    if args.use_masked_traj and "masked_traj_frames" in sample:
        masked_traj_imgs = [
            Image.open(p).convert("RGB")
            for p in sample["masked_traj_frames"][:16]
        ]
        pipe.load_models_to_device(["vae"])
        preencoded_visual_latent = encode_visual_condition(
            pipe, obs_img, masked_traj_imgs,
            args.height, args.width, args.num_frames,
        )
        skip_condition_vae_encode = True

    video = pipe(
        prompt="",
        negative_prompt="",
        input_image=obs_img,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        tiled=True,
        preencoded_action_tokens=action_tokens,
        preencoded_visual_latent=preencoded_visual_latent,
        skip_condition_vae_encode=skip_condition_vae_encode,
    )

    return video


def save_results(video, sample: dict, output_dir: str, sample_idx: int):
    """Save generated frames as images."""
    task_name = sample.get("task", f"sample_{sample_idx:04d}")
    sample_dir = os.path.join(output_dir, task_name)
    os.makedirs(sample_dir, exist_ok=True)

    if isinstance(video, torch.Tensor):
        video = video.squeeze(0)
        frames = []
        for t in range(video.shape[1]):
            frame = video[:, t].permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(frame))
    elif isinstance(video, list) and len(video) > 0 and isinstance(video[0], Image.Image):
        frames = video
    elif isinstance(video, list) and len(video) > 0 and isinstance(video[0], list):
        frames = video[0]
    else:
        frames = video

    for i, frame in enumerate(frames):
        frame.save(os.path.join(sample_dir, f"frame_{i:04d}.png"))

    with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
        json.dump({
            "task": task_name,
            "obs_frame": sample["obs_frame"],
            "target_frames": sample.get("target_frames", []),
            "num_generated_frames": len(frames),
        }, f, indent=2)

    return len(frames)


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="ACWM Inference")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to test metadata JSON file.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to Wan2.2-TI2V-5B model directory.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to unified checkpoint (.safetensors) containing "
                             "LoRA + action_encoder + optional temporal_adapter.")
    parser.add_argument("--output_dir", type=str, default="./inference_results")

    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=17)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--action_embed_dim", type=int, default=1024)
    parser.add_argument("--action_num_layers", type=int, default=2)

    parser.add_argument("--use_masked_traj", action="store_true", default=False,
                        help="Use masked_traj_frames from JSON if available.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to run (None=all).")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.json_path, "r") as f:
        samples = json.load(f)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    print(f"Loaded {len(samples)} samples from {args.json_path}")

    pipe, action_encoder, has_ta = load_pipeline(args)
    print(f"\nReady. temporal_adapter={'enabled' if has_ta else 'disabled'}, "
          f"masked_traj={'enabled' if args.use_masked_traj else 'disabled'}\n")

    for idx, sample in enumerate(tqdm(samples, desc="Inference")):
        video = run_inference(pipe, action_encoder, sample, args)
        n = save_results(video, sample, args.output_dir, idx)
        if idx == 0:
            task = sample.get("task", "")
            print(f"  First sample: {n} frames -> {args.output_dir}/{task}")

    print(f"\nDone. Results in {args.output_dir}")


if __name__ == "__main__":
    main()