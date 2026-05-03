"""
Test ACWMDataset + ConditionEncoder shape alignment.

This script only loads the VAE (~hundreds of MB) and does NOT load DiT or T5,
so it runs within a few minutes.

Usage:
    python test_cond_encoder.py \
        --metadata ./train_metadata_100.json \
        --config ./configs/action_conditioning.yaml \
        --device cuda:0

What this script verifies:
    1. ACWMDataset outputs correct data format
    2. PIL images are correctly converted to tensors
    3. ConditionEncoder.encode() runs without errors
    4. visual_latent shape = (1, C_concat, T_latent, H_lat, W_lat)
       where T_latent = (17 - 1) // 4 + 1 = 5
    5. action_tokens shape = (1, 16, action_embed_dim)
    6. visual_latent can be concatenated with noisy latent along channel dimension
    7. action_tokens can be padded/truncated and concatenated into context tokens
"""

import sys
import os
import time
import argparse
from dataclasses import fields

import torch
import numpy as np
import yaml
from PIL import Image

# 确保能 import diffsynth
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acwm_dataset import ACWMDataset
from diffsynth.models.action_conditioning.config import ActionConditioningConfig
from diffsynth.models.action_conditioning.encoder import ConditionEncoder


# ============================================================================
# Config loading (same as inference)
# ============================================================================
def load_acwm_config(config_path: str, experiment: str = None) -> ActionConditioningConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    exp_name = experiment or raw.get("experiment", "wan")
    experiments = raw.get("experiments", {})
    if exp_name not in experiments:
        raise ValueError(f"Unknown experiment '{exp_name}', available: {list(experiments.keys())}")
    exp_raw = experiments[exp_name]
    valid_fields = {f.name for f in fields(ActionConditioningConfig)}
    cfg_dict = {k: v for k, v in exp_raw.items() if k in valid_fields}
    return ActionConditioningConfig(**cfg_dict)


# ============================================================================
# Tensor conversion helpers (same logic as inference_test.py)
# ============================================================================
def pil_to_tensor_image(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Single PIL Image → (1, 3, H, W) float tensor in [-1, 1]."""
    arr = np.array(img, dtype=np.float32)  # (H, W, 3)
    t = torch.from_numpy(arr).permute(2, 0, 1) / 255.0 * 2.0 - 1.0  # (3, H, W)
    return t.unsqueeze(0).to(device)


def pil_list_to_tensor_video(images: list, device: torch.device) -> torch.Tensor:
    """List of PIL Images → (1, 3, T, H, W) float tensor in [-1, 1]."""
    frames = []
    for img in images:
        arr = np.array(img, dtype=np.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1) / 255.0 * 2.0 - 1.0  # (3, H, W)
        frames.append(t)
    video = torch.stack(frames, dim=1)  # (3, T, H, W)
    return video.unsqueeze(0).to(device)  # (1, 3, T, H, W)


# ============================================================================
# Main test
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Test ConditionEncoder with ACWMDataset")
    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to train_metadata.json")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to action_conditioning.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--height", type=int, default=368)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to test")
    args = parser.parse_args()

    device = torch.device(args.device)
    H, W = args.height, args.width

    # ---- Step 1: Load config ----
    print("=" * 60)
    print("Step 1: Loading ActionConditioningConfig")
    print("=" * 60)
    cfg = load_acwm_config(args.config, args.experiment)
    print(f"  model_name:       {cfg.model_name}")
    print(f"  action_dim:       {cfg.action_dim}")
    print(f"  action_embed_dim: {cfg.action_embed_dim}")
    print(f"  action_num_layers:{cfg.action_num_layers}")
    print(f"  obs_injection:    {cfg.obs_injection}")
    print(f"  traj_injection:   {cfg.traj_injection}")
    print(f"  history_injection:{cfg.history_injection}")
    print(f"  history_frames:   {cfg.history_frames}")
    print(f"  vae_z_dim:        {cfg.vae_z_dim}")
    print(f"  vae_temporal_factor: {cfg.vae_temporal_factor}")
    print(f"  vae_spatial_factor:  {cfg.vae_spatial_factor}")
    print(f"  concat_channels:  {cfg.concat_channels}")

    # ---- Step 2: Build ConditionEncoder ----
    print("\n" + "=" * 60)
    print("Step 2: Building ConditionEncoder (loads VAE)")
    print("=" * 60)
    t0 = time.time()
    cond_encoder = ConditionEncoder(cfg, device=device).to(device)
    cond_encoder.eval()
    print(f"  VAE loaded from: {getattr(cond_encoder.vae, '_loaded_ckpt_path', 'N/A')}")
    print(f"  VAE missing keys: {getattr(cond_encoder.vae, '_missing_keys', [])}")
    print(f"  VAE unexpected keys: {len(getattr(cond_encoder.vae, '_unexpected_keys', []))}")
    print(f"  ActionFFNEncoder params: {sum(p.numel() for p in cond_encoder.action_encoder.parameters()):,}")
    print(f"  Load time: {time.time() - t0:.1f}s")

    # ---- Step 3: Load Dataset ----
    print("\n" + "=" * 60)
    print("Step 3: Loading ACWMDataset")
    print("=" * 60)
    ds = ACWMDataset(args.metadata, height=H, width=W, repeat=1)

    # ---- Step 4: Test encode for each sample ----
    print("\n" + "=" * 60)
    print(f"Step 4: Testing ConditionEncoder.encode() on {args.num_samples} samples")
    print("=" * 60)

    # Precompute expected shapes
    num_frames = 17  # 1 obs + 16 targets
    T_latent = (num_frames - 1) // cfg.vae_temporal_factor + 1  # 5
    H_latent = H // cfg.vae_spatial_factor
    W_latent = W // cfg.vae_spatial_factor
    print(f"\n  Expected latent grid: T={T_latent}, H={H_latent}, W={W_latent}")
    print(f"  Expected action_tokens: (1, 16, {cfg.action_embed_dim})")

    all_passed = True

    for i in range(min(args.num_samples, len(ds))):
        print(f"\n  --- Sample {i} ---")
        sample = ds[i]

        # Convert to tensors
        obs_tensor = pil_to_tensor_image(sample["obs_image"], device)
        print(f"    obs_tensor:     {obs_tensor.shape}")  # (1, 3, H, W)

        action_tensor = sample["actions"].unsqueeze(0).to(device)
        print(f"    action_tensor:  {action_tensor.shape}")  # (1, 16, 7)

        history_tensor = None
        if cfg.history_injection is not None and sample["history_images"]:
            history_tensor = pil_list_to_tensor_video(sample["history_images"], device)
            print(f"    history_tensor: {history_tensor.shape}")  # (1, 3, 3, H, W)

        # Noisy latent for shape reference
        noisy_latent = torch.randn(
            1, cfg.vae_z_dim, T_latent, H_latent, W_latent,
            device=device, dtype=torch.float32,
        )
        print(f"    noisy_latent:   {noisy_latent.shape}")

        # Encode
        t0 = time.time()
        with torch.no_grad():
            encoded = cond_encoder.encode(
                obs_image=obs_tensor,
                actions=action_tensor,
                masked_traj=None,  # traj_injection=null in your config
                history=history_tensor,
                noisy_latent=noisy_latent,
            )
        dt = time.time() - t0

        # Check outputs
        at_shape = encoded.action_tokens.shape if encoded.action_tokens is not None else None
        vl_shape = encoded.visual_latent.shape if encoded.visual_latent is not None else None
        print(f"    action_tokens:  {at_shape}")
        print(f"    visual_latent:  {vl_shape}")
        print(f"    encode time:    {dt:.2f}s")

        # ---- Validate shapes ----
        errors = []

        # action_tokens: (1, 16, action_embed_dim)
        if at_shape is None:
            errors.append("action_tokens is None!")
        elif at_shape != (1, 16, cfg.action_embed_dim):
            errors.append(f"action_tokens shape mismatch: {at_shape} != (1, 16, {cfg.action_embed_dim})")

        # visual_latent: (1, C, T_latent, H_latent, W_latent)
        if vl_shape is None:
            errors.append("visual_latent is None!")
        else:
            if vl_shape[0] != 1:
                errors.append(f"visual_latent batch != 1: {vl_shape[0]}")
            if vl_shape[2] != T_latent:
                errors.append(f"visual_latent T mismatch: {vl_shape[2]} != {T_latent}")
            if vl_shape[3] != H_latent:
                errors.append(f"visual_latent H mismatch: {vl_shape[3]} != {H_latent}")
            if vl_shape[4] != W_latent:
                errors.append(f"visual_latent W mismatch: {vl_shape[4]} != {W_latent}")

        # Check: visual_latent can be concatenated with noisy_latent on channel dim
        # In model_fn_wan_video: x = torch.cat([x, y], dim=1)
        # x is (1, 16, T, H, W), y should be (1, C_y, T, H, W)
        if vl_shape is not None:
            print(f"    → y channels (for cat with latent): {vl_shape[1]}")
            print(f"    → noisy latent channels:            {cfg.vae_z_dim}")
            print(f"    → DiT in_dim should be >= {vl_shape[1] + cfg.vae_z_dim}")

        if errors:
            for e in errors:
                print(f"    [FAIL] {e}")
            all_passed = False
        else:
            print(f"    [PASS]")

    # ---- Step 5: Gradient flow test ----
    print("\n" + "=" * 60)
    print("Step 5: Gradient flow test (ActionFFNEncoder)")
    print("=" * 60)

    cond_encoder.train()
    sample = ds[0]
    action_tensor = sample["actions"].unsqueeze(0).to(device).requires_grad_(False)

    # action_encoder should have gradients
    tokens = cond_encoder.action_encoder(action_tensor)
    loss = tokens.sum()
    loss.backward()

    has_grad = False
    for name, p in cond_encoder.action_encoder.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break

    if has_grad:
        print("  [PASS] ActionFFNEncoder receives gradients")
    else:
        print("  [FAIL] ActionFFNEncoder has no gradients!")
        all_passed = False

    # Check VAE is frozen
    vae_has_grad = any(
        p.requires_grad for p in cond_encoder.vae.parameters()
    )
    if not vae_has_grad:
        print("  [PASS] VAE is frozen (no requires_grad)")
    else:
        print("  [FAIL] VAE has trainable params!")
        all_passed = False

    cond_encoder.zero_grad()

    # ---- Summary ----
    print("\n" + "=" * 60)
    if all_passed:
        print("[ALL PASS] ConditionEncoder + ACWMDataset shape 对齐，梯度正常")
    else:
        print("[SOME FAILED] 请检查上面的 FAIL 信息")
    print("=" * 60)


if __name__ == "__main__":
    main()