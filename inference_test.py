"""
Action-Conditioned World Model Inference.

Input JSON format (list of items):
[
    {
        "action": [[a1, a2, ..., a8], ...],   # shape [T_action, 8]
        "start_id": 173,
        "image_folder": "/path/to/folder"
        # folder contains: frame_000173.png, frame_000174.png, ...
        # optionally:      masked_000173.png, masked_000174.png, ...
    },
    ...
]

Usage:
    python inference_acwm.py \
        --config configs/action_conditioning.yaml \
        --data_json data/inference_items.json \
        --output outputs/run1 \
        --num_inference_steps 30 \
        --seed 0
"""

import os
import sys
import json
import math
import argparse
from dataclasses import fields
from typing import Optional

import numpy as np
import torch
import yaml
from PIL import Image

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.action_conditioning.config import ActionConditioningConfig
from diffsynth.models.action_conditioning.encoder import ConditionEncoder


# ============================================================================
# 常量
# ============================================================================
CHUNK_SIZE = 16          # 每次生成的 action 帧数
NUM_FRAMES_PER_CALL = 17 # pipeline 每次生成的视频帧数 (17-1)//4+1=5 latent frames


# ============================================================================
# 辅助函数
# ============================================================================
def load_frame(folder: str, frame_id: int) -> np.ndarray:
    """加载单张 RGB 帧，返回 uint8 HWC array。"""
    path = os.path.join(folder, f"frame_{frame_id:06d}.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Frame not found: {path}")
    return np.array(Image.open(path).convert("RGB"))


def load_masked_traj_frames(folder: str, start_id: int, count: int, H: int, W: int):
    """尝试加载 masked trajectory 帧序列。如果任何一帧缺失，返回 None。"""
    frames = []
    for i in range(count):
        path = os.path.join(folder, f"masked_{start_id + i:06d}.png")
        if not os.path.exists(path):
            return None
        img = np.array(Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS))
        frames.append(img)
    return frames


def np_to_tensor_image(img_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """HWC uint8 numpy → (1, 3, H, W) float tensor in [-1, 1]."""
    t = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return t.unsqueeze(0).to(device)


def np_list_to_tensor_video(frames: list, device: torch.device) -> torch.Tensor:
    """List of HWC uint8 numpy → (1, 3, T, H, W) float tensor in [-1, 1]."""
    arr = np.stack(frames, axis=0)  # (T, H, W, 3)
    t = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0  # (T, 3, H, W)
    return t.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (1, 3, T, H, W)


def pad_actions(actions: np.ndarray, target_dim: int) -> np.ndarray:
    """把 (T, D_in) 的 action zero-pad 到 (T, target_dim)。"""
    T, D_in = actions.shape
    if D_in >= target_dim:
        return actions[:, :target_dim]
    padded = np.zeros((T, target_dim), dtype=actions.dtype)
    padded[:, :D_in] = actions
    return padded


# ============================================================================
# 构建 Pipeline 和 ConditionEncoder
# ============================================================================
def load_yaml_config(config_path: str, experiment: str = None):
    """加载 YAML config，返回 ActionConditioningConfig 和原始 exp dict。"""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    exp_name = experiment or raw.get("experiment", "wan")
    experiments = raw.get("experiments", {})
    if exp_name not in experiments:
        raise ValueError(f"Unknown experiment '{exp_name}', available: {list(experiments.keys())}")

    exp_raw = experiments[exp_name]
    valid_fields = {f.name for f in fields(ActionConditioningConfig)}
    exp_cfg_dict = {k: v for k, v in exp_raw.items() if k in valid_fields}
    cfg = ActionConditioningConfig(**exp_cfg_dict)
    return cfg, exp_raw


def build_condition_encoder(cfg: ActionConditioningConfig, device: torch.device) -> ConditionEncoder:
    """构建 ConditionEncoder（包含冻结的 VAE + 可训练的 action encoder）。"""
    cond_encoder = ConditionEncoder(cfg, device=device).to(device)
    cond_encoder.eval()
    print(f"[CondEncoder] VAE loaded from: {getattr(cond_encoder.vae, '_loaded_ckpt_path', 'N/A')}")
    print(f"[CondEncoder] action_dim={cfg.action_dim}, embed_dim={cfg.action_embed_dim}")
    return cond_encoder


def build_pipeline(model_dir: str, device: str = "cuda") -> WanVideoPipeline:
    """构建 WanVideoPipeline，自动检测模型文件。"""

    # DiT shards
    def find_dit_shards(subdir):
        d = os.path.join(model_dir, subdir)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"DiT directory not found: {d}")
        shards = sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith(".safetensors")])
        if not shards:
            raise FileNotFoundError(f"No .safetensors files in {d}")
        return shards

    # VAE
    vae_path = None
    for name in ["Wan2.1_VAE.pth", "Wan2.2_VAE.pth", "Wan2.1_VAE.safetensors", "Wan2.2_VAE.safetensors"]:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            vae_path = p
            break
    assert vae_path is not None, f"VAE not found in {model_dir}"

    # T5
    t5_path = None
    for name in ["models_t5_umt5-xxl-enc-bf16.pth", "models_t5_umt5-xxl-enc-bf16.safetensors"]:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            t5_path = p
            break
    assert t5_path is not None, f"T5 encoder not found in {model_dir}"

    # Tokenizer
    tokenizer_path = None
    for name in ["google/umt5-xxl", "tokenizer"]:
        p = os.path.join(model_dir, name)
        if os.path.isdir(p):
            tokenizer_path = p
            break
    assert tokenizer_path is not None, f"Tokenizer not found in {model_dir}"

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=find_dit_shards("high_noise_model"), offload_device="cpu"),
            ModelConfig(path=find_dit_shards("low_noise_model"), offload_device="cpu"),
            ModelConfig(path=t5_path, offload_device="cpu"),
            ModelConfig(path=vae_path, offload_device="cpu"),
        ],
        tokenizer_config=ModelConfig(path=tokenizer_path),
    )
    return pipe


# ============================================================================
# 单个 item 的推理
# ============================================================================
def run_inference_item(
    item: dict,
    cond_encoder: ConditionEncoder,
    pipe: WanVideoPipeline,
    cfg: ActionConditioningConfig,
    device: torch.device,
    num_inference_steps: int = 50,
    seed: int = 0,
    H_gen: int = 480,
    W_gen: int = 832,
):
    """
    对单个 JSON item 做推理，返回生成的 PIL Image 列表。

    流程：
    1. 把 actions 分成 CHUNK_SIZE=16 一组的 chunks
    2. 每个 chunk:
       a. 准备 obs_image, actions, (masked_traj), (history)
       b. ConditionEncoder.encode() → visual_latent + action_tokens
       c. pipe() → 17 帧视频
       d. 第一帧替换为精确的 obs_image
       e. chunk 0 保留全部 17 帧，后续 chunk 去掉第一帧（避免跟上一 chunk 末帧重复）
    3. 最后按 action 总长度裁剪
    """
    folder = item["image_folder"]
    start_id = item["start_id"]
    raw_actions = np.array(item["action"], dtype=np.float32)  # (T_total, D_in)
    T_total = raw_actions.shape[0]

    # Pad action dim
    actions_padded = pad_actions(raw_actions, cfg.action_dim)  # (T_total, action_dim)

    # 计算 chunk 数量，补齐到 CHUNK_SIZE 的倍数
    n_chunks = math.ceil(T_total / CHUNK_SIZE)
    total_padded = n_chunks * CHUNK_SIZE
    if total_padded > T_total:
        pad_len = total_padded - T_total
        actions_padded = np.concatenate(
            [actions_padded, np.tile(actions_padded[-1:], (pad_len, 1))], axis=0
        )

    print(f"  T_total={T_total}, n_chunks={n_chunks}, padded_to={total_padded}")

    # 加载第一帧 observation
    obs_frame_np = load_frame(folder, start_id)
    obs_pil = Image.fromarray(obs_frame_np).resize((W_gen, H_gen), Image.LANCZOS)

    # latent 维度参考
    T_latent = (NUM_FRAMES_PER_CALL - 1) // cfg.vae_temporal_factor + 1  # 5
    H_latent = H_gen // cfg.vae_spatial_factor
    W_latent = W_gen // cfg.vae_spatial_factor

    all_generated_frames = []

    for chunk_idx in range(n_chunks):
        c_start = chunk_idx * CHUNK_SIZE
        c_end = c_start + CHUNK_SIZE

        # ---- 准备当前 chunk 的输入 ----

        # Action tensor: (1, 16, action_dim)
        chunk_actions = actions_padded[c_start:c_end]
        action_tensor = torch.from_numpy(chunk_actions).float().unsqueeze(0).to(device)

        # Observation image
        if chunk_idx == 0:
            obs_np = np.array(obs_pil)
        else:
            # 用上一个 chunk 最后一帧作为新的 observation
            last_frame = all_generated_frames[-1]
            obs_np = np.array(last_frame) if isinstance(last_frame, Image.Image) else last_frame

        obs_pil_current = Image.fromarray(obs_np).resize((W_gen, H_gen), Image.LANCZOS)
        obs_tensor = np_to_tensor_image(np.array(obs_pil_current), device)  # (1, 3, H, W)

        # History（从已生成的帧中均匀采样 K 帧）
        history_tensor = None
        if chunk_idx > 0 and cfg.history_injection is not None:
            n_total = len(all_generated_frames)
            n_hist = min(cfg.history_frames, n_total)
            # 均匀采样 indices，覆盖整个历史跨度
            if n_hist == 1:
                hist_indices = [n_total - 1]
            else:
                hist_indices = [n_total * i // (n_hist - 1) for i in range(n_hist)]
                hist_indices[-1] = min(hist_indices[-1], n_total - 1)
            hist_frames = []
            for idx in hist_indices:
                f = all_generated_frames[idx]
                f_np = np.array(f) if isinstance(f, Image.Image) else f
                f_np = np.array(Image.fromarray(f_np).resize((W_gen, H_gen), Image.LANCZOS))
                hist_frames.append(f_np)
            history_tensor = np_list_to_tensor_video(hist_frames, device)  # (1, 3, K, H, W)

        # Masked trajectory（可选）
        masked_traj_tensor = None
        if cfg.traj_injection is not None:
            traj_frames = load_masked_traj_frames(
                folder, start_id + c_start, CHUNK_SIZE, H_gen, W_gen
            )
            if traj_frames is not None:
                masked_traj_tensor = np_list_to_tensor_video(traj_frames, device)  # (1, 3, 16, H, W)

        # Noisy latent（shape 参考，用于 ConditionEncoder 内部对齐 temporal 维度）
        noisy_latent = torch.randn(
            1, cfg.vae_z_dim, T_latent, H_latent, W_latent,
            device=device, dtype=torch.float32,
        )

        # ---- ConditionEncoder 编码 ----
        with torch.no_grad():
            encoded = cond_encoder.encode(
                obs_image=obs_tensor,
                actions=action_tensor,
                masked_traj=masked_traj_tensor,
                history=history_tensor,
                noisy_latent=noisy_latent,
            )

        print(f"  chunk {chunk_idx}/{n_chunks}: "
              f"action_tokens={encoded.action_tokens.shape if encoded.action_tokens is not None else None}, "
              f"visual_latent={encoded.visual_latent.shape if encoded.visual_latent is not None else None}")

        # ---- Pipeline 生成 ----
        video_frames = pipe(
            prompt="",
            negative_prompt="",
            input_image=obs_pil_current,       # 只用于 CLIP image embedding
            height=H_gen,
            width=W_gen,
            num_frames=NUM_FRAMES_PER_CALL,    # 生成 17 帧
            num_inference_steps=num_inference_steps,
            seed=seed + chunk_idx,
            tiled=True,
            preencoded_visual_latent=encoded.visual_latent,   # VAE 条件（已编码）
            preencoded_action_tokens=encoded.action_tokens,   # 动作条件
            skip_condition_vae_encode=True,                   # 跳过 pipeline 内部 VAE 编码
        )
        # video_frames: list of PIL Image, 长度 = NUM_FRAMES_PER_CALL = 17

        # ---- 后处理 ----
        # 第一帧替换为精确的 observation image
        video_frames[0] = obs_pil_current

        if chunk_idx == 0:
            # 第一个 chunk：保留全部 17 帧（1 obs + 16 generated）
            all_generated_frames.extend(video_frames)
        else:
            # 后续 chunk：去掉第一帧（跟上一 chunk 最后一帧重复），保留 16 帧
            all_generated_frames.extend(video_frames[1:])

        torch.cuda.empty_cache()

    # 按实际 action 长度裁剪
    # chunk 0 贡献 17 帧，后续每个 chunk 贡献 16 帧
    # 总帧数应该是 T_total + 1（T_total 个 action 帧 + 1 个初始 obs 帧）
    all_generated_frames = all_generated_frames[:T_total + 1]
    return all_generated_frames


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="ACWM Inference")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to action_conditioning.yaml")
    parser.add_argument("--data_json", type=str, required=True,
                        help="Path to JSON file with inference items")
    parser.add_argument("--output", type=str, default="outputs/acwm_inference",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--experiment", type=str, default=None,
                        help="Override experiment name in YAML")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N items (for testing)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- 加载 config ----
    cfg, exp_raw = load_yaml_config(args.config, args.experiment)
    print(f"[Config] model_name={cfg.model_name}, action_dim={cfg.action_dim}")
    print(f"[Config] obs_injection={cfg.obs_injection}, traj_injection={cfg.traj_injection}, "
          f"history_injection={cfg.history_injection}, history_frames={cfg.history_frames}")

    # ---- 构建 ConditionEncoder ----
    print("[Init] Building ConditionEncoder...")
    cond_encoder = build_condition_encoder(cfg, device)

    # ---- 构建 Pipeline ----
    model_dir = exp_raw.get("model_dir", exp_raw.get("model_root"))
    if model_dir is None:
        raise ValueError("Please set model_dir or model_root in YAML config")
    print(f"[Init] Building WanVideoPipeline from {model_dir}...")
    pipe = build_pipeline(model_dir, device=args.device)
    print("[Init] Pipeline ready.")

    # ---- 加载数据 ----
    with open(args.data_json, "r") as f:
        items = json.load(f)
    if isinstance(items, dict):
        items = [items]
    if args.limit is not None:
        items = items[:args.limit]
    print(f"[Data] {len(items)} inference items loaded.")

    # ---- 推理 ----
    os.makedirs(args.output, exist_ok=True)

    for item_idx, item in enumerate(items):
        print(f"\n=== Item {item_idx}/{len(items)} ===")
        print(f"  folder: {item['image_folder']}")
        print(f"  start_id: {item['start_id']}, action_len: {len(item['action'])}")

        try:
            generated_frames = run_inference_item(
                item=item,
                cond_encoder=cond_encoder,
                pipe=pipe,
                cfg=cfg,
                device=device,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                H_gen=args.height,
                W_gen=args.width,
            )

            # 保存视频
            out_name = f"item_{item_idx:04d}_start{item['start_id']}"
            out_dir = os.path.join(args.output, out_name)
            os.makedirs(out_dir, exist_ok=True)

            # mp4
            mp4_path = os.path.join(out_dir, "output.mp4")
            save_video(generated_frames, mp4_path, fps=8, quality=5)
            print(f"  Saved video: {mp4_path} ({len(generated_frames)} frames)")

            # 单帧 PNG（方便 debug）
            for fi, frame in enumerate(generated_frames):
                if isinstance(frame, Image.Image):
                    frame.save(os.path.join(out_dir, f"frame_{fi:06d}.png"))
                else:
                    Image.fromarray(frame).save(os.path.join(out_dir, f"frame_{fi:06d}.png"))

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n[Done] All items processed. Results in {args.output}/")


if __name__ == "__main__":
    main()