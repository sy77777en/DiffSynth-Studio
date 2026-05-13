"""
ACWM v2 — Action-Conditioned World Model training on Wan DiT.

Conditions:
  - obs_image:   observation frame (first frame, VAE+CLIP encoded by pipeline)
  - masked_traj: rendered trajectory images (future frames with masked EEF path)
  - actions:     delta action sequence (encoded by ActionFFNEncoder)

No history frames. No custom ConditionEncoder.
Pipeline handles VAE(obs)->mask->y and CLIP(obs)->clip_feature natively.
ActionFFNEncoder is the only new trainable module.

Trainable:
  - DiT LoRA (q,k,v,o,ffn.0,ffn.2)
  - ActionFFNEncoder (full params)
Frozen:
  - VAE, T5 text encoder, CLIP image encoder (if loaded)

IMPORTANT: requires one change in diffsynth/pipelines/wan_video.py
  In model_fn_wan_video(), change the action_tokens block to REPLACE context:

    if preencoded_action_tokens is not None:
        action_tokens = preencoded_action_tokens.to(dtype=context.dtype, device=context.device)
        ctx_dim = context.shape[-1]
        if action_tokens.shape[-1] < ctx_dim:
            pad = torch.zeros(
                (*action_tokens.shape[:-1], ctx_dim - action_tokens.shape[-1]),
                dtype=action_tokens.dtype, device=action_tokens.device,
            )
            action_tokens = torch.cat([action_tokens, pad], dim=-1)
        elif action_tokens.shape[-1] > ctx_dim:
            action_tokens = action_tokens[..., :ctx_dim]
        context = action_tokens   # <-- REPLACE, not concat

Example launch:
  accelerate launch examples/wanvideo/model_training/train_acwm_v2.py \\
    --dataset_base_path . \\
    --dataset_metadata_path ./train_metadata.json \\
    --height 368 --width 640 --num_frames 17 \\
    --model_id_with_origin_paths \\
      "Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model-00001-of-00002.safetensors,high_noise_model/diffusion_pytorch_model-00002-of-00002.safetensors;Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth;Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \\
    --output_path ./output/acwm_v2 \\
    --trainable_models dit \\
    --lora_base_model dit \\
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \\
    --lora_rank 32 \\
    --extra_inputs input_image \\
    --learning_rate 1e-4 \\
    --task sft \\
    --action_dim 7 \\
    --action_embed_dim 1024
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from diffsynth.models.wan_video_dit import TemporalAttentionAdapter, FramewiseCrossAttention

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_MT_DIR = os.path.dirname(os.path.abspath(__file__))
if _MT_DIR not in sys.path:
    sys.path.insert(0, _MT_DIR)

import accelerate
from diffsynth.diffusion import ModelLogger
from diffsynth.diffusion.runner import (
    initialize_deepspeed_gradient_checkpointing,
    launch_data_process_task,
)

from train import WanTrainingModule, wan_parser
from acwm_dataset import ACWMDataset


# ============================================================================
# ActionFFNEncoder
# ============================================================================


class ActionFFNEncoder(nn.Module):
    """MLP encoder: (B, T, action_dim) -> (B, T, embed_dim).

    Output tokens replace the text context in DiT cross-attention.
    """

    def __init__(self, action_dim: int, embed_dim: int, num_layers: int = 2, max_timesteps: int = 16):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(action_dim, embed_dim), nn.GELU()]
        for _ in range(max(0, num_layers - 2)):
            layers += [nn.Linear(embed_dim, embed_dim), nn.GELU()]
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)

        pe = self._sinusoidal_pe(max_timesteps, embed_dim)
        self.temporal_pe = nn.Parameter(pe)  # (max_timesteps, embed_dim)

    @staticmethod
    def _sinusoidal_pe(length: int, dim: int) -> torch.Tensor:
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe
  
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        # return self.norm(self.mlp(actions))
        x = self.mlp(actions)
        T = actions.shape[1]
        x = x + self.temporal_pe[:T].to(dtype=x.dtype, device=x.device)
        return self.norm(x)


# ============================================================================
# Training Module
# ============================================================================


class ACWMv2TrainingModule(WanTrainingModule):
    """Extends WanTrainingModule for action-conditioned world model training.

    New inputs beyond standard Wan I2V:
      - actions -> ActionFFNEncoder -> action_tokens (cross-attn context)
      - masked_traj -> VAE encode -> concat with obs latent as visual condition

    Pipeline natively handles:
      - obs_image -> VAE encode -> mask + latent -> y (via ImageEmbedderVAE)
      - obs_image -> CLIP encode -> clip_feature (via ImageEmbedderCLIP, if loaded)
      - video (obs + targets) -> VAE encode -> input_latents (training GT)

    The masked_traj visual condition is encoded here and injected as
    preencoded_visual_latent, which ImageEmbedderVAE picks up via
    skip_condition_vae_encode=True.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_embed_dim: int = 1024,
        action_num_layers: int = 3,
        use_masked_traj: bool = True,
        enable_temporal_adapter: bool = False,
        enable_framewise_cross_attn: bool = False,   # add this
        *args,
        extra_inputs: str | None = None,
        **kwargs,
    ):
        # input_image triggers CLIP encoding of obs (if CLIP is loaded)
        if extra_inputs is None:
            extra_inputs = "input_image"
        elif "input_image" not in extra_inputs:
            extra_inputs = f"input_image,{extra_inputs}"

        super().__init__(*args, extra_inputs=extra_inputs, **kwargs)

        if enable_temporal_adapter:
            temporal_adapter_layers = [12, 16, 20]
            for i, block in enumerate(self.pipe.dit.blocks):
                if i in temporal_adapter_layers:
                    block.use_temporal_adapter = True
                    block.temporal_adapter = TemporalAttentionAdapter(
                        dim=self.pipe.dit.dim,
                        num_heads=block.num_heads,
                    )
                    for p in block.temporal_adapter.parameters():
                        p.requires_grad = True

        self.use_masked_traj = use_masked_traj

        # Trainable action encoder
        self.action_encoder = ActionFFNEncoder(
            action_dim=action_dim,
            embed_dim=action_embed_dim,
            num_layers=action_num_layers,
        )
        self.action_encoder.train()
        for p in self.action_encoder.parameters():
            p.requires_grad = True

        if not enable_temporal_adapter:
            for block in self.pipe.dit.blocks:
                block.use_temporal_adapter = False
        else:
            for block in self.pipe.dit.blocks:
                if block.use_temporal_adapter:
                    for p in block.temporal_adapter.parameters():
                        p.requires_grad = True

        # Attach framewise cross-attention to each DiT block
        if enable_framewise_cross_attn:
            for block in self.pipe.dit.blocks:
                block.framewise_cross_attn = FramewiseCrossAttention(
                    dim=self.pipe.dit.dim,
                    num_heads=block.num_heads,
                ).to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
                for p in block.framewise_cross_attn.parameters():
                    p.requires_grad = True
                # freeze original cross_attn (loaded but unused)
                for p in block.cross_attn.parameters():
                    p.requires_grad = False

    def _encode_visual_condition(self, data: dict, inputs_shared: dict) -> dict:
        """Encode obs + masked_traj into the visual condition (y).

        Builds a combined visual latent:
          y = [4ch mask | 16ch VAE latent]  shape (1, 20, T_lat, H', W')

        where the VAE latent is:
          - history frames (prev 4 frames)
          - obs frame at t=0 (from obs_image)
          - masked_traj frames for t>0 (future trajectory visualization)
          - zero-padded to match noisy_latent temporal length

        This replaces the default ImageEmbedderVAE behavior by setting
        skip_condition_vae_encode=True and providing preencoded_visual_latent.
        """
        if not self.use_masked_traj or "masked_traj" not in data:
            # No masked_traj — let pipeline's ImageEmbedderVAE handle obs normally
            return inputs_shared

        device = self.pipe.device
        dtype = self.pipe.torch_dtype

        height = inputs_shared["height"]
        width = inputs_shared["width"]
        num_frames = inputs_shared["num_frames"]

        # Build the condition video: [obs] + masked_traj frames
        obs_img = data["obs_image"]
        traj_imgs = data["masked_traj"]

        if "history_frames" in data and data["history_frames"] is not None:
            history_frames = data["history_frames"]
            if len(history_frames) == 0:
                history_frames = [obs_img] * 4
            elif len(history_frames) > 4:
                history_frames = history_frames[-4:]
            elif 0 < len(history_frames) < 4:
                history_frames = history_frames + [history_frames[-1]] * (4 - len(history_frames))
        else:
            history_frames = [obs_img] * 4

        history_tensors = []
        for img in history_frames:
            if img.size != (width, height):
                img = img.resize((width, height), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32)
            t = torch.from_numpy(arr).permute(2, 0, 1) / 255.0 * 2.0 - 1.0
            history_tensors.append(t)
        history_video = torch.stack(history_tensors, dim=1)  # (3, 4, H, W)

        # Preprocess to tensor: list of PIL -> (3, T, H, W) in [-1, 1]
        all_frames = [obs_img] + traj_imgs
        frame_tensors = []
        for img in all_frames:
            if img.size != (width, height):
                img = img.resize((width, height), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32)
            t = torch.from_numpy(arr).permute(2, 0, 1) / 255.0 * 2.0 - 1.0
            frame_tensors.append(t)
        # (3, T_cond, H, W)
        cond_video = torch.stack(frame_tensors, dim=1)

        # Pad or truncate to num_frames
        T_cond = cond_video.shape[1]
        if T_cond < num_frames:
            pad = torch.zeros(3, num_frames - T_cond, height, width)
            cond_video = torch.cat([cond_video, pad], dim=1)
        elif T_cond > num_frames:
            cond_video = cond_video[:, :num_frames]

        # VAE encode the condition video
        self.pipe.load_models_to_device(["vae"])
        history_video = history_video.to(dtype=dtype, device=device)
        cond_video = cond_video.to(dtype=dtype, device=device)
        # vae.encode expects list of (3, 4, H, W)
        history_latent = self.pipe.vae.encode(
          [history_video], device=device,
        )[0]
        # vae.encode expects list of (3, T, H, W)
        y = self.pipe.vae.encode(
            [cond_video], device=device,
        )[0]  # (C, T_lat, H', W')
        y = torch.cat([history_latent, y], dim=1)
        y = y.to(dtype=dtype, device=device)

        # Build 4-channel mask: 1 where we have real condition, 0 elsewhere
        # The obs frame (t=0) is always a real condition
        T_lat = y.shape[1]
        H_lat, W_lat = y.shape[2], y.shape[3]

        history_msk = torch.ones(4, 1, H_lat, W_lat, device=device)
        obs_future_msk = torch.zeros(1, num_frames, H_lat, W_lat, device=device)
        obs_future_msk[:, :1 + len(traj_imgs)] = 1

        obs_future_msk = torch.cat(
            [torch.repeat_interleave(obs_future_msk[:, 0:1], repeats=4, dim=1), obs_future_msk[:, 1:]],
            dim=1,
        )
        obs_future_msk = obs_future_msk.view(1, obs_future_msk.shape[1] // 4, 4, H_lat, W_lat)
        obs_future_msk = obs_future_msk.transpose(1, 2)[0]  # (4, T_lat_obs_future, H', W')

        # Concat along time
        msk = torch.cat([history_msk, obs_future_msk], dim=1)  # (4, T_lat, H', W')

      
        # n_real_frames = 4 + 1 + len(traj_imgs)  # history + obs + traj frames
        # # How many latent frames correspond to real condition frames
        # n_real_lat = (n_real_frames - 1) // 4 + 1
        # n_real_lat = min(n_real_lat, T_lat)

        # msk = torch.zeros(1, 4 + num_frames, H_lat, W_lat, device=device)
        # msk[:, :n_real_frames] = 1  # mark real frames

        # # Reshape mask to latent temporal resolution (same as ImageEmbedderVAE)
        # msk = torch.cat(
        #     [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
        #     dim=1,
        # )
        # msk = msk.view(1, msk.shape[1] // 4, 4, H_lat, W_lat)
        # msk = msk.transpose(1, 2)[0]  # (4, T_lat, H', W')

        # y = [mask (4ch) | latent (16ch)]
        y = torch.cat([msk, y])  # (20, T_lat, H', W')
        y = y.unsqueeze(0).to(dtype=dtype, device=device)  # (1, 20, T_lat, H', W')

        inputs_shared["preencoded_visual_latent"] = y
        inputs_shared["skip_condition_vae_encode"] = True

        return inputs_shared

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(
            inputs, self.pipe.device, self.pipe.torch_dtype
        )
        inputs_shared, inputs_posi, inputs_nega = inputs

        # --- 1. Encode actions ---
        device = self.pipe.device
        dtype = next(self.action_encoder.parameters()).dtype
        actions = data["actions"].unsqueeze(0).to(device=device, dtype=dtype)
        action_tokens = self.action_encoder(actions)  # (1, T, embed_dim)
        inputs_shared["preencoded_action_tokens"] = action_tokens

        # --- 2. Encode visual condition (obs + masked_traj) ---
        has_traj = "masked_traj" in data and data["masked_traj"] is not None
        if not hasattr(self, '_logged_traj'):
            print(f"[ACWM] use_masked_traj={self.use_masked_traj}, "
                  f"data has masked_traj={has_traj}, "
                  f"n_traj_frames={len(data['masked_traj']) if has_traj else 0}")
            self._logged_traj = True
        inputs_shared = self._encode_visual_condition(data, inputs_shared)

        # --- 3. Run pipeline ---
        # Remaining units handle:
        #   PromptEmbedder:     T5("") -> context (replaced by action_tokens in model_fn)
        #   InputVideoEmbedder: VAE(video) -> input_latents + noise
        #   ImageEmbedderVAE:   skip (we provided preencoded_visual_latent)
        #   ImageEmbedderCLIP:  CLIP(obs) -> clip_feature (if loaded)
        #   model_fn_wan_video: context = action_tokens, x = cat([noisy, y]) -> DiT
        #   FlowMatchSFTLoss:   MSE(predicted, target)

        inputs = (inputs_shared, inputs_posi, inputs_nega)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


# ============================================================================
# Training Launcher
# ============================================================================


def launch_acwm_training_task(
    accelerator: accelerate.Accelerator,
    dataset: torch.utils.data.Dataset,
    model: ACWMv2TrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int | None = None,
    num_epochs: int = 1,
    args=None,
):
    from tqdm import tqdm

    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    action_lr = getattr(args, "action_encoder_lr", None) if args is not None else None

    # Optional separate LR for action encoder
    if action_lr is not None:
        enc_params = list(model.action_encoder.parameters())
        enc_ids = {id(p) for p in enc_params}
        other_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in enc_ids
        ]
        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": learning_rate, "weight_decay": weight_decay},
            {"params": enc_params, "lr": action_lr, "weight_decay": weight_decay},
        ])
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda x: x[0],
        num_workers=num_workers,
    )

    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler,
    )
    initialize_deepspeed_gradient_checkpointing(accelerator)

    # Logging
    log_every = getattr(args, "log_every_n_steps", 50) if args is not None else 50
    log_csv_path = None
    if getattr(args, "log_loss_to_csv", False) and getattr(args, "output_path", None):
        log_csv_path = os.path.join(args.output_path, "training_loss.csv")
        if accelerator.is_main_process:
            os.makedirs(args.output_path, exist_ok=True)
            if not os.path.isfile(log_csv_path):
                with open(log_csv_path, "w", newline="") as f:
                    csv.writer(f).writerow(["global_step", "epoch", "loss"])

    load_from_cache = getattr(dataset, "load_from_cache", False)
    global_step = 0

    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader, desc=f"epoch {epoch_id}"):
            with accelerator.accumulate(model):
                loss = model({}, inputs=data) if load_from_cache else model(data)
                accelerator.backward(loss)
              
                grad_norm = None
                if accelerator.sync_gradients:
                    total_grad_norm = accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=1e9,   # effectively no clipping
                    )
              
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)

            global_step += 1
            if log_every > 0 and global_step % log_every == 0:
                lv = loss.detach().float()
                if accelerator.num_processes > 1:
                    lv = (
                        accelerator.reduce(lv, reduction="mean")
                        if hasattr(accelerator, "reduce")
                        else accelerator.gather(lv.unsqueeze(0)).mean()
                    )
                loss_val = (
                    lv.item() if isinstance(lv, torch.Tensor) else float(lv)
                )
                if accelerator.is_main_process:
                    msg = (
                        f"[train] epoch={epoch_id} step={global_step} "
                        f"loss={loss_val:.6f}"
                    )
                    if grad_norm is not None:
                        msg += f" grad_norm={float(grad_norm):.6f}"
                    tqdm.write(msg)
                    # tqdm.write(
                    #     f"[train] epoch={epoch_id} step={global_step} "
                    #     f"loss={loss_val:.6f}"
                    # )
                    if log_csv_path is not None:
                        with open(log_csv_path, "a", newline="") as f:
                            csv.writer(f).writerow(
                                [global_step, epoch_id, f"{loss_val:.8f}"]
                            )

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)


# ============================================================================
# CLI
# ============================================================================


def acwm_v2_parser() -> argparse.ArgumentParser:
    parser = wan_parser()
    parser.add_argument(
        "--action_dim", type=int, default=7,
        help="Dimension of each action vector.",
    )
    parser.add_argument(
        "--action_embed_dim", type=int, default=1024,
        help="Action encoder output dim (should match DiT cross-attn dim).",
    )
    parser.add_argument(
        "--action_num_layers", type=int, default=2,
        help="Number of MLP layers in action encoder.",
    )
    parser.add_argument(
        "--action_encoder_lr", type=float, default=None,
        help="Separate LR for ActionFFNEncoder. Default: same as --learning_rate.",
    )
    parser.add_argument(
        "--no_masked_traj", action="store_true",
        help="Disable masked trajectory conditioning (obs-only visual condition).",
    )
    parser.add_argument(
        "--log_every_n_steps", type=int, default=50,
        help="Print loss every N steps (0=disable).",
    )
    parser.add_argument(
        "--log_loss_to_csv", action="store_true",
        help="Append loss to --output_path/training_loss.csv.",
    )
    parser.add_argument(
        "--enable_temporal_adapter", action="store_true", default=False,
        help="Enable temporal attention adapter in DiT blocks 12/16/20.",
    )
    parser.add_argument(
        "--enable_framewise_cross_attn",
        action="store_true",
        default=False,
        help="Enable framewise cross-attention in DiT blocks.",
    )
    # Relax dataset_base_path
    for action in parser._actions:
        if "--dataset_base_path" in action.option_strings:
            action.required = False
            if action.default is None or action.default == "":
                action.default = "."
            break
    return parser


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = acwm_v2_parser()
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters,
            ),
        ],
    )

    if args.height is None or args.width is None:
        raise SystemExit("--height and --width are required (e.g. 368 640).")
    if args.dataset_metadata_path is None:
        raise SystemExit("--dataset_metadata_path is required.")

    dataset = ACWMDataset(
        args.dataset_metadata_path,
        height=args.height,
        width=args.width,
        repeat=args.dataset_repeat,
    )

    model = ACWMv2TrainingModule(
        action_dim=args.action_dim,
        action_embed_dim=args.action_embed_dim,
        action_num_layers=args.action_num_layers,
        use_masked_traj=not args.no_masked_traj,
        enable_temporal_adapter=args.enable_temporal_adapter,
        enable_framewise_cross_attn=args.enable_framewise_cross_attn,
        # --- Parent class args ---
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device=(
            "cpu"
            if getattr(args, "initialize_model_on_cpu", False)
            else accelerator.device
        ),
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_acwm_training_task,
        "sft:train": launch_acwm_training_task,
        "direct_distill": launch_acwm_training_task,
        "direct_distill:train": launch_acwm_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
