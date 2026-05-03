"""
ACWM (Action-Conditioned World Model) — LoRA on DiT + full training on ActionFFNEncoder.

Prerequisites:
  - Same repo layout as other wanvideo scripts; run from DiffSynth-Studio root or set PYTHONPATH.
  - `pip install -e .` recommended.

Example:
  accelerate launch examples/wanvideo/model_training/train_acwm.py \\
    --dataset_base_path . \\
    --dataset_metadata_path ./train_metadata_100.json \\
    --height 368 --width 640 --num_frames 17 \\
    --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,..." \\
    --acwm_config ./configs/action_conditioning.yaml \\
    --output_path ./models/train/acwm_high_noise \\
    --lora_base_model dit \\
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \\
    --extra_inputs input_image \\
    --learning_rate 1e-4 \\
    --max_timestep_boundary 0.358 --min_timestep_boundary 0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml
from dataclasses import fields
from PIL import Image

# Repo root (parent of `examples/`)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import accelerate
from diffsynth.diffusion import ModelLogger
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing, launch_data_process_task
from diffsynth.models.action_conditioning.config import ActionConditioningConfig
from diffsynth.models.action_conditioning.encoder import ConditionEncoder

from train import WanTrainingModule, wan_parser

_MT_DIR = os.path.dirname(os.path.abspath(__file__))
if _MT_DIR not in sys.path:
    sys.path.insert(0, _MT_DIR)


# ---------------------------------------------------------------------------
# Config / tensor helpers (aligned with test_cond_encoder.py)
# ---------------------------------------------------------------------------


def load_acwm_config(config_path: str, experiment: str | None = None) -> ActionConditioningConfig:
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


def pil_to_tensor_image(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32)
    t = torch.from_numpy(arr).permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return t.unsqueeze(0).to(device)


def pil_list_to_tensor_video(images: list, device: torch.device) -> torch.Tensor:
    frames = []
    for img in images:
        arr = np.array(img, dtype=np.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        frames.append(t)
    video = torch.stack(frames, dim=1)
    return video.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Training module
# ---------------------------------------------------------------------------


class ACWMTrainingModule(WanTrainingModule):
    """
    Extends WanTrainingModule with a trainable `ConditionEncoder.action_encoder` (ActionFFN)
    and injects pre-encoded visual / action tensors into the Wan pipeline (same flags as inference).

    DiT is trained via LoRA from the parent class (`lora_base_model="dit"`).
    """

    def __init__(
        self,
        acwm_config_path: str,
        acwm_experiment: str | None = None,
        *args,
        extra_inputs: str | None = None,
        **kwargs,
    ):
        if extra_inputs is None:
            extra_inputs = "input_image"
        super().__init__(*args, extra_inputs=extra_inputs, **kwargs)
        self.acwm_cfg = load_acwm_config(acwm_config_path, acwm_experiment)
        dev = self.pipe.device
        self.condition_encoder = ConditionEncoder(self.acwm_cfg, device=dev).to(dev)
        for p in self.condition_encoder.action_encoder.parameters():
            p.requires_grad = True
        self.condition_encoder.train()

    def _encode_acwm_conditions(self, data: dict, inputs_shared: dict) -> dict:
        device = self.pipe.device
        dtype = self.pipe.torch_dtype
        cfg = self.acwm_cfg

        height = inputs_shared["height"]
        width = inputs_shared["width"]
        num_frames = inputs_shared["num_frames"]
        t_latent = (num_frames - 1) // cfg.vae_temporal_factor + 1
        h_latent = height // cfg.vae_spatial_factor
        w_latent = width // cfg.vae_spatial_factor

        noisy_ref = torch.zeros(
            1, cfg.vae_z_dim, t_latent, h_latent, w_latent,
            device=device, dtype=dtype,
        )

        obs_tensor = pil_to_tensor_image(data["obs_image"], device)
        action_tensor = data["actions"].unsqueeze(0).to(device=device, dtype=torch.float32)

        history_tensor = None
        if cfg.history_injection is not None and data.get("history_images"):
            history_tensor = pil_list_to_tensor_video(data["history_images"], device)

        encoded = self.condition_encoder.encode(
            obs_image=obs_tensor,
            actions=action_tensor,
            masked_traj=None,
            history=history_tensor,
            noisy_latent=noisy_ref,
        )
        if encoded.visual_latent is None or encoded.action_tokens is None:
            raise RuntimeError(
                "ConditionEncoder returned None for visual_latent or action_tokens; "
                "check action_conditioning.yaml (obs/history injection vs traj)."
            )

        inputs_shared["preencoded_visual_latent"] = encoded.visual_latent
        inputs_shared["preencoded_action_tokens"] = encoded.action_tokens
        inputs_shared["skip_condition_vae_encode"] = True
        return inputs_shared

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        inputs_shared, inputs_posi, inputs_nega = inputs
        inputs_shared = self._encode_acwm_conditions(data, inputs_shared)
        inputs = (inputs_shared, inputs_posi, inputs_nega)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


# ---------------------------------------------------------------------------
# Optimizer: optional separate LR for ActionFFNEncoder
# ---------------------------------------------------------------------------


def launch_acwm_training_task(
    accelerator: accelerate.Accelerator,
    dataset: torch.utils.data.Dataset,
    model: ACWMTrainingModule,
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

    if action_lr is not None:
        enc_params = list(model.condition_encoder.action_encoder.parameters())
        enc_ids = {id(p) for p in enc_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": learning_rate, "weight_decay": weight_decay},
                {"params": enc_params, "lr": action_lr, "weight_decay": weight_decay},
            ]
        )
    else:
        optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    log_every = getattr(args, "log_every_n_steps", 50) if args is not None else 50
    log_csv = getattr(args, "log_loss_to_csv", False) if args is not None else False
    log_csv_path = None
    if log_csv and args is not None and getattr(args, "output_path", None):
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
                if load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)

            global_step += 1
            if log_every > 0 and global_step % log_every == 0:
                lv = loss.detach().float()
                if accelerator.num_processes > 1:
                    if hasattr(accelerator, "reduce"):
                        lv = accelerator.reduce(lv, reduction="mean")
                    elif hasattr(accelerator, "gather_for_metrics"):
                        lv = accelerator.gather_for_metrics(lv.unsqueeze(0)).mean()
                    else:
                        lv = accelerator.gather(lv.unsqueeze(0)).mean()
                loss_val = lv.item() if isinstance(lv, torch.Tensor) else float(lv)
                if accelerator.is_main_process:
                    tqdm.write(f"[train] epoch={epoch_id} step={global_step} loss={loss_val:.6f}")
                    if log_csv_path is not None:
                        with open(log_csv_path, "a", newline="") as f:
                            csv.writer(f).writerow([global_step, epoch_id, f"{loss_val:.8f}"])
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def acwm_train_parser() -> argparse.ArgumentParser:
    parser = wan_parser()
    parser.add_argument(
        "--acwm_config",
        type=str,
        required=True,
        help="YAML with ActionConditioningConfig (same as inference_test / test_cond_encoder).",
    )
    parser.add_argument("--acwm_experiment", type=str, default=None, help="Experiment key under `experiments:` in YAML.")
    parser.add_argument(
        "--action_encoder_lr",
        type=float,
        default=None,
        help="Optional LR for ActionFFNEncoder only; --learning_rate applies to LoRA (DiT) and other trainable params.",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
        help="Print averaged loss every N dataloader steps (0 = disable). Uses accelerator.reduce for multi-GPU mean.",
    )
    parser.add_argument(
        "--log_loss_to_csv",
        action="store_true",
        help="Append rows to --output_path/training_loss.csv (columns: global_step, epoch, loss).",
    )
    # Relax dataset_base_path: ACWM metadata uses absolute frame paths.
    for action in parser._actions:
        if "--dataset_base_path" in action.option_strings:
            action.required = False
            if action.default is None or action.default == "":
                action.default = "."
            break
    return parser


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = acwm_train_parser()
    args = parser.parse_args()

    from acwm_dataset import ACWMDataset

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    if args.height is None or args.width is None:
        raise SystemExit("ACWM training requires explicit --height and --width (e.g. 368 640).")
    if args.dataset_metadata_path is None:
        raise SystemExit("--dataset_metadata_path is required (path to train_metadata.json).")

    dataset = ACWMDataset(
        args.dataset_metadata_path,
        height=args.height,
        width=args.width,
        repeat=args.dataset_repeat,
    )

    model = ACWMTrainingModule(
        acwm_config_path=args.acwm_config,
        acwm_experiment=args.acwm_experiment,
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
        device="cpu" if getattr(args, "initialize_model_on_cpu", False) else accelerator.device,
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
