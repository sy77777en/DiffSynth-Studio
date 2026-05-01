import argparse
from typing import List

import torch

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


def _wan_dit_shards(model_dir: str, noise_subdir: str) -> List[str]:
    return [
        f"{model_dir}/{noise_subdir}/diffusion_pytorch_model-00001-of-00006.safetensors",
        f"{model_dir}/{noise_subdir}/diffusion_pytorch_model-00002-of-00006.safetensors",
        f"{model_dir}/{noise_subdir}/diffusion_pytorch_model-00003-of-00006.safetensors",
        f"{model_dir}/{noise_subdir}/diffusion_pytorch_model-00004-of-00006.safetensors",
        f"{model_dir}/{noise_subdir}/diffusion_pytorch_model-00005-of-00006.safetensors",
        f"{model_dir}/{noise_subdir}/diffusion_pytorch_model-00006-of-00006.safetensors",
    ]


def _longcat_dit_shards(model_dir: str) -> List[str]:
    return [
        f"{model_dir}/dit/diffusion_pytorch_model-00001-of-00006.safetensors",
        f"{model_dir}/dit/diffusion_pytorch_model-00002-of-00006.safetensors",
        f"{model_dir}/dit/diffusion_pytorch_model-00003-of-00006.safetensors",
        f"{model_dir}/dit/diffusion_pytorch_model-00004-of-00006.safetensors",
        f"{model_dir}/dit/diffusion_pytorch_model-00005-of-00006.safetensors",
        f"{model_dir}/dit/diffusion_pytorch_model-00006-of-00006.safetensors",
    ]


def build_model_configs(model_name: str, model_dir: str, noise_mode: str = "high_noise") -> List[ModelConfig]:
    if model_name == "wan_video_dit":
        noise_subdir = "high_noise_model" if noise_mode == "high_noise" else "low_noise_model"
        dit_path = _wan_dit_shards(model_dir, noise_subdir)
        text_path = f"{model_dir}/models_t5_umt5-xxl-enc-bf16.pth"
        vae_path = f"{model_dir}/Wan2.1_VAE.pth"
    elif model_name == "longcat_video_dit":
        dit_path = _longcat_dit_shards(model_dir)
        # Current DiffSynth setup typically pairs LongCat DiT with Wan text/VAE files.
        text_path = f"{model_dir}/models_t5_umt5-xxl-enc-bf16.pth"
        vae_path = f"{model_dir}/Wan2.1_VAE.pth"
    else:
        raise ValueError(f"Unsupported model_name: {model_name!r}")

    return [
        ModelConfig(path=dit_path, offload_device="cpu"),
        ModelConfig(path=text_path, offload_device="cpu"),
        ModelConfig(path=vae_path, offload_device="cpu"),
    ]


def build_pipe(
    model_name: str,
    model_dir: str,
    device: str = "cuda",
    noise_mode: str = "high_noise",
) -> WanVideoPipeline:
    model_configs = build_model_configs(model_name=model_name, model_dir=model_dir, noise_mode=noise_mode)
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build WanVideoPipeline from local model folder.")
    parser.add_argument("--model_name", type=str, required=True, choices=["wan_video_dit", "longcat_video_dit"])
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--noise_mode", type=str, default="high_noise", choices=["high_noise", "low_noise"])
    args = parser.parse_args()

    pipe = build_pipe(
        model_name=args.model_name,
        model_dir=args.model_dir,
        device=args.device,
        noise_mode=args.noise_mode,
    )
    print(f"Pipeline ready: model_name={args.model_name}, device={args.device}, noise_mode={args.noise_mode}")
    print(f"Loaded DiT type: {type(pipe.dit).__name__ if pipe.dit is not None else 'None'}")
    print(f"Loaded VAE type: {type(pipe.vae).__name__ if pipe.vae is not None else 'None'}")
