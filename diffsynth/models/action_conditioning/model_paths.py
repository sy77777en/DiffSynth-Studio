from __future__ import annotations

import os
from typing import Optional

from .config import ActionConditioningConfig

VAE_PATH_CANDIDATES = {
    "wan_video_vae": [
        "Wan2.1_VAE.pth",
        "Wan2.1_VAE.safetensors",
        "Wan2.2_VAE.pth",
        "Wan2.2_VAE.safetensors",
    ],
    "longcat_video_vae": [
        "vae/diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.safetensors",
    ],
}


def resolve_vae_ckpt_path(config: ActionConditioningConfig) -> Optional[str]:
    """
    Resolve VAE checkpoint path from config.

    Priority:
    1) explicit config.vae_path
    2) inferred from config.model_root + model_name defaults
    """
    if config.vae_path:
        return config.vae_path

    if not config.model_root:
        return None

    rel_candidates = VAE_PATH_CANDIDATES.get(config.vae_model_name, [])
    for rel_path in rel_candidates:
        candidate = os.path.join(config.model_root, rel_path)
        if os.path.exists(candidate):
            return candidate

    return None
