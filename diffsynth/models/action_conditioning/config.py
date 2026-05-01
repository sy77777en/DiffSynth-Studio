"""
Configuration for action-conditioned video DiT.
Select model by model_name, then derive backbone/vae settings from mapping.
"""

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ActionConditioningConfig:
    # Single source of truth
    model_name: Literal["wan_video_dit", "longcat_video_dit"] = "wan_video_dit"

    # Action
    action_dim: int = 14
    action_embed_dim: int = 1024
    action_num_layers: int = 3

    # Injection methods (None = disabled)
    # Action condition is routed as context tokens for transformer cross-attn.
    action_injection: Optional[Literal["adaln", "cross_attn"]] = "cross_attn"
    # Visual condition is routed via a dedicated visual attention branch.
    visual_injection: Optional[Literal["spatial_attn", "input_concat", "cross_attn"]] = "spatial_attn"

    # Visual stream switches used by encoder assembly.
    obs_injection: Optional[Literal["input_concat", "cross_attn"]] = "input_concat"
    traj_injection: Optional[Literal["input_concat", "cross_attn"]] = None
    history_injection: Optional[Literal["input_concat"]] = None
    history_frames: int = 3

    # Optional fallback weight path (debug only)
    vae_path: Optional[str] = None
    # Optional override for VAE model name routing.
    vae_model_name_override: Optional[str] = None
    # Optional model root for automatic ckpt path resolution by model_name.
    model_root: Optional[str] = None

    # Perceiver
    traj_perceiver_num_queries: int = 64
    traj_perceiver_depth: int = 4
    traj_perceiver_num_heads: int = 8

    @property
    def backbone(self) -> str:
        return {
            "wan_video_dit": "wan",
            "longcat_video_dit": "longcat",
        }[self.model_name]

    @property
    def vae_model_name(self) -> str:
        if self.vae_model_name_override is not None:
            return self.vae_model_name_override
        return {
            "wan_video_dit": "wan_video_vae",
            "longcat_video_dit": "wan_video_vae",
        }[self.model_name]

    @property
    def vae_z_dim(self) -> int:
        return {"wan_video_vae": 16}[self.vae_model_name]

    @property
    def vae_temporal_factor(self) -> int:
        return {"wan_video_vae": 4}[self.vae_model_name]

    @property
    def vae_spatial_factor(self) -> int:
        return {"wan_video_vae": 8}[self.vae_model_name]

    @property
    def concat_channels(self) -> int:
        n = 0
        if self.obs_injection == "input_concat":
            n += self.vae_z_dim
        if self.traj_injection == "input_concat":
            n += self.vae_z_dim
        if self.history_injection == "input_concat":
            n += self.vae_z_dim
        return n