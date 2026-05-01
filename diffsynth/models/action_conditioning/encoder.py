"""
Condition Encoder — reads config to select the right VAE, encodes all
conditions, and produces concat-ready latents.

Usage:
    config = ActionConditioningConfig(backbone="wan", vae_path="path/to/vae.pt")
    cond_enc = ConditionEncoder(config, device="cuda")

    encoded = cond_enc.encode(
        obs_image=obs,            # (B, 3, H, W)
        actions=delta_actions,    # (B, T, action_dim)
        masked_traj=traj_imgs,    # (B, 3, T, H, W)
    )
    # encoded.concat_latent: (B, C_concat, T', H', W') — ready for input_concat
    # encoded.action_tokens: (B, T, embed_dim) — for adaln or cross_attn
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .config import ActionConditioningConfig
from .model_paths import resolve_vae_ckpt_path
# from .encoders.action_encoder import ActionFFNEncoder


class ActionFFNEncoder(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int, num_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(action_dim, embed_dim), nn.GELU()]
        for _ in range(max(0, num_layers - 2)):
            layers += [nn.Linear(embed_dim, embed_dim), nn.GELU()]
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mlp(actions))


@dataclass
class EncodedConditions:
    """All encoded conditions, ready for the DiT wrapper."""

    action_tokens: Optional[torch.Tensor] = None   # (B, T, embed_dim)
    obs_latent: Optional[torch.Tensor] = None       # (B, C, 1, H', W')
    traj_latent: Optional[torch.Tensor] = None      # (B, C, T', H', W')
    history_latent: Optional[torch.Tensor] = None    # (B, C, K', H', W')
    visual_latent: Optional[torch.Tensor] = None     # (B, C, T_vis, H', W') — unified visual condition
    concat_latent: Optional[torch.Tensor] = None     # (B, C_total, T', H', W') — all input_concat streams merged


def _build_vae(config: ActionConditioningConfig):
    # model-name driven
    if config.vae_model_name == "wan_video_vae":
        from diffsynth.models.wan_video_vae import WanVideoVAE
        vae = WanVideoVAE(z_dim=config.vae_z_dim)
    else:
        raise ValueError(f"Unknown vae_model_name: {config.vae_model_name!r}")

    # optional load from explicit path or model_root-derived path
    vae_ckpt_path = resolve_vae_ckpt_path(config)
    missing_keys = []
    unexpected_keys = []
    if vae_ckpt_path is not None:
        from diffsynth.core.loader.file import load_state_dict
        state_dict = load_state_dict(vae_ckpt_path, device="cpu")
        if hasattr(vae, "state_dict_converter"):
            converter = vae.state_dict_converter()
            state_dict = converter.from_civitai(state_dict)
        incompatible = vae.load_state_dict(state_dict, strict=False)
        if incompatible is not None:
            missing_keys = list(getattr(incompatible, "missing_keys", []))
            unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
    else:
        raise ValueError(
            "VAE checkpoint path is not resolved. Please set config.vae_path "
            "or config.model_root with a valid Wan VAE file."
        )

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    # Attach debug metadata for verification in tests/scripts.
    vae._loaded_ckpt_path = vae_ckpt_path
    vae._missing_keys = missing_keys
    vae._unexpected_keys = unexpected_keys
    return vae


class ConditionEncoder(nn.Module):
    """
    Reads config.backbone to select the right VAE, encodes all visual
    conditions through it, and produces concat-ready latents.

    The VAE is frozen. Only the action_encoder is trainable.
    """

    def __init__(self, config: ActionConditioningConfig, device: torch.device = torch.device("cuda")):
        super().__init__()
        self.config = config
        self.device = device

        # Build frozen VAE (not registered as submodule to avoid saving its weights)
        self._vae = _build_vae(config)

        # Trainable action encoder
        self.action_encoder = ActionFFNEncoder(
            action_dim=config.action_dim,
            embed_dim=config.action_embed_dim,
            num_layers=config.action_num_layers,
        )

    @property
    def vae(self):
        return self._vae

    def to(self, *args, **kwargs):
        """Override to also move the frozen VAE."""
        super().to(*args, **kwargs)
        self._vae = self._vae.to(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # VAE encoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _vae_encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, C, 1, H', W')"""
        videos = [img.unsqueeze(1) for img in image]
        return self._vae.encode(videos, self.device)

    @torch.no_grad()
    def _vae_encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """(B, 3, T, H, W) → (B, C, T', H', W')"""
        videos = [v for v in video]
        return self._vae.encode(videos, self.device)

    # ------------------------------------------------------------------
    # Main encode method
    # ------------------------------------------------------------------

    def encode(
        self,
        obs_image: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        masked_traj: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        noisy_latent: Optional[torch.Tensor] = None,
    ) -> EncodedConditions:
        """
        Encode all conditions and assemble concat_latent.

        Parameters
        ----------
        obs_image : (B, 3, H, W) — observation frame
        actions : (B, T, action_dim) — delta action vectors
        masked_traj : (B, 3, T, H, W) — rendered trajectory images
        history : (B, 3, K, H, W) — past frames
        noisy_latent : (B, C, T', H', W') — the noisy latent (needed for
            shape reference when broadcasting obs to T' frames)

        Returns
        -------
        EncodedConditions
        """
        result = EncodedConditions()
        cfg = self.config

        # --- Action tokens (trainable) ---
        if actions is not None:
            result.action_tokens = self.action_encoder(actions)

        # --- VAE encode visual conditions (frozen) ---
        # If both observation and history are provided, encode them together in pixel space
        # so temporal compression happens on the combined past context stream.
        if (
            obs_image is not None
            and history is not None
            and (cfg.obs_injection is not None or cfg.history_injection is not None)
        ):
            past_video = torch.cat([obs_image.unsqueeze(2), history], dim=2)  # (B, 3, 1+K, H, W)
            result.history_latent = self._vae_encode_video(past_video)
            result.obs_latent = None
        elif obs_image is not None and cfg.obs_injection is not None:
            result.obs_latent = self._vae_encode_image(obs_image)

        if masked_traj is not None and cfg.traj_injection is not None:
            result.traj_latent = self._vae_encode_video(masked_traj)

        if history is not None and cfg.history_injection is not None and result.history_latent is None:
            result.history_latent = self._vae_encode_video(history)

        # --- Assemble unified visual condition ---
        # Merge observation into history first, then concat with trajectory.
        # This keeps one "past context" stream + one "future trajectory" stream.
        visual_parts = []

        history_merged = result.history_latent
        if result.obs_latent is not None:
            if history_merged is None:
                history_merged = result.obs_latent
            else:
                history_merged = torch.cat([result.obs_latent, history_merged], dim=2)

        if result.traj_latent is not None:
            visual_parts.append(result.traj_latent)
        if history_merged is not None:
            visual_parts.append(history_merged)

        if visual_parts:
            result.visual_latent = torch.cat(visual_parts, dim=2)
        
        # --- Build condition mask (4-channel, per sub-frame) ---
        if result.visual_latent is not None and noisy_latent is not None:
            B = noisy_latent.shape[0]
            _, C_vis, T_vis, H_lat, W_lat = result.visual_latent.shape  # ← 用 visual_latent 的维度

            mask = torch.zeros(B, 4, T_vis, H_lat, W_lat,              # ← T_vis 而不是 T_lat
                               device=result.visual_latent.device,
                               dtype=result.visual_latent.dtype)

            n_traj_pixels = masked_traj.shape[2] if masked_traj is not None else 0
            real_start_lat = (n_traj_pixels + 3) // 4
            mask[:, :, real_start_lat:] = 1.0

            result.visual_latent = torch.cat([mask, result.visual_latent], dim=1)
            result.concat_latent = result.visual_latent

        # # Backward compatibility for existing callers/tests.
        # if (
        #     cfg.obs_injection == "input_concat"
        #     or cfg.traj_injection == "input_concat"
        #     or cfg.history_injection == "input_concat"
        # ):
        #     result.concat_latent = result.visual_latent

        return result