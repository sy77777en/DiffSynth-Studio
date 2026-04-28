from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .condition_pipeline import ConditionPipeline
from .config import ActionConditioningConfig


class ActionConditionedDiT(nn.Module):
    """Orchestrates condition processing and backbone execution."""

    def __init__(self, backbone: nn.Module, config: ActionConditioningConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.condition_pipeline = ConditionPipeline(config)
        if config.backbone not in ("wan", "longcat", "cogvideo"):
            raise ValueError(f"Unknown backbone: {config.backbone!r}")
        if config.action.enabled and config.action.injection_type == "adaln":
            raise NotImplementedError(
                "Native integration mode does not modify backbone timestep path. "
                "Please use action.injection_type='cross_attn'."
            )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        obs_image_emb: Optional[torch.Tensor] = None,
        masked_image_emb_seq: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        return_condition_metadata: bool = False,
        **backbone_kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Pure world-model forward with optional text branch.

        Args:
            noisy_latent: (B, C, T, H, W)
            timestep: (B,)
            actions: (B, T, action_dim)
            obs_image_emb: (B, N_obs, D_obs)
            masked_image_emb_seq: (B, T, N_mask, D_mask)
            text_context: (B, S, text_dim), optional
        """
        packet = self.condition_pipeline.build(
            noisy_latent=noisy_latent,
            actions=actions,
            obs_image_emb=obs_image_emb,
            masked_image_emb_seq=masked_image_emb_seq,
            text_context=text_context,
        )
        context = packet.context_tokens

        if self.config.backbone == "wan":
            out = self.backbone(
                noisy_latent,
                timestep,
                context,
                **backbone_kwargs,
            )
        elif self.config.backbone == "longcat":
            if context.dim() == 3:
                context = context.unsqueeze(1)
            encoder_attention_mask = backbone_kwargs.pop("encoder_attention_mask", None)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    (context.shape[0], 1, 1, context.shape[2]),
                    dtype=torch.int64,
                    device=context.device,
                )
            out = self.backbone(
                noisy_latent,
                timestep,
                context,
                encoder_attention_mask=encoder_attention_mask,
                **backbone_kwargs,
            )
        elif self.config.backbone == "cogvideo":
            raise NotImplementedError("CogVideo native wrapper is not implemented yet.")
        else:
            raise ValueError(f"Unknown backbone: {self.config.backbone!r}")
        if return_condition_metadata:
            return out, packet.metadata
        return out
