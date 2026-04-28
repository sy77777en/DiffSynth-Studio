from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .config import ActionConditioningConfig, ConditionStreamConfig
from .encoder import ActionEncoder, MLPActionEncoder, PerceiverActionEncoder


@dataclass
class ConditionPacket:
    context_tokens: torch.Tensor
    adaln_delta: Optional[torch.Tensor]
    metadata: dict


def _build_action_encoder(
    cfg: ConditionStreamConfig,
    action_dim: int,
    perceiver_kwargs: dict,
) -> ActionEncoder:
    if cfg.encoder_type == "perceiver":
        return PerceiverActionEncoder(
            action_dim=action_dim,
            embed_dim=cfg.embed_dim,
            num_queries=cfg.num_queries or perceiver_kwargs["num_queries"],
            depth=perceiver_kwargs["depth"],
            num_heads=perceiver_kwargs["num_heads"],
            ff_mult=perceiver_kwargs["ff_mult"],
        )
    if cfg.encoder_type == "mlp":
        return MLPActionEncoder(action_dim=action_dim, embed_dim=cfg.embed_dim)
    raise ValueError(f"Unsupported action encoder_type: {cfg.encoder_type!r}")


class ConditionPipeline(nn.Module):
    """
    Convert raw condition inputs into backbone-ready tensors.

    Outputs a ConditionPacket:
    - context_tokens: tokens appended to backbone context input.
    - adaln_delta: kept for future non-native adapters.
    """

    def __init__(self, config: ActionConditioningConfig):
        super().__init__()
        self.config = config

        perceiver_kw = dict(
            num_queries=config.perceiver_num_queries,
            depth=config.perceiver_depth,
            num_heads=config.perceiver_num_heads,
            ff_mult=config.perceiver_ff_mult,
        )

        # Action branch
        self.action_encoder: Optional[ActionEncoder] = None
        self.action_cross_proj: Optional[nn.Linear] = None
        self.action_adaln_proj: Optional[nn.Sequential] = None
        if config.action.enabled:
            self.action_encoder = _build_action_encoder(config.action, config.action_dim, perceiver_kw)
            if config.action.injection_type == "cross_attn":
                self.action_cross_proj = nn.Linear(config.action.embed_dim, config.condition_context_dim)
                nn.init.zeros_(self.action_cross_proj.weight)
                nn.init.zeros_(self.action_cross_proj.bias)
            elif config.action.injection_type == "adaln":
                self.action_adaln_proj = nn.Sequential(
                    nn.Linear(config.action.embed_dim, config.backbone_dim),
                    nn.SiLU(),
                    nn.Linear(config.backbone_dim, config.backbone_dim),
                )
                nn.init.zeros_(self.action_adaln_proj[-1].weight)
                nn.init.zeros_(self.action_adaln_proj[-1].bias)
            else:
                raise ValueError(f"Unsupported action injection_type: {config.action.injection_type!r}")

        # obs image tokens (already from ViT)
        self.obs_proj: Optional[nn.Linear] = None
        if config.obs_image.enabled:
            self.obs_proj = nn.Linear(config.obs_image.embed_dim, config.condition_context_dim)
            nn.init.zeros_(self.obs_proj.weight)
            nn.init.zeros_(self.obs_proj.bias)

        # masked image sequence tokens (already from ViT)
        self.masked_proj: Optional[nn.Linear] = None
        if config.masked_image.enabled:
            self.masked_proj = nn.Linear(config.masked_image.embed_dim, config.condition_context_dim)
            nn.init.zeros_(self.masked_proj.weight)
            nn.init.zeros_(self.masked_proj.bias)

        self.null_context = nn.Parameter(torch.zeros(1, 1, config.condition_context_dim))

    def _validate_frame_alignment(
        self,
        noisy_latent: torch.Tensor,
        actions: Optional[torch.Tensor],
        masked_image_emb_seq: Optional[torch.Tensor],
    ) -> None:
        if not self.config.require_frame_alignment:
            return
        t_gen = noisy_latent.shape[2]
        if actions is not None and actions.shape[1] != t_gen:
            raise ValueError(f"Frame mismatch: action T={actions.shape[1]} != latent T={t_gen}")
        if masked_image_emb_seq is not None and masked_image_emb_seq.shape[1] != t_gen:
            raise ValueError(
                f"Frame mismatch: masked_image_seq T={masked_image_emb_seq.shape[1]} != latent T={t_gen}"
            )

    def build(
        self,
        noisy_latent: torch.Tensor,
        actions: Optional[torch.Tensor],
        obs_image_emb: Optional[torch.Tensor],
        masked_image_emb_seq: Optional[torch.Tensor],
        text_context: Optional[torch.Tensor] = None,
    ) -> ConditionPacket:
        """
        Args:
            noisy_latent: (B, C, T, H, W)
            actions: (B, T, action_dim)
            obs_image_emb: (B, N_obs, D_obs)
            masked_image_emb_seq: (B, T, N_mask, D_mask)
            text_context: optional raw text tokens
        """
        self._validate_frame_alignment(noisy_latent, actions, masked_image_emb_seq)

        context_parts = []
        adaln_delta = None

        if self.config.use_text and text_context is not None:
            if text_context.dim() == 4:
                text_context = text_context.squeeze(1)
            context_parts.append(text_context)

        if self.obs_proj is not None and obs_image_emb is not None:
            context_parts.append(self.obs_proj(obs_image_emb))

        if self.masked_proj is not None and masked_image_emb_seq is not None:
            b, t, n, d = masked_image_emb_seq.shape
            masked_tokens = masked_image_emb_seq.reshape(b, t * n, d)
            context_parts.append(self.masked_proj(masked_tokens))

        if self.action_encoder is not None and actions is not None:
            action_tokens = self.action_encoder(actions)
            if self.action_cross_proj is not None:
                context_parts.append(self.action_cross_proj(action_tokens))
            if self.action_adaln_proj is not None:
                adaln_delta = self.action_adaln_proj(action_tokens.mean(dim=1))

        if len(context_parts) == 0:
            b = noisy_latent.shape[0]
            context = self.null_context.expand(b, -1, -1)
        else:
            context = torch.cat(context_parts, dim=1)

        metadata = {
            "t_gen": noisy_latent.shape[2],
            "t_action": actions.shape[1] if actions is not None else None,
            "t_masked": masked_image_emb_seq.shape[1] if masked_image_emb_seq is not None else None,
            "n_context_tokens": context.shape[1],
        }
        return ConditionPacket(context_tokens=context, adaln_delta=adaln_delta, metadata=metadata)
