from typing import Optional

import torch
import torch.amp as amp

from .base import BaseBackboneAdapter


class LongCatBackboneAdapter(BaseBackboneAdapter):
    def maybe_project_text(self, text_context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if text_context is None:
            return None
        y = self.backbone.y_embedder(text_context)
        if y.dim() == 4:
            y = y.squeeze(1)
        return y

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        context_tokens: torch.Tensor,
        adaln_delta: Optional[torch.Tensor],
        **backbone_kwargs,
    ) -> torch.Tensor:
        bb = self.backbone
        b, _, t, h, w = noisy_latent.shape

        n_t = t // bb.patch_size[0]
        n_h = h // bb.patch_size[1]
        n_w = w // bb.patch_size[2]

        x = bb.x_embedder(noisy_latent)

        if len(timestep.shape) == 1:
            timestep_expanded = timestep.unsqueeze(1).expand(-1, n_t).clone()
        else:
            timestep_expanded = timestep

        with amp.autocast(device_type=noisy_latent.device.type, dtype=torch.float32):
            t_emb = bb.t_embedder(timestep_expanded.float().flatten(), dtype=torch.float32).reshape(b, n_t, -1)

        if adaln_delta is not None:
            t_emb = t_emb + adaln_delta.unsqueeze(1)

        encoder_attention_mask = backbone_kwargs.get("encoder_attention_mask", None)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            context = context_tokens.masked_select(encoder_attention_mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist()
        else:
            context = context_tokens.view(1, -1, x.shape[-1])
            y_seqlens = [context_tokens.shape[1]] * b

        for block in bb.blocks:
            x = block(x=x, y=context, t=t_emb, y_seqlen=y_seqlens, latent_shape=(n_t, n_h, n_w))

        x = bb.final_layer(x, t_emb, (n_t, n_h, n_w))
        x = bb.unpatchify(x, n_t, n_h, n_w)
        return x.to(torch.float32)
