from typing import Optional

import torch
from einops import rearrange

from .base import BaseBackboneAdapter
from ...wan_video_dit import sinusoidal_embedding_1d


class WanBackboneAdapter(BaseBackboneAdapter):
    def maybe_project_text(self, text_context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if text_context is None:
            return None
        return self.backbone.text_embedding(text_context)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        context_tokens: torch.Tensor,
        adaln_delta: Optional[torch.Tensor],
        **backbone_kwargs,
    ) -> torch.Tensor:
        bb = self.backbone
        x = noisy_latent

        t = bb.time_embedding(sinusoidal_embedding_1d(bb.freq_dim, timestep).to(x.dtype))
        if adaln_delta is not None:
            t = t + adaln_delta
        t_mod = bb.time_projection(t).unflatten(1, (6, bb.dim))

        x = bb.patch_embedding(x)
        b, c, f, h, w = x.shape
        x = rearrange(x, "b c f h w -> b (f h w) c")

        freqs = torch.cat(
            [
                bb.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                bb.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                bb.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(x.device)

        for block in bb.blocks:
            x = block(x, context_tokens, t_mod, freqs)

        x = bb.head(x, t)
        x = bb.unpatchify(x, (f, h, w))
        return x
