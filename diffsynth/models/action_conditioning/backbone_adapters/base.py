from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseBackboneAdapter(nn.Module, ABC):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    @abstractmethod
    def maybe_project_text(self, text_context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        pass

    @abstractmethod
    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        context_tokens: torch.Tensor,
        adaln_delta: Optional[torch.Tensor],
        **backbone_kwargs,
    ) -> torch.Tensor:
        pass
