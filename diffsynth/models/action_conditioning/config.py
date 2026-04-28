from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ConditionStreamConfig:
    """Configuration for one conditioning stream."""

    injection_type: Literal["cross_attn", "adaln"]
    encoder_type: Literal["perceiver", "mlp", "identity"]
    embed_dim: int = 1024
    num_queries: int = 16
    enabled: bool = True


@dataclass
class ActionConditioningConfig:
    """Config for pure world action conditioning."""

    backbone: Literal["wan", "cogvideo", "longcat"] = "wan"
    backbone_dim: int = 5120
    use_text: bool = False
    text_dim: int = 4096
    condition_context_dim: int = 4096
    action_dim: int = 14
    obs_image_dim: int = 1024
    masked_image_dim: int = 1024
    require_frame_alignment: bool = True

    # Condition streams
    action: ConditionStreamConfig = field(
        default_factory=lambda: ConditionStreamConfig(
            injection_type="cross_attn",
            encoder_type="perceiver",
            embed_dim=1024,
        )
    )
    obs_image: ConditionStreamConfig = field(
        default_factory=lambda: ConditionStreamConfig(
            injection_type="cross_attn",
            encoder_type="identity",
            embed_dim=1024,
            num_queries=1,
        )
    )
    masked_image: ConditionStreamConfig = field(
        default_factory=lambda: ConditionStreamConfig(
            injection_type="cross_attn",
            encoder_type="identity",
            embed_dim=1024,
            num_queries=16,
        )
    )

    # Shared Perceiver defaults
    perceiver_num_queries: int = 16
    perceiver_depth: int = 4
    perceiver_num_heads: int = 8
    perceiver_ff_mult: int = 4
