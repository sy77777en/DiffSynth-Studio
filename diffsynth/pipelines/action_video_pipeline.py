import torch
from PIL import Image
from tqdm import tqdm
from typing import Optional, Union

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.wan_video_dit import WanModel
from ..models.longcat_video_dit import LongCatVideoTransformer3DModel
from ..models.wan_video_vae import WanVideoVAE
from ..models.action_conditioning import ActionConditionedDiT, ActionConditioningConfig


class ActionVideoPipeline(BasePipeline):
    """
    Native DiffSynth pipeline for action-conditioned world-model video generation.

    The denoising loop and unit-based preprocessing follow the existing DiffSynth
    pipeline style. The DiT backbone itself is not modified; all condition signals
    are routed through ActionConditionedDiT.
    """

    def __init__(
        self,
        action_conditioning_config: ActionConditioningConfig,
        device=get_device_type(),
        torch_dtype=torch.bfloat16,
    ):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.action_conditioning_config = action_conditioning_config

        # Models
        self.dit_backbone: Optional[torch.nn.Module] = None
        self.dit: Optional[ActionConditionedDiT] = None
        self.vae: Optional[WanVideoVAE] = None

        self.in_iteration_models = ("dit",)
        self.units = [
            ActionVideoUnit_ShapeChecker(),
            ActionVideoUnit_NoiseInitializer(),
            ActionVideoUnit_InputVideoEmbedder(),
            ActionVideoUnit_ConditionValidator(),
        ]
        self.model_fn = model_fn_action_video
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        action_conditioning_config: ActionConditioningConfig,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        vram_limit: float = None,
    ):
        pipe = ActionVideoPipeline(
            action_conditioning_config=action_conditioning_config,
            device=device,
            torch_dtype=torch_dtype,
        )
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Prefer Wan DiT, fallback to LongCat DiT.
        backbone = model_pool.fetch_model("wan_video_dit")
        if backbone is None:
            backbone = model_pool.fetch_model("longcat_video_dit")
        if backbone is None:
            raise ValueError("No supported DiT backbone found. Expect wan_video_dit or longcat_video_dit.")
        if not isinstance(backbone, (WanModel, LongCatVideoTransformer3DModel)):
            raise ValueError(f"Unsupported backbone type: {type(backbone)}")

        pipe.dit_backbone = backbone
        pipe.dit = ActionConditionedDiT(backbone, action_conditioning_config)
        pipe.vae = model_pool.fetch_model("wan_video_vae")

        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Condition inputs
        actions: torch.Tensor,
        obs_image_emb: Optional[torch.Tensor] = None,
        masked_image_emb_seq: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        # Optional init
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: float = 1.0,
        # Shape
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        # Randomness
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        # Scheduler
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        # VAE decode
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        # Output
        output_type: str = "quantized",
        progress_bar_cmd=tqdm,
        # Backbone kwargs
        **backbone_kwargs,
    ):
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )

        inputs_shared = {
            "actions": actions,
            "obs_image_emb": obs_image_emb,
            "masked_image_emb_seq": masked_image_emb_seq,
            "text_context": text_context,
            "input_video": input_video,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "seed": seed,
            "rand_device": rand_device,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        inputs_posi = {}
        inputs_nega = {}
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.model_fn(
                **models,
                latents=inputs_shared["latents"],
                timestep=timestep,
                actions=inputs_shared["actions"],
                obs_image_emb=inputs_shared.get("obs_image_emb"),
                masked_image_emb_seq=inputs_shared.get("masked_image_emb_seq"),
                text_context=inputs_shared.get("text_context"),
                **backbone_kwargs,
            )
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])

        # Decode
        if self.vae is None:
            raise ValueError("wan_video_vae is required for decoding output latents.")
        self.load_models_to_device(["vae"])
        video = self.vae.decode(
            inputs_shared["latents"],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


class ActionVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: ActionVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class ActionVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: ActionVideoPipeline, height, width, num_frames, seed, rand_device):
        if pipe.vae is None:
            raise ValueError("wan_video_vae is required to infer latent shape.")
        length = (num_frames - 1) // 4 + 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class ActionVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: ActionVideoPipeline, input_video, noise, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise, "input_latents": None}
        if pipe.vae is None:
            raise ValueError("wan_video_vae is required for input_video encoding.")
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(
            input_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        return {"latents": latents, "input_latents": input_latents}


class ActionVideoUnit_ConditionValidator(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("actions", "num_frames", "masked_image_emb_seq"),
            output_params=("actions", "masked_image_emb_seq"),
        )

    def process(self, pipe: ActionVideoPipeline, actions, num_frames, masked_image_emb_seq):
        if actions is None:
            raise ValueError("`actions` is required for action-conditioned world model.")
        if actions.shape[1] != num_frames:
            raise ValueError(f"actions length ({actions.shape[1]}) must equal num_frames ({num_frames}).")
        if masked_image_emb_seq is not None and masked_image_emb_seq.shape[1] != num_frames:
            raise ValueError(
                f"masked_image_emb_seq length ({masked_image_emb_seq.shape[1]}) must equal num_frames ({num_frames})."
            )
        return {"actions": actions, "masked_image_emb_seq": masked_image_emb_seq}


def model_fn_action_video(
    dit: ActionConditionedDiT,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    actions: torch.Tensor,
    obs_image_emb: Optional[torch.Tensor] = None,
    masked_image_emb_seq: Optional[torch.Tensor] = None,
    text_context: Optional[torch.Tensor] = None,
    **backbone_kwargs,
):
    return dit(
        noisy_latent=latents,
        timestep=timestep,
        actions=actions,
        obs_image_emb=obs_image_emb,
        masked_image_emb_seq=masked_image_emb_seq,
        text_context=text_context,
        **backbone_kwargs,
    )
