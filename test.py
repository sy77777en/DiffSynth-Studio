import torch
import yaml
from dataclasses import fields
from PIL import Image

from diffsynth.models.action_conditioning.encoder import ConditionEncoder
from diffsynth.models.action_conditioning.config import ActionConditioningConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

DEVICE = "cuda:7"
CFG_PATH = "/data2/siyuanc4/code/DiffSynth-Studio/configs/action_conditioning.yaml"

with open(CFG_PATH, "r") as f:
    raw = yaml.safe_load(f)

exp_name = raw.get("experiment", "wan")
experiments = raw.get("experiments", {})
if exp_name not in experiments:
    raise ValueError(f"Unknown experiment '{exp_name}', available: {list(experiments.keys())}")

exp_raw = experiments[exp_name]
exp_cfg = exp_raw
valid = {f.name for f in fields(ActionConditioningConfig)}
exp_cfg = {k: v for k, v in exp_cfg.items() if k in valid}

cfg = ActionConditioningConfig(**exp_cfg)

print(f"[Config] experiment={exp_name}")
print(f"[Config] model_name={cfg.model_name}, backbone={cfg.backbone}, vae_model_name={cfg.vae_model_name}")

cond_encoder = ConditionEncoder(cfg, device=torch.device(DEVICE)).to(DEVICE)
print(f"[Runtime] vae_class={type(cond_encoder.vae).__name__}")
print(f"[Runtime] vae_ckpt_path={getattr(cond_encoder.vae, '_loaded_ckpt_path', None)}")
print(
    f"[Runtime] load_state_dict missing_keys={len(getattr(cond_encoder.vae, '_missing_keys', []))}, "
    f"unexpected_keys={len(getattr(cond_encoder.vae, '_unexpected_keys', []))}"
)

# Build full Wan pipeline from local ckpts (DiT + TextEncoder + VAE).
MODEL_DIR = exp_raw.get("model_dir", exp_raw.get("model_root"))
if MODEL_DIR is None:
    raise ValueError("Please set model_dir (or model_root) in YAML experiment config.")

# pipe = WanVideoPipeline.from_pretrained(
#     torch_dtype=torch.bfloat16,
#     device=DEVICE,
#     model_configs=[
#         ModelConfig(path=[f"{MODEL_DIR}/high_noise_model/diffusion_pytorch_model-0000{i}-of-00006.safetensors" for i in range(1, 7)], offload_device="cpu"),
#         ModelConfig(path=[f"{MODEL_DIR}/low_noise_model/diffusion_pytorch_model-0000{i}-of-00006.safetensors" for i in range(1, 7)], offload_device="cpu"),
#         ModelConfig(path=f"{MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
#         ModelConfig(path=f"{MODEL_DIR}/Wan2.1_VAE.pth", offload_device="cpu"),
#     ],
#     tokenizer_config=ModelConfig(path=f"{MODEL_DIR}/google/umt5-xxl"),
# )

B, T, K, H, W = 1, 16, 3, 224, 224
obs_image    = torch.randn(B, 3, H, W, device=DEVICE)
masked_traj  = torch.randn(B, 3, T, H, W, device=DEVICE)
history      = torch.randn(B, 3, K, H, W, device=DEVICE)
actions      = torch.randn(B, T, cfg.action_dim, device=DEVICE)
noisy_latent = torch.randn(B, cfg.vae_z_dim, (T + 1 - 1) // 4 + 1, 28, 28, device=DEVICE) # (17-1)//4+1 = 5

encoded = cond_encoder.encode(
    obs_image=obs_image,
    actions=actions,
    masked_traj=masked_traj,
    history=history,
    noisy_latent=noisy_latent,
)

print("action_tokens:", None if encoded.action_tokens is None else encoded.action_tokens.shape)
print("obs_latent:", None if encoded.obs_latent is None else encoded.obs_latent.shape)
print("traj_latent:", None if encoded.traj_latent is None else encoded.traj_latent.shape)
print("history_latent:", None if encoded.history_latent is None else encoded.history_latent.shape)
print("concat_latent:", None if encoded.concat_latent is None else encoded.concat_latent.shape)

# ---- concat shape sanity check ----
if encoded.concat_latent is not None:
    t_obs = 0 if encoded.obs_latent is None else encoded.obs_latent.shape[2]
    t_traj = 0 if encoded.traj_latent is None else encoded.traj_latent.shape[2]
    t_hist = 0 if encoded.history_latent is None else encoded.history_latent.shape[2]
    t_expected = t_obs + t_traj + t_hist
    c_expected = cfg.vae_z_dim + 4  # 16 VAE channels + 4 mask channels = 20

    print(f"[Check] T parts: obs={t_obs}, traj={t_traj}, hist={t_hist}, expected_total={t_expected}")
    print(f"[Check] concat shape: {tuple(encoded.concat_latent.shape)}")

    assert encoded.concat_latent.shape[0] == B
    assert encoded.concat_latent.shape[1] == c_expected
    assert encoded.concat_latent.shape[2] == t_expected
    print("[Check] concat_latent shape check passed.")

# # Run pipeline while skipping internal condition VAE encode.
# dummy_input_image = Image.new("RGB", (832, 480), color=(127, 127, 127))
# video = pipe(
#     # prompt="a robot arm moves smoothly in a lab",
#     negative_prompt="",
#     input_image=dummy_input_image,
#     height=480,
#     width=832,
#     num_frames=17,
#     seed=0,
#     tiled=True,
#     preencoded_visual_latent=encoded.visual_latent,
#     skip_condition_vae_encode=True,
# )
# print("video_frames:", len(video))