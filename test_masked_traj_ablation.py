import torch
import numpy as np
import yaml
from dataclasses import fields
from PIL import Image

from diffsynth.models.action_conditioning.encoder import ConditionEncoder
from diffsynth.models.action_conditioning.config import ActionConditioningConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

DEVICE = "cuda:0"
CFG_PATH = "./configs/action_conditioning.yaml"


def frame_checksum(video_frames):
    return float(torch.from_numpy(np.array(video_frames[0])).float().mean().item())


def run_pipe_case(pipe, name, input_image, action_tokens, visual_latent, use_preencoded):
    kwargs = dict(
        prompt="",
        negative_prompt="",
        input_image=input_image,
        height=368,
        width=640,
        num_frames=17,
        num_inference_steps=6,
        seed=0,
        tiled=True,
    )
    if use_preencoded:
        kwargs.update(
            preencoded_visual_latent=visual_latent,
            preencoded_action_tokens=action_tokens,
            skip_condition_vae_encode=True,
        )
    video = pipe(**kwargs)
    print(f"[Case:{name}] frames={len(video)}, checksum={frame_checksum(video):.6f}")
    return video


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

# 1) External condition encoder forward
cond_encoder = ConditionEncoder(cfg, device=torch.device(DEVICE)).to(DEVICE)
print(f"[Runtime] vae_class={type(cond_encoder.vae).__name__}")
print(f"[Runtime] vae_ckpt_path={getattr(cond_encoder.vae, '_loaded_ckpt_path', None)}")
print(
    f"[Runtime] load_state_dict missing_keys={len(getattr(cond_encoder.vae, '_missing_keys', []))}, "
    f"unexpected_keys={len(getattr(cond_encoder.vae, '_unexpected_keys', []))}"
)

B, T, K, H, W = 1, 16, 3, 368, 640
obs_image = torch.randn(B, 3, H, W, device=DEVICE)
masked_traj = torch.randn(B, 3, T, H, W, device=DEVICE)
history = torch.randn(B, 3, K, H, W, device=DEVICE)
actions = torch.randn(B, T, cfg.action_dim, device=DEVICE)
noisy_latent = torch.randn(B, cfg.vae_z_dim, (17 - 1) // 4 + 1, 28, 28, device=DEVICE)  # T_latent=5

# Encode two conditions for the intended ablation:
#   A: action + observation + history
#   B: action + observation + history + masked trajectory image
# Keep action / obs_image / history / noisy_latent fixed. Only masked_traj changes.
masked_traj_zero = torch.zeros_like(masked_traj)

encoded_no_traj = cond_encoder.encode(
    obs_image=obs_image,
    actions=actions,
    masked_traj=None,
    history=history,
    noisy_latent=noisy_latent,
)

# encoded_with_traj = cond_encoder.encode(
#     obs_image=obs_image,
#     actions=actions,
#     masked_traj=masked_traj,
#     history=history,
#     noisy_latent=noisy_latent,
# )

print("[Encoded:no_traj] action_tokens:", None if encoded_no_traj.action_tokens is None else encoded_no_traj.action_tokens.shape)
print("[Encoded:no_traj] obs_latent:", None if encoded_no_traj.obs_latent is None else encoded_no_traj.obs_latent.shape)
print("[Encoded:no_traj] traj_latent:", None if encoded_no_traj.traj_latent is None else encoded_no_traj.traj_latent.shape)
print("[Encoded:no_traj] history_latent:", None if encoded_no_traj.history_latent is None else encoded_no_traj.history_latent.shape)
print("[Encoded:no_traj] visual_latent:", None if encoded_no_traj.visual_latent is None else encoded_no_traj.visual_latent.shape)

# print("[Encoded:with_traj] action_tokens:", None if encoded_with_traj.action_tokens is None else encoded_with_traj.action_tokens.shape)
# print("[Encoded:with_traj] obs_latent:", None if encoded_with_traj.obs_latent is None else encoded_with_traj.obs_latent.shape)
# print("[Encoded:with_traj] traj_latent:", None if encoded_with_traj.traj_latent is None else encoded_with_traj.traj_latent.shape)
# print("[Encoded:with_traj] history_latent:", None if encoded_with_traj.history_latent is None else encoded_with_traj.history_latent.shape)
# print("[Encoded:with_traj] visual_latent:", None if encoded_with_traj.visual_latent is None else encoded_with_traj.visual_latent.shape)

# # Sanity check: action should be held fixed across this ablation.
# if encoded_no_traj.action_tokens is not None and encoded_with_traj.action_tokens is not None:
#     action_token_delta = (encoded_with_traj.action_tokens - encoded_no_traj.action_tokens).abs().max().item()
#     print(f"[Sanity] max action token delta(no_traj vs with_traj)={action_token_delta:.6e}")

# 2) Build Wan pipeline (single DiT to reduce OOM risk)
MODEL_DIR = exp_raw.get("model_dir", exp_raw.get("model_root"))
if MODEL_DIR is None:
    raise ValueError("Please set model_dir (or model_root) in YAML experiment config.")

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=DEVICE,
    model_configs=[
        ModelConfig(path=[f"{MODEL_DIR}/high_noise_model/diffusion_pytorch_model-0000{i}-of-00006.safetensors" for i in range(1, 7)], offload_device="cpu"),
        ModelConfig(path=[f"{MODEL_DIR}/low_noise_model/diffusion_pytorch_model-0000{i}-of-00006.safetensors" for i in range(1, 7)], offload_device="cpu"),
        ModelConfig(path=f"{MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(path=f"{MODEL_DIR}/Wan2.1_VAE.pth", offload_device="cpu"),
    ],
    tokenizer_config=ModelConfig(path=f"{MODEL_DIR}/google/umt5-xxl"),
)

dummy_input_image = Image.new("RGB", (832, 480), color=(127, 127, 127))

# 3) Test cases
# Intended ablation:
#   A: action + observation + history
#   B: action + observation + history + masked trajectory image
video_no_traj = run_pipe_case(
    pipe,
    name="action+obs+history",
    input_image=dummy_input_image,
    action_tokens=encoded_no_traj.action_tokens,
    visual_latent=encoded_no_traj.visual_latent,
    use_preencoded=True,
)

# video_with_traj = run_pipe_case(
#     pipe,
#     name="action+obs+history+masked_traj",
#     input_image=dummy_input_image,
#     action_tokens=encoded_with_traj.action_tokens,
#     visual_latent=encoded_with_traj.visual_latent,
#     use_preencoded=True,
# )

# # Optional baseline: internal condition path (no preencoded).
# # This is not part of the masked trajectory ablation; it only checks whether
# # the preencoded path differs from the pipeline's default internal path.
# video_baseline = run_pipe_case(
#     pipe,
#     name="baseline_internal_condition",
#     input_image=dummy_input_image,
#     action_tokens=None,
#     visual_latent=None,
#     use_preencoded=False,
# )

# delta_traj = abs(frame_checksum(video_with_traj) - frame_checksum(video_no_traj))
# delta_with_traj_vs_base = abs(frame_checksum(video_with_traj) - frame_checksum(video_baseline))
# delta_no_traj_vs_base = abs(frame_checksum(video_no_traj) - frame_checksum(video_baseline))

# print(f"[Result] checksum delta(with_traj vs no_traj)={delta_traj:.6f}")
# print(f"[Result] checksum delta(with_traj vs baseline)={delta_with_traj_vs_base:.6f}")
# print(f"[Result] checksum delta(no_traj vs baseline)={delta_no_traj_vs_base:.6f}")
# print("[PASS] Wan DiT masked trajectory ablation forward test finished.")
