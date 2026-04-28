import torch
from transformers import AutoProcessor, SiglipVisionModel

from diffsynth.core import ModelConfig
from diffsynth.models.action_conditioning import (
    ActionConditioningConfig,
    ConditionStreamConfig,
)
from diffsynth.pipelines.action_video_pipeline import ActionVideoPipeline


def p(name, x):
    if x is None:
        print(f"[Shape] {name}: None")
        return
    if isinstance(x, torch.Tensor):
        print(f"[Shape] {name}: {tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
    else:
        print(f"[Shape] {name}: type={type(x)}")


# -----------------------------
# 0) Device / dtype
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"[Info] device={device}, torch_dtype={torch_dtype}")

# -----------------------------
# 1) 原始输入
# -----------------------------
actions = torch.randn(1, 20, 7)
obs_image = torch.randn(1, 3, 224, 224)
p("actions(raw)", actions)
p("obs_image(raw)", obs_image)

# -----------------------------
# 2) SigLIP
# -----------------------------
siglip_id = "google/siglip-so400m-patch14-224"
print(f"[Info] Loading SigLIP: {siglip_id}")
processor = AutoProcessor.from_pretrained(siglip_id)
vision = SiglipVisionModel.from_pretrained(siglip_id).to(device=device, dtype=torch_dtype).eval()
print(f"[Info] SigLIP hidden_size={vision.config.hidden_size}")


def to_pil_batch(images_bchw: torch.Tensor):
    x = images_bchw.detach().cpu()
    if x.max() > 1.5:
        x = x / 255.0
    x = x.clamp(0, 1)
    pil_images = []
    for i in range(x.shape[0]):
        arr = (x[i] * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        from PIL import Image
        pil_images.append(Image.fromarray(arr))
    return pil_images


@torch.no_grad()
def encode_obs_with_siglip(obs_bchw: torch.Tensor) -> torch.Tensor:
    p("obs_bchw(in)", obs_bchw)
    pil = to_pil_batch(obs_bchw)
    print(f"[Shape] obs PIL batch len: {len(pil)}")
    inputs = processor(images=pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch_dtype)
    p("obs pixel_values", pixel_values)
    out = vision(pixel_values=pixel_values)
    p("obs last_hidden_state", out.last_hidden_state)
    return out.last_hidden_state  # (B, N_obs, D_obs)


@torch.no_grad()
def encode_masked_seq_with_siglip(masked_btchw: torch.Tensor) -> torch.Tensor:
    p("masked_btchw(in)", masked_btchw)
    b, t, c, h, w = masked_btchw.shape
    flat = masked_btchw.reshape(b * t, c, h, w)
    p("masked flat(B*T,C,H,W)", flat)
    pil = to_pil_batch(flat)
    print(f"[Shape] masked PIL batch len: {len(pil)}")
    inputs = processor(images=pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch_dtype)
    p("masked pixel_values", pixel_values)
    tokens = vision(pixel_values=pixel_values).last_hidden_state
    p("masked tokens(B*T,N,D)", tokens)
    seq = tokens.view(b, t, tokens.shape[1], tokens.shape[2])
    p("masked_image_emb_seq(B,T,N,D)", seq)
    return seq


obs_image_emb = encode_obs_with_siglip(obs_image)
siglip_dim = obs_image_emb.shape[-1]

# 这里先随机模拟 action map 之后的 masked image seq
masked_image_seq = torch.randn(1, 20, 3, 224, 224)
p("masked_image_seq(raw)", masked_image_seq)
masked_image_emb_seq = encode_masked_seq_with_siglip(masked_image_seq)

# -----------------------------
# 3) 配置
# -----------------------------
cfg = ActionConditioningConfig(
    backbone="wan",
    action_dim=7,
    condition_context_dim=4096,
    use_text=False,
    require_frame_alignment=True,
    action=ConditionStreamConfig(
        injection_type="cross_attn",
        encoder_type="perceiver",
        embed_dim=1024,
        num_queries=8,
        enabled=True,
    ),
    obs_image=ConditionStreamConfig(
        injection_type="cross_attn",
        encoder_type="identity",
        embed_dim=siglip_dim,
        num_queries=1,
        enabled=True,
    ),
    masked_image=ConditionStreamConfig(
        injection_type="cross_attn",
        encoder_type="identity",
        embed_dim=siglip_dim,
        num_queries=8,
        enabled=True,
    ),
)
print("[Info] Config ready.")
print(f"[Info] action_dim={cfg.action_dim}, condition_context_dim={cfg.condition_context_dim}")
print(f"[Info] obs_embed_dim={cfg.obs_image.embed_dim}, masked_embed_dim={cfg.masked_image.embed_dim}")

# -----------------------------
# 4) 构建 pipeline
# -----------------------------
model_configs = [
    # 你自己填：至少 dit + vae
    # ModelConfig(path="...wan_video_dit..."),
    # ModelConfig(path="...wan_video_vae..."),
]

pipe = ActionVideoPipeline.from_pretrained(
    action_conditioning_config=cfg,
    model_configs=model_configs,
    torch_dtype=torch_dtype,
    device=device,
)
print("[Info] Pipeline built.")
print(f"[Info] pipe.device={pipe.device}, pipe.torch_dtype={pipe.torch_dtype}")

# -----------------------------
# 5) 喂入前最后检查
# -----------------------------
actions_in = actions.to(pipe.device, dtype=pipe.torch_dtype)
obs_emb_in = obs_image_emb.to(pipe.device, dtype=pipe.torch_dtype)
masked_emb_in = masked_image_emb_seq.to(pipe.device, dtype=pipe.torch_dtype)

p("actions(in_pipe)", actions_in)
p("obs_image_emb(in_pipe)", obs_emb_in)
p("masked_image_emb_seq(in_pipe)", masked_emb_in)

print("[Check] T_action:", actions_in.shape[1])
print("[Check] T_masked:", masked_emb_in.shape[1])
print("[Check] num_frames:", 20)

# -----------------------------
# 6) 运行
# -----------------------------
video = pipe(
    actions=actions_in,
    obs_image_emb=obs_emb_in,
    masked_image_emb_seq=masked_emb_in,
    num_frames=20,
    height=480,
    width=832,
    num_inference_steps=30,
    seed=123,
    output_type="quantized",
)
print("[Done] type(video)=", type(video), "len(video)=", len(video))