import torch
from transformers import AutoProcessor, SiglipVisionModel
from diffsynth.core import ModelConfig
from diffsynth.models.action_conditioning import (
    ActionConditioningConfig,
    ConditionStreamConfig,
)
from diffsynth.pipelines.action_video_pipeline import ActionVideoPipeline
# -----------------------------
# 0) Device / dtype
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
# -----------------------------
# 1) 原始输入
# -----------------------------
actions = torch.randn(1, 20, 7)          # (B,T,A)
obs_image = torch.randn(1, 3, 224, 224)  # (B,3,H,W), 示例随机
# -----------------------------
# 2) 用 SigLIP 生成 obs / masked embeddings
# -----------------------------
siglip_id = "google/siglip-so400m-patch14-224"
processor = AutoProcessor.from_pretrained(siglip_id)
vision = SiglipVisionModel.from_pretrained(siglip_id).to(device=device, dtype=torch_dtype).eval()
def to_pil_batch(images_bchw: torch.Tensor):
    # 输入可为 [0,1] 或 [0,255]
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