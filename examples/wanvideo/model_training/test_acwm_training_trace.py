#!/usr/bin/env python3
"""
一次性看清 ACWM 训练在代码里到底做了什么。

镜像 train_acwm.ACWMTrainingModule.forward 的路径：
  Dataset batch → get_pipeline_inputs → _encode_acwm_conditions
  → 依次跑 pipe.units → FlowMatchSFTLoss

模型权重（三选一，与 wan_acwm_finetune.sh 一致）:

  A) 自动拼路径（推荐，把你刚才那种「目录」用在 checkpoint_root 上）:

    --wan_checkpoint_root /path/to/Wan2.2-I2V-A14B \\
    --wan_noise_subdir high_noise_model

  B) JSON 文件（内容为一层数组: [ [shard1,...], "t5.pth", "vae.pth" ]）:

    --model_paths_file ./my_model_paths.json

  C) 命令行里直接写 **JSON 字符串**（注意引号）:

    --model_paths '[[\"/.../00001-of-00006.safetensors\",...],\"/.../t5.pth\",\"/.../VAE.pth\"]'

  不要把模型根目录传给 --model_paths（会触发 JSONDecodeError）。

完整示例:

  cd DiffSynth-Studio
  python examples/wanvideo/model_training/test_acwm_training_trace.py \\
    --dataset_metadata_path train_metadata_100.json \\
    --height 368 --width 640 --num_frames 17 \\
    --wan_checkpoint_root /path/to/Wan2.2-I2V-A14B \\
    --wan_noise_subdir high_noise_model \\
    --acwm_config configs/action_conditioning.yaml \\
    --max_timestep_boundary 0.417 --min_timestep_boundary 0 \\
    --lora_base_model dit --lora_rank 16 \\
    --extra_inputs input_image \\
    --sample-index 0 \\
    --seed 0

可选：
  --backward     在打印 loss 后再做一次 backward（耗显存，仅用于确认梯度连通）
  --no-trace-units  不逐个 unit 打印（仅看首尾与 loss）
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_MT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_MT_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _MT_DIR not in sys.path:
    sys.path.insert(0, _MT_DIR)

import torch

from train_acwm import ACWMTrainingModule, acwm_train_parser
from acwm_dataset import ACWMDataset


def build_model_paths_json(model_dir: str, noise_subdir: str) -> str:
    """
    与 wan_acwm_finetune.sh 里 MODEL_PATHS_JSON 相同结构:
    [ DiT shard 路径列表, T5 路径, VAE 路径 ]
    """
    noise_dir = os.path.join(model_dir, noise_subdir)
    pattern = os.path.join(noise_dir, "diffusion_pytorch_model*.safetensors")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise ValueError(
            f"在 {noise_dir!r} 下未找到 diffusion_pytorch_model*.safetensors，"
            f"请检查 --wan_checkpoint_root 与 --wan_noise_subdir。"
        )
    t5 = os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    if not os.path.isfile(t5):
        raise FileNotFoundError(f"未找到 T5: {t5}")
    vae_path = None
    for name in ("Wan2.1_VAE.pth", "Wan2.2_VAE.pth"):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            vae_path = p
            break
    if vae_path is None:
        raise FileNotFoundError(
            f"在 {model_dir!r} 下未找到 Wan2.1_VAE.pth 或 Wan2.2_VAE.pth"
        )
    return json.dumps([shards, t5, vae_path])


def resolve_model_paths(args: argparse.Namespace) -> str | None:
    """得到传给 WanTrainingModule 的 JSON 字符串；若无需加载则返回 None。"""
    if getattr(args, "wan_checkpoint_root", None):
        return build_model_paths_json(args.wan_checkpoint_root, args.wan_noise_subdir)
    if getattr(args, "model_paths_file", None):
        path = args.model_paths_file
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--model_paths_file 不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        json.loads(raw)  # validate
        return raw
    mp = args.model_paths
    if not mp:
        return None
    if os.path.isdir(mp):
        raise SystemExit(
            "\n错误: 你把模型目录传给了 --model_paths。\n"
            "此处需要的是 **JSON 数组字符串**，不是文件夹路径。\n\n"
            "请改用其一:\n"
            "  --wan_checkpoint_root \"" + mp + "\" --wan_noise_subdir high_noise_model\n"
            "或把 JSON 放进文件后用 --model_paths_file path/to.json\n"
            "或与 wan_acwm_finetune.sh 里 MODEL_PATHS_JSON 相同的单行 JSON。\n"
        )
    # 若用户把「json 文件路径」误传给 --model_paths，顺带兼容
    if os.path.isfile(mp) and mp.lower().endswith(".json"):
        with open(mp, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        json.loads(raw)
        return raw
    try:
        json.loads(mp)
    except json.JSONDecodeError as e:
        raise SystemExit(
            "\n错误: --model_paths 必须是合法 JSON（数组），解析失败:\n"
            f"  {e}\n\n"
            "示例请见本文件顶部文档；推荐使用:\n"
            "  --wan_checkpoint_root /path/to/Wan2.2-I2V-A14B --wan_noise_subdir high_noise_model\n"
        ) from e
    return mp


def _summarize_tensor(x, name: str, max_el: int = 4) -> str:
    if x is None:
        return f"  {name}: None"
    if torch.is_tensor(x):
        flat = x.detach().flatten()
        n = min(flat.numel(), max_el)
        preview = flat[:n].tolist()
        return (
            f"  {name}: Tensor shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
            f"preview[:{n}]={preview}"
        )
    return f"  {name}: {type(x).__name__} {repr(x)[:120]}"


def _print_dict_tensors(d: dict, title: str, keys: list[str] | None = None):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    if keys is None:
        keys = sorted(d.keys())
    for k in keys:
        if k not in d:
            continue
        v = d[k]
        if torch.is_tensor(v):
            print(_summarize_tensor(v, k))
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            print(f"  {k}: list[len={len(v)}] of tensors, first shape={tuple(v[0].shape)}")
        elif isinstance(v, list) and v:
            print(f"  {k}: list[len={len(v)}], first_elem_type={type(v[0]).__name__}")
        else:
            s = repr(v)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"  {k}: {s}")


def trace_training_forward(
    model: ACWMTrainingModule,
    data: dict,
    *,
    trace_units: bool,
    explain_loss: bool,
):
    """逐步执行与 ACWMTrainingModule.forward 相同逻辑，并打印。"""
    pipe = model.pipe

    print("\n" + "#" * 72)
    print("# STEP 0: 原始 data batch（ACWMDataset __getitem__）")
    print("#" * 72)
    print(f"  keys: {list(data.keys())}")
    print(f"  video: {len(data['video'])} PIL frames")
    print(f"  obs_image: {data['obs_image'].size}")
    print(f"  history_images: {len(data['history_images'])} frames")
    print(f"  actions: {tuple(data['actions'].shape)}")
    print(f"  prompt: {data['prompt']!r}")

    print("\n" + "#" * 72)
    print("# STEP 1: get_pipeline_inputs（WanTrainingModule）→ (inputs_shared, posi, nega)")
    print("#" * 72)
    inputs = model.get_pipeline_inputs(data)
    inputs = model.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
    inputs_shared, inputs_posi, inputs_nega = inputs
    _print_dict_tensors(
        inputs_shared,
        "inputs_shared（编码条件之前）",
        [
            "height",
            "width",
            "num_frames",
            "cfg_scale",
            "max_timestep_boundary",
            "min_timestep_boundary",
            "input_video",
            "input_image",
        ],
    )
    print(f"  inputs_posi keys: {list(inputs_posi.keys())}")
    print(f"  inputs_nega keys: {list(inputs_nega.keys())}")

    print("\n" + "#" * 72)
    print("# STEP 2: _encode_acwm_conditions（ConditionEncoder → 预计算 visual / action）")
    print("#" * 72)
    inputs_shared = model._encode_acwm_conditions(data, inputs_shared)
    if inputs_shared.get("preencoded_visual_latent") is not None:
        y = inputs_shared["preencoded_visual_latent"]
        print(_summarize_tensor(y, "preencoded_visual_latent (作为 pipeline 的 y / concat 条件)"))
    if inputs_shared.get("preencoded_action_tokens") is not None:
        a = inputs_shared["preencoded_action_tokens"]
        print(_summarize_tensor(a, "preencoded_action_tokens（将拼进 text cross-attn context）"))
    print(f"  skip_condition_vae_encode = {inputs_shared.get('skip_condition_vae_encode')}")

    inputs = (inputs_shared, inputs_posi, inputs_nega)

    print("\n" + "#" * 72)
    print("# STEP 3: 依次执行 pipe.units（与真实训练相同顺序）")
    print("#" * 72)
    N = len(pipe.scheduler.timesteps)
    print(f"  len(pipe.scheduler.timesteps) = {N}  （FlowMatch 里 timestep 边界会乘这个数）")

    for i, unit in enumerate(pipe.units):
        name = type(unit).__name__
        keys_before = set(inputs[0].keys())
        inputs = model.pipe.unit_runner(unit, pipe, *inputs)
        keys_after = set(inputs[0].keys())
        added = sorted(keys_after - keys_before)
        if trace_units:
            extra = f"  |  inputs_shared 新增 key: {added}" if added else ""
            print(f"  [{i:02d}] {name}{extra}")

    inputs_shared, inputs_posi, inputs_nega = inputs

    print("\n" + "#" * 72)
    print("# STEP 4: units 之后、进入 FlowMatchSFTLoss 之前的核心张量")
    print("#" * 72)
    for key in (
        "input_latents",
        "latents",
        "y",
        "clip_feature",
        "first_frame_latents",
        "preencoded_action_tokens",
        "preencoded_visual_latent",
    ):
        if key in inputs_shared:
            print(_summarize_tensor(inputs_shared[key], key))

    if explain_loss:
        mx = float(inputs_shared.get("max_timestep_boundary", 1.0))
        mn = float(inputs_shared.get("min_timestep_boundary", 0.0))
        max_idx = int(mx * N)
        min_idx = int(mn * N)
        print("\n  FlowMatchSFTLoss 会用到的 timestep 索引区间（见 diffsynth/diffusion/loss.py）:")
        print(f"    max_timestep_boundary={mx} → max_idx = int({mx} * {N}) = {max_idx}")
        print(f"    min_timestep_boundary={mn} → min_idx = int({mn} * {N}) = {min_idx}")
        print(f"    timestep_id ~ Uniform{{ {min_idx}, ..., {max_idx - 1} }}  （上界开区间）")

    print("\n" + "#" * 72)
    print("# STEP 5: FlowMatchSFTLoss → 标量 loss（与 train 里 self.task=='sft' 一致）")
    print("#" * 72)
    loss = model.task_to_loss[model.task](pipe, *inputs)
    print(f"  task={model.task!r}")
    print(_summarize_tensor(loss, "loss"))
    return loss


def print_trainable_overview(model: ACWMTrainingModule):
    names = model.trainable_param_names()
    n_lora = sum(1 for n in names if "lora_" in n)
    n_ae = sum(1 for n in names if n.startswith("condition_encoder.action_encoder"))
    other = len(names) - n_lora - n_ae
    print("\n" + "=" * 72)
    print("可训练参数名概览（export_trainable_state_dict 会保存这些）")
    print("=" * 72)
    print(f"  含 'lora_' 的键数量: {n_lora}")
    print(f"  condition_encoder.action_encoder 键数量: {n_ae}")
    print(f"  其它可训练键数量: {other}")
    print(f"  总计可训练键数: {len(names)}")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = acwm_train_parser()
    parser.add_argument("--sample-index", type=int, default=0, help="ACWMDataset 样本下标")
    parser.add_argument("--seed", type=int, default=0, help="固定随机性（含 loss 里 timestep 采样）")
    parser.add_argument("--backward", action="store_true", help="再执行一次 loss.backward()（吃显存）")
    parser.add_argument("--no-trace-units", action="store_true", help="不打印每个 unit 一行")
    parser.add_argument(
        "--wan_checkpoint_root",
        type=str,
        default=None,
        help="Wan 模型根目录（含 high_noise_model/、T5、VAE）；自动拼成 --model_paths JSON",
    )
    parser.add_argument(
        "--wan_noise_subdir",
        type=str,
        default="high_noise_model",
        help="相对 checkpoint_root 的 DiT 子目录，如 high_noise_model 或 low_noise_model",
    )
    parser.add_argument(
        "--model_paths_file",
        type=str,
        default=None,
        help="含 model_paths 结构的 JSON 文件（数组），避免在 shell 里手写长 JSON",
    )
    args = parser.parse_args()

    if args.height is None or args.width is None:
        raise SystemExit("需要 --height 与 --width")
    if args.dataset_metadata_path is None:
        raise SystemExit("需要 --dataset_metadata_path")

    resolved_paths = resolve_model_paths(args)
    if resolved_paths is None and args.model_id_with_origin_paths is None:
        raise SystemExit(
            "需要指定模型来源之一: "
            "--wan_checkpoint_root，或 --model_paths_file，或 --model_paths '<JSON>'，"
            "或 --model_id_with_origin_paths"
        )
    if resolved_paths is not None:
        args.model_paths = resolved_paths

    torch.manual_seed(args.seed)

    dataset = ACWMDataset(
        args.dataset_metadata_path,
        height=args.height,
        width=args.width,
        repeat=args.dataset_repeat,
    )
    idx = args.sample_index % len(dataset)
    data = dataset[idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 72)
    print(f"设备: {device}（若要用 CPU 初始化大模型可加 train_acwm 里的 --initialize_model_on_cpu，此处未接）")
    print("=" * 72)

    model = ACWMTrainingModule(
        acwm_config_path=args.acwm_config,
        acwm_experiment=args.acwm_experiment,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device=device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model.train()

    print_trainable_overview(model)

    loss = trace_training_forward(
        model,
        data,
        trace_units=not args.no_trace_units,
        explain_loss=True,
    )

    if args.backward:
        print("\n" + "#" * 72)
        print("# OPTIONAL: backward（验证梯度是否贯通 LoRA + ActionEncoder）")
        print("#" * 72)
        loss.backward()
        nonzero = [
            (n, p.grad.norm().item())
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        ]
        print(f"  有梯度的参数个数: {len(nonzero)}")
        for n, gnorm in nonzero[:8]:
            print(f"    ||grad||={gnorm:.6e}  {n[:100]}")
        if len(nonzero) > 8:
            print(f"    ... 另有 {len(nonzero) - 8} 个")
        if not nonzero:
            print("  警告: 未找到非零 grad（可能图被截断或某分支未参与 loss）")

    print("\n" + "=" * 72)
    print("追踪结束。若要与真实训练完全一致，请继续用 accelerate launch train_acwm.py。")
    print("=" * 72)


if __name__ == "__main__":
    main()
