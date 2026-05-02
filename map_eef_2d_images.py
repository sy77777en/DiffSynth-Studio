#!/usr/bin/env python3
"""
Generate BOTH:
1) Overlay video (EEF projected on original frames), saved next to input video.
2) Masked images in cam_<cam_id>/images/, named masked_xxxxxx.png.

Input:
    --video .../RH20T_cfg5/task_xxx/.../cam_<cam_id>/color.mp4
Outputs:
    - .../cam_<cam_id>/<video_stem>_mapped.mp4
    - .../cam_<cam_id>/images/masked_xxxxxx.png
"""

import os
import re
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


CALIB_W = 1280
CALIB_H = 720
GRIPPER_MAX_DEFAULT = 80.0


# ─────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────
def infer_scene_and_cam_from_video(video_path: str):
    video_path = os.path.abspath(video_path)
    p = Path(video_path)
    if p.name != "color.mp4":
        raise ValueError(f"Expected 'color.mp4', got: {p.name}")
    cam_folder = p.parent
    scene_root = cam_folder.parent
    cfg_root = scene_root.parent
    m = re.match(r"cam_(\d+)", cam_folder.name)
    if m is None:
        raise ValueError(f"Cannot parse cam_id from: {cam_folder.name}")
    cam_id = m.group(1)
    return str(cfg_root), str(scene_root), cam_id, str(cam_folder)


def load_scene_metadata(scene_root: str):
    meta_path = os.path.join(scene_root, "metadata.json")
    with open(meta_path, "r") as f:
        return json.load(f)


def resolve_calib_dir(scene_root: str):
    metadata = load_scene_metadata(scene_root)
    calib_id = str(metadata["calib"])
    cfg_root = os.path.dirname(scene_root)
    calib_dir = os.path.join(cfg_root, "calib", calib_id)
    return calib_dir, calib_id


def resolve_tcp_path(scene_root: str):
    return os.path.join(scene_root, "transformed", "tcp.npy")


def resolve_gripper_path(scene_root: str):
    return os.path.join(scene_root, "transformed", "gripper.npy")


# ─────────────────────────────────────────────────────────────
# Camera loading (intrinsics only — tcp is already in cam frame)
# ─────────────────────────────────────────────────────────────
def load_intrinsics(calib_dir: str, cam_id: str, image_width: int, image_height: int):
    intr_path = os.path.join(calib_dir, "intrinsics.npy")
    intr_dict = np.load(intr_path, allow_pickle=True).item()
    if cam_id not in intr_dict:
        raise KeyError(f"cam_id '{cam_id}' not in intrinsics. Available: {list(intr_dict.keys())}")
    K = np.asarray(intr_dict[cam_id], dtype=np.float64)[:3, :3].copy()
    sx = image_width / CALIB_W
    sy = image_height / CALIB_H
    K[0, 0] *= sx;  K[0, 2] *= sx
    K[1, 1] *= sy;  K[1, 2] *= sy
    return K


# ─────────────────────────────────────────────────────────────
# Timestamp & TCP loading
# ─────────────────────────────────────────────────────────────
def load_video_timestamps(scene_root: str, cam_id: str):
    ts_path = os.path.join(scene_root, f"cam_{cam_id}", "timestamps.npy")
    data = np.load(ts_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()
    if isinstance(data, dict):
        ts = np.asarray(data.get("color", data[next(iter(data.keys()))]))
    else:
        ts = np.asarray(data)
    return ts.astype(np.int64)


def load_tcp(tcp_path: str, cam_id: str):
    data = np.load(tcp_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()
    return data[cam_id]


def build_tcp_timestamp_arrays(tcp_list):
    tcp_timestamps = []
    tcp_values = []
    for item in tcp_list:
        tcp_timestamps.append(int(item["timestamp"]))
        tcp_values.append(np.asarray(item["tcp"], dtype=np.float64))
    return np.asarray(tcp_timestamps, dtype=np.int64), np.asarray(tcp_values, dtype=np.float64)


# ─────────────────────────────────────────────────────────────
# Gripper loading
# ─────────────────────────────────────────────────────────────
def load_gripper(gripper_path: str, cam_id: str):
    """
    gripper.npy structure: {cam_id: {timestamp: {"gripper_info": [val, ...], ...}, ...}}
    Returns sorted arrays: (timestamps, gripper_values) where gripper_values = gripper_info[0].
    """
    data = np.load(gripper_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()
    if cam_id not in data:
        raise KeyError(f"cam_id '{cam_id}' not in gripper.npy. Available: {list(data.keys())}")

    cam_data = data[cam_id]
    timestamps = sorted(cam_data.keys())
    grip_ts = np.array(timestamps, dtype=np.int64)
    grip_vals = np.array([float(cam_data[t]["gripper_info"][0]) for t in timestamps],
                         dtype=np.float64)
    return grip_ts, grip_vals


def nearest_index(sorted_array: np.ndarray, value: int):
    pos = np.searchsorted(sorted_array, value)
    if pos == 0:
        return 0
    if pos >= len(sorted_array):
        return len(sorted_array) - 1
    before = pos - 1
    after = pos
    return after if abs(sorted_array[after] - value) < abs(sorted_array[before] - value) else before


# ─────────────────────────────────────────────────────────────
# Color from gripper openness
# ─────────────────────────────────────────────────────────────
def gripper_to_color_bgr(gripper_val: float, gripper_max: float) -> tuple:
    """
    Map gripper openness to BGR color.
    0 (closed) -> green (0, 200, 0)
    1 (fully open) -> red (0, 0, 255)
    Linear interpolation in between.
    """
    t = np.clip(gripper_val / gripper_max, 0.0, 1.0)
    b = 0
    g = int(round(200 * (1.0 - t)))
    r = int(round(255 * t))
    return (b, g, r)


# ─────────────────────────────────────────────────────────────
# Draw EEF on black mask
# ─────────────────────────────────────────────────────────────
def draw_eef_mask(H, W, tcp, K, circle_color_bgr,
                  axis_length=0.05, radius=50, tool_offset=0.0):
    """Draw EEF circle + axis on a black (H, W, 3) image."""
    mask = np.zeros((H, W, 3), dtype=np.uint8)

    xyz = tcp[:3]
    quat_wxyz = tcp[3:7]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    R = Rotation.from_quat(quat_xyzw).as_matrix()

    xyz = xyz + R @ np.array([0.0, 0.0, tool_offset])

    pts_cam = np.array([
        xyz,
        xyz + R @ np.array([axis_length, 0.0, 0.0]),
        xyz + R @ np.array([0.0, axis_length, 0.0]),
        xyz + R @ np.array([0.0, 0.0, axis_length]),
    ])

    uvs, valid = [], []
    for p in pts_cam:
        if p[2] <= 1e-6:
            uvs.append((0, 0)); valid.append(False); continue
        uv = K @ p
        uv = uv[:2] / uv[2]
        x, y = float(uv[0]), float(uv[1])
        uvs.append((int(round(x)), int(round(y))))
        valid.append(0 <= x < W and 0 <= y < H)

    if not valid[0]:
        return mask

    ox, oy = uvs[0]
    cv2.circle(mask, (ox, oy), radius, circle_color_bgr, -1)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # x-red, y-green, z-blue
    for j in range(1, 4):
        if valid[j]:
            cv2.line(mask, (ox, oy), uvs[j], colors[j - 1], 3)

    return mask


# ─────────────────────────────────────────────────────────────
# Draw EEF on original frame (for overlay video)
# ─────────────────────────────────────────────────────────────
def draw_eef_overlay(frame, tcp, K, circle_color_bgr,
                     axis_length=0.05, radius=8, tool_offset=0.0):
    H, W = frame.shape[:2]
    img = frame.copy()

    xyz = tcp[:3]
    quat_wxyz = tcp[3:7]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    R = Rotation.from_quat(quat_xyzw).as_matrix()

    xyz = xyz + R @ np.array([0.0, 0.0, tool_offset])

    pts_cam = np.array([
        xyz,
        xyz + R @ np.array([axis_length, 0.0, 0.0]),
        xyz + R @ np.array([0.0, axis_length, 0.0]),
        xyz + R @ np.array([0.0, 0.0, axis_length]),
    ])

    uvs, valid = [], []
    for p in pts_cam:
        if p[2] <= 1e-6:
            uvs.append((0, 0)); valid.append(False); continue
        uv = K @ p
        uv = uv[:2] / uv[2]
        x, y = float(uv[0]), float(uv[1])
        uvs.append((int(round(x)), int(round(y))))
        valid.append(0 <= x < W and 0 <= y < H)

    if not valid[0]:
        return img

    ox, oy = uvs[0]
    cv2.circle(img, (ox, oy), radius, circle_color_bgr, -1)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # x-red, y-green, z-blue
    for j in range(1, 4):
        if valid[j]:
            cv2.line(img, (ox, oy), uvs[j], colors[j - 1], 3)

    return img


def process_single_video(
    video_path: str,
    radius: int,
    axis_length: float,
    tool_offset: float,
    gripper_max: float,
    output_video: str = None,
    no_video: bool = False,
    no_masks: bool = False,
    show_frame_progress: bool = False,
):
    cfg_root, scene_root, cam_id, cam_folder = infer_scene_and_cam_from_video(video_path)
    calib_dir, calib_id = resolve_calib_dir(scene_root)
    tcp_path = resolve_tcp_path(scene_root)
    gripper_path = resolve_gripper_path(scene_root)

    images_dir = os.path.join(cam_folder, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images/ dir not found: {images_dir}")

    print(f"[INFO] scene_root   = {scene_root}")
    print(f"[INFO] cam_id       = {cam_id}")
    print(f"[INFO] images_dir   = {images_dir}")
    print(f"[INFO] gripper_max  = {gripper_max}")

    # Load data
    video_timestamps = load_video_timestamps(scene_root, cam_id)
    tcp_list = load_tcp(tcp_path, cam_id)
    tcp_timestamps, tcp_values = build_tcp_timestamp_arrays(tcp_list)
    grip_timestamps, grip_values = load_gripper(gripper_path, cam_id)

    print(f"[INFO] num video timestamps = {len(video_timestamps)}")
    print(f"[INFO] num tcp entries       = {len(tcp_timestamps)}")
    print(f"[INFO] num gripper entries   = {len(grip_timestamps)}")
    print(f"[INFO] gripper range         = [{grip_values.min():.4f}, {grip_values.max():.4f}]")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video size = {W}x{H}, fps={fps:.2f}, frames={total_frames}")

    K = load_intrinsics(calib_dir, cam_id, W, H)

    if len(video_timestamps) != total_frames:
        print(
            f"[WARN] timestamps ({len(video_timestamps)}) != frames ({total_frames}), "
            "using min(frame_idx, len(timestamps)-1)."
        )

    # Existing frame list (for deciding which masks to write)
    frame_files = sorted([
        f for f in os.listdir(images_dir)
        if re.match(r"frame_(\d+)\.png", f)
    ])
    frame_indices_for_masks = set()
    for f in frame_files:
        m = re.match(r"frame_(\d+)\.png", f)
        if m is not None:
            frame_indices_for_masks.add(int(m.group(1)))
    print(f"[INFO] Found {len(frame_indices_for_masks)} frame_xxxxxx.png in images/")

    # Setup output video path (default next to input video)
    if output_video is None:
        vp = Path(video_path)
        output_video = str(vp.with_name(f"{vp.stem}_mapped.mp4"))

    out = None
    if not no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, fps, (W, H))
        if not out.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {output_video}")
        print(f"[INFO] Overlay video will be written to: {output_video}")

    print(f"[INFO] radius={radius}, axis_length={axis_length}")
    masks_written = 0
    frame_idx = 0

    pbar = None
    if show_frame_progress:
        pbar = tqdm(total=total_frames, desc=f"Frames cam_{cam_id}", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts_idx = min(frame_idx, len(video_timestamps) - 1)
        frame_ts = int(video_timestamps[ts_idx])

        tcp_idx = nearest_index(tcp_timestamps, frame_ts)
        tcp = tcp_values[tcp_idx]

        grip_idx = nearest_index(grip_timestamps, frame_ts)
        grip_val = grip_values[grip_idx]
        color_bgr = gripper_to_color_bgr(grip_val, gripper_max)

        if not no_video and out is not None:
            overlay = draw_eef_overlay(
                frame=frame,
                tcp=tcp,
                K=K,
                circle_color_bgr=color_bgr,
                axis_length=axis_length,
                radius=max(1, radius // 6),
                tool_offset=tool_offset,
            )
            out.write(overlay)

        if (not no_masks) and (frame_idx in frame_indices_for_masks):
            mask = draw_eef_mask(
                H, W, tcp, K,
                circle_color_bgr=color_bgr,
                axis_length=axis_length,
                radius=radius,
                tool_offset=tool_offset,
            )
            out_name = f"masked_{frame_idx:06d}.png"
            out_path = os.path.join(images_dir, out_name)
            cv2.imwrite(out_path, mask)
            masks_written += 1

        frame_idx += 1
        if pbar is not None:
            pbar.update(1)
        elif frame_idx % 50 == 0 or frame_idx == total_frames:
            print(f"[INFO] processed {frame_idx}/{total_frames}")

    if pbar is not None:
        pbar.close()

    cap.release()
    if out is not None:
        out.release()
        print(f"[INFO] Saved overlay video: {output_video}")

    if not no_masks:
        print(f"[INFO] Generated {masks_written} masked images in {images_dir}")


def find_task_video_paths(cfg_root: str, cam_folder_name: str):
    cfg_root = os.path.abspath(cfg_root)
    task_dirs = sorted([
        os.path.join(cfg_root, d)
        for d in os.listdir(cfg_root)
        if (
            d.startswith("task_")
            and ("human" not in d.lower())
            and os.path.isdir(os.path.join(cfg_root, d))
        )
    ])

    video_paths = []
    for task_dir in task_dirs:
        video_path = os.path.join(task_dir, cam_folder_name, "color.mp4")
        if os.path.exists(video_path):
            video_paths.append(video_path)
    return video_paths


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--video",
        default=None,
        help="Path to one color.mp4 under RH20T scene",
    )
    parser.add_argument(
        "--cfg_root",
        default=None,
        help="RH20T cfg root, e.g. /path/to/RH20T_cfg5",
    )
    parser.add_argument(
        "--cam",
        default=None,
        help="Camera folder name or id, e.g. cam_036422060215 or 036422060215",
    )
    parser.add_argument("--radius", type=int, default=50,
                        help="Circle radius in pixels (default 50)")
    parser.add_argument("--axis_length", type=float, default=0.05,
                        help="Axis length in meters (default 0.05)")
    parser.add_argument("--tool_offset", type=float, default=0.0,
                        help="Offset along local z-axis (default 0.0)")
    parser.add_argument("--gripper_max", type=float, default=GRIPPER_MAX_DEFAULT,
                        help="Max gripper opening value for Franka Panda (default 80)")
    parser.add_argument(
        "--output_video",
        default=None,
        help="Overlay video output path. Default: next to input video as <name>_mapped.mp4",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Do not generate overlay video.",
    )
    parser.add_argument(
        "--no_masks",
        action="store_true",
        help="Do not generate masked images.",
    )
    parser.add_argument(
        "--show_frame_progress",
        action="store_true",
        help="Show tqdm progress bar for frames.",
    )
    args = parser.parse_args()

    if args.video is None and args.cfg_root is None:
        raise ValueError("Provide either --video or --cfg_root.")
    if args.video is not None and args.cfg_root is not None:
        raise ValueError("Use only one mode: --video OR --cfg_root.")

    if args.video is not None:
        process_single_video(
            video_path=args.video,
            radius=args.radius,
            axis_length=args.axis_length,
            tool_offset=args.tool_offset,
            gripper_max=args.gripper_max,
            output_video=args.output_video,
            no_video=args.no_video,
            no_masks=args.no_masks,
            show_frame_progress=args.show_frame_progress,
        )
        return

    if args.cam is None:
        raise ValueError("When using --cfg_root mode, --cam is required.")

    cam_folder_name = args.cam if args.cam.startswith("cam_") else f"cam_{args.cam}"
    video_paths = find_task_video_paths(args.cfg_root, cam_folder_name)
    if not video_paths:
        raise ValueError(f"No matching videos found under {args.cfg_root} for {cam_folder_name}")

    print(f"[INFO] Found {len(video_paths)} tasks with {cam_folder_name}/color.mp4")

    success = 0
    failed = 0
    for video_path in tqdm(video_paths, desc=f"Tasks {cam_folder_name}"):
        try:
            process_single_video(
                video_path=video_path,
                radius=args.radius,
                axis_length=args.axis_length,
                tool_offset=args.tool_offset,
                gripper_max=args.gripper_max,
                output_video=None,  # per-task default: next to color.mp4
                no_video=args.no_video,
                no_masks=args.no_masks,
                show_frame_progress=args.show_frame_progress,
            )
            success += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] Failed: {video_path}")
            print(f"        {type(e).__name__}: {e}")

    print(f"[INFO] Batch finished. success={success}, failed={failed}")


if __name__ == "__main__":
    main()