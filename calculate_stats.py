"""
统计 RH20T 数据集中所有 episode 的 delta action 归一化参数。

从 transformed/tcp.npy 和 transformed/gripper.npy 中读取数据，
只处理 cam_036422060215 这个 camera，计算:
  delta_action = [dx, dy, dz, droll, dpitch, dyaw, gripper]  (7维)

其中:
  - dx, dy, dz: 相邻帧 xyz 差值
  - droll, dpitch, dyaw: 相邻帧 quaternion 转 euler 后求差值
  - gripper: gripper_command[0] 的绝对值（不做 delta）

输出 mean, std, min, max, q01, q99 到 JSON 文件。

Usage:
    python compute_action_stats.py \
        --data_root /net/holy-isilon/ifs/rc_labs/ydu_lab/sycen/data/rh20t/RH20T_cfg5 \
        --cam_id 036422060215 \
        --output action_stats.json
"""

import os
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


CAM_ID = "036422060215"


def extract_tcp_array(tcp_cam):
    """从 tcp dict list 中提取 (T, 7) 的 numpy array。任何 entry 异常则返回 None。"""
    for entry in tcp_cam:
        val = entry['tcp']
        if val is None or not hasattr(val, '__len__') or len(val) != 7:
            return None
    return np.array([entry['tcp'] for entry in tcp_cam])


def extract_gripper_array(gripper_cam, tcp_cam):
    """从 gripper dict 中提取 (T,) 的 gripper 值，按 tcp 的 timestamp 顺序对齐。"""
    timestamps = [entry['timestamp'] for entry in tcp_cam]
    gripper_vals = []
    for ts in timestamps:
        if ts in gripper_cam:
            g = gripper_cam[ts]['gripper_command'][0]
        else:
            g = 0.0
        gripper_vals.append(g)
    return np.array(gripper_vals)


def compute_delta_actions(tcp_array, gripper_array):
    """
    计算 delta actions: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    
    tcp_array: (T, 7) — xyz(3) + quat(4)
    gripper_array: (T,)
    返回: (T-1, 7) — delta xyz(3) + delta euler(3) + gripper(1)
    """
    T = tcp_array.shape[0]
    if T < 2:
        return np.zeros((0, 7))

    # Delta xyz
    delta_xyz = tcp_array[1:, :3] - tcp_array[:-1, :3]  # (T-1, 3)

    # Delta rotation: quat → euler，然后求差
    # RH20T 的 quat 格式是 xyzw（scipy 默认也是 xyzw）
    rot_prev = Rotation.from_quat(tcp_array[:-1, 3:7])
    rot_curr = Rotation.from_quat(tcp_array[1:, 3:7])

    # 相对旋转: R_delta = R_curr * R_prev^{-1}
    rot_delta = rot_curr * rot_prev.inv()
    delta_euler = rot_delta.as_euler('xyz', degrees=False)  # (T-1, 3)

    # Normalize angles to [-pi, pi]
    delta_euler = np.mod(delta_euler + np.pi, 2 * np.pi) - np.pi

    # Gripper: 用当前帧的绝对值（不做 delta）
    gripper_vals = gripper_array[1:].reshape(-1, 1)  # (T-1, 1)

    # 拼接: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    delta_actions = np.concatenate([delta_xyz, delta_euler, gripper_vals], axis=1)  # (T-1, 7)
    return delta_actions


def process_episode(task_dir, cam_id, verbose=True):
    """处理单个 episode，返回 (N, 7) 的 delta actions 或 None。"""
    task_name = os.path.basename(task_dir)
    tcp_path = os.path.join(task_dir, "transformed", "tcp.npy")
    gripper_path = os.path.join(task_dir, "transformed", "gripper.npy")

    if not os.path.exists(tcp_path) or not os.path.exists(gripper_path):
        if verbose:
            print(f"  SKIP {task_name}: missing tcp.npy or gripper.npy")
        return None

    tcp_data = np.load(tcp_path, allow_pickle=True).item()
    gripper_data = np.load(gripper_path, allow_pickle=True).item()

    if cam_id not in tcp_data or cam_id not in gripper_data:
        if verbose:
            print(f"  SKIP {task_name}: cam {cam_id} not found")
        return None

    tcp_cam = tcp_data[cam_id]
    gripper_cam = gripper_data[cam_id]

    tcp_array = extract_tcp_array(tcp_cam)
    if tcp_array is None:
        if verbose:
            print(f"  SKIP {task_name}: tcp has None or invalid entries")
        return None

    gripper_array = extract_gripper_array(gripper_cam, tcp_cam)

    if len(tcp_array) < 2:
        if verbose:
            print(f"  SKIP {task_name}: too few frames ({len(tcp_array)})")
        return None

    delta_actions = compute_delta_actions(tcp_array, gripper_array)
    if verbose:
        print(f"  OK   {task_name}: {len(tcp_array)} frames -> {len(delta_actions)} deltas")
    return delta_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to RH20T_cfg5 directory")
    parser.add_argument("--cam_id", type=str, default=CAM_ID)
    parser.add_argument("--output", type=str, default="action_stats.json")
    args = parser.parse_args()

    # 找到所有 task 目录（排除 _human 后缀的）
    task_dirs = []
    for name in sorted(os.listdir(args.data_root)):
        full_path = os.path.join(args.data_root, name)
        if os.path.isdir(full_path) and not name.endswith("_human"):
            task_dirs.append(full_path)

    print(f"Found {len(task_dirs)} task directories (excluding _human)")

    # 收集所有 delta actions
    all_deltas = []
    skipped = 0
    for task_dir in tqdm(task_dirs, desc="Processing episodes"):
        delta = process_episode(task_dir, args.cam_id)
        if delta is not None and len(delta) > 0:
            all_deltas.append(delta)
        else:
            skipped += 1

    print(f"Processed {len(all_deltas)} episodes, skipped {skipped}")

    if len(all_deltas) == 0:
        print("ERROR: No valid episodes found!")
        return

    # 合并
    all_deltas = np.concatenate(all_deltas, axis=0)  # (N_total, 7)
    print(f"Total delta action samples: {all_deltas.shape[0]}")

    # 计算统计量
    dim_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

    stats = {
        "dim_names": dim_names,
        "num_samples": int(all_deltas.shape[0]),
        "num_episodes": len(task_dirs) - skipped,
        "mean": all_deltas.mean(axis=0).tolist(),
        "std": all_deltas.std(axis=0).tolist(),
        "min": all_deltas.min(axis=0).tolist(),
        "max": all_deltas.max(axis=0).tolist(),
        "q01": np.percentile(all_deltas, 1, axis=0).tolist(),
        "q99": np.percentile(all_deltas, 99, axis=0).tolist(),
        "median": np.median(all_deltas, axis=0).tolist(),
    }

    # 打印
    print("\n" + "=" * 60)
    print("Delta Action Statistics")
    print("=" * 60)
    for i, name in enumerate(dim_names):
        print(f"  {name:10s}: mean={stats['mean'][i]:+.6e}  std={stats['std'][i]:.6e}  "
              f"range=[{stats['min'][i]:+.4e}, {stats['max'][i]:+.4e}]  "
              f"q01={stats['q01'][i]:+.4e}  q99={stats['q99'][i]:+.4e}")

    # 保存
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()