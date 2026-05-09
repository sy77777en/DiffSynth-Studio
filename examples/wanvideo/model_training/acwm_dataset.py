"""
ACWMDataset — Action-Conditioned World Model training dataset

Reads `train_metadata.json` produced by prepare_training_data.py.

Each sample returns:
  - "video":          list[PIL.Image], length 17 (1 obs + 16 targets)
  - "obs_image":      PIL.Image
  - "actions":        torch.Tensor, shape (16, 7)
  - "prompt":         ""

Usage:
    from acwm_dataset import ACWMDataset
    ds = ACWMDataset("/path/to/train_metadata.json", height=368, width=640)
    sample = ds[0]
"""

import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ACWMDataset(Dataset):
    """
    Fixed specification:
      - 1 observation frame
      - 16 target frames
      - 16 action vectors (each is 7D)
      - Images are resized to (height, width), both must be multiples of 16
    """

    def __init__(
        self,
        metadata_path: str,
        height: int = 368,
        width: int = 640,
        repeat: int = 1,
    ):
        with open(metadata_path, "r") as f:
            self.data = json.load(f)

        assert height % 32 == 0, f"height={height} must be a multiple of 32"
        assert width % 32 == 0, f"width={width} must be a multiple of 32"

        self.height = height
        self.width = width
        self.repeat = repeat

        # Validate the first sample format
        # s = self.data[0]
        # assert len(s["target_frames"]) >= 16, \
        #     f"Not enough target_frames: {len(s['target_frames'])}"
        # assert len(s["actions"]) >= 16, \
        #     f"Not enough actions: {len(s['actions'])}"

        print(
            f"[ACWMDataset] {len(self.data)} samples, repeat={repeat}, "
            f"effective={len(self)}, resize=({height}, {width})"
        )

    def __len__(self):
        return len(self.data) * self.repeat

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height), Image.LANCZOS)
        return img

    def __getitem__(self, idx):
        sample = self.data[idx % len(self.data)]

        obs_img = self._load_image(sample["obs_frame"])
        target_imgs = [
            self._load_image(p) for p in sample["target_frames"][:16]
        ]

        actions = torch.tensor(
            sample["actions"][:16], dtype=torch.float32
        )  # shape: (16, 7)

        # video = observation + 16 target frames = 17 frames total
        video = [obs_img] + target_imgs

        result = {
            "video": video,
            "obs_image": obs_img,
            "actions": actions,
            "prompt": "",
        }

        if "target_masked" in sample:
            result["masked_traj] = [
                self._load_image(p) for p in sample["masked_traj"][:16]
            ]

        return result


# ======================================================================
# Test script — run:
#   python acwm_dataset.py /path/to/train_metadata.json
# ======================================================================
if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage: python acwm_dataset.py <train_metadata.json> [height] [width]")
        sys.exit(1)

    metadata_path = sys.argv[1]
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 368
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 640

    ds = ACWMDataset(metadata_path, height=height, width=width, repeat=1)

    print("\n--- Testing first 3 samples ---")
    for i in range(min(3, len(ds))):
        t0 = time.time()
        sample = ds[i]
        dt = time.time() - t0

        video = sample["video"]
        obs = sample["obs_image"]
        actions = sample["actions"]

        print(f"\nSample {i} (load time: {dt:.2f}s):")
        print(f"  video:    {len(video)} frames, first frame size={video[0].size}")
        print(f"  obs:      size={obs.size}")
        print(f"  actions:  shape={actions.shape}, dtype={actions.dtype}")
        print(f"  actions[0]: {actions[0].tolist()}")
        print(f"  prompt:   '{sample['prompt']}'")

    # DataLoader test
    print("\n--- DataLoader test (batch_size=1, 4 workers) ---")
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        # Do not stack since video/images are lists of PIL images
        return batch[0]

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= 5:
            break
    dt = time.time() - t0

    print(f"  5 batches loaded in {dt:.2f}s ({dt/5:.2f}s per batch)")
    print("\n[PASS] Dataset test passed")
