from torch.utils.data import Dataset
import os
import imageio
import numpy as np
import torch

from .bracket_ops import ENCODINGS


class RGBImageDataset(Dataset):
    """Loads (target, guidance) pairs of RGB images.

    The CSV file lists one pair per line:
        target_rel_path,guidance_rel_path

    Both files store (H, W, 3) RGB. Two storage conventions are supported and
    auto-detected:
      - uint8 PNG/JPG/.npy in [0, 255]      (8-bit sRGB pipeline)
      - float32 .npy in [0, 1]              (linear-RGB pipeline)

    `target_encoding` optionally reparametrises the TARGET only:
      - "none" (default) -- the target is the stored linear value, unchanged.
      - "pu21"           -- the target becomes pu21(L), which is bounded in
                            [0, 1]. Decode predictions with utils.pu21_decode
                            at inference.
    The guidance is never encoded.
    """

    def __init__(self, dataset_path, file_list, transforms=None,
                 target_encoding="none") -> None:
        super().__init__()

        self.dataset_path = dataset_path
        self.file_list = os.path.join(dataset_path, file_list)
        if target_encoding not in ENCODINGS:
            raise ValueError(f"bad target_encoding: {target_encoding}")
        self.target_encoding = target_encoding

        self.data = self.load()
        self.transforms = transforms

    def load(self):
        data = []

        with open(self.file_list, "r") as f_read:
            item_list = [line.strip() for line in f_read.readlines()]

        for item in item_list:
            if not item:
                continue
            parts = item.split(",")
            assert len(parts) == 2, f"invalid item: {item}"
            target_rel_path, guidance_rel_path = parts

            target_path = os.path.join(self.dataset_path, target_rel_path)
            guidance_path = os.path.join(self.dataset_path, guidance_rel_path)

            if os.path.exists(target_path) and os.path.exists(guidance_path):
                data.append((target_path, guidance_path))
            else:
                print(f"Warning: {target_path} or {guidance_path} does not exist")

        return data

    def np2tensor(self, array):
        return torch.Tensor(array).permute(2, 0, 1)

    def _load_rgb(self, path):
        if os.path.splitext(path)[1] == ".npy":
            arr = np.load(path)
        else:
            arr = imageio.imread(path)
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float32)            # already in [0, 1]
        return arr.astype(np.float32) / 255.0        # uint8 sRGB -> [0, 1]

    def __getitem__(self, idx: int):
        target_path, guidance_path = self.data[idx]

        target_data = self._load_rgb(target_path)
        guidance_data = self._load_rgb(guidance_path)

        if target_data.shape[:2] != guidance_data.shape[:2]:
            raise ValueError(
                f"target.shape: {target_data.shape}, guidance.shape: {guidance_data.shape}, "
                f"file_name: {target_path}, {guidance_path}"
            )

        if self.transforms is not None:
            target_data, guidance_data = self.transforms(target_data, guidance_data)

        # after any crop/flip, and never applied to the guidance
        target_data = ENCODINGS[self.target_encoding](target_data)

        target = self.np2tensor(target_data).float() * 2 - 1   # [-1, 1]
        guide = self.np2tensor(guidance_data).float() * 2 - 1  # [-1, 1]

        out_dict = {
            "target_data": target,
            "guidance_data": guide,
            "path": os.path.relpath(guidance_path, self.dataset_path),
        }

        return out_dict

    def __len__(self):
        return len(self.data)
