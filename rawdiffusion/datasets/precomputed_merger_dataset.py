"""Dataset over precomputed ddim24 predictions, for merger training.

Reads one precomputed x_hat per (patch, stop-draw), plus the stops used. `L`
comes from the split as usual and `L0` is regenerated from the stops.

Rotation and flip augmentation is applied identically to all arrays, so they
stay pixel-aligned. There is no random crop: the stored patches are already the
training patch size.

Batch keys:
    x_hat, guidance_data, target_data  -- all CHW in [-1, 1]
"""
import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .bracket_ops import make_bracket


class PrecomputedMergerDataset(Dataset):
    def __init__(self, precomputed_dir, data_dir, augment=True,
                 precomputed_dir2=None):
        """precomputed_dir2 adds a second x_hat per sample, for the
        two-specialist router. Both directories must have been produced with
        the same seed, so that the drawn stops and the keys are identical and
        the two predictions correspond to the same guidance."""
        self.pre = precomputed_dir
        self.pre2 = precomputed_dir2
        self.data_dir = data_dir
        self.augment = augment
        idx = os.path.join(precomputed_dir, "index.csv")
        if not os.path.exists(idx):
            raise FileNotFoundError(
                f"{idx} missing; run precompute_ddim.py first")
        with open(idx) as fh:
            self.rows = list(csv.DictReader(fh))
        if not self.rows:
            raise ValueError(f"{idx} is empty")
        missing = 0
        keep = []
        for r in self.rows:
            ok = os.path.exists(os.path.join(self.pre, "xhat",
                                             r["key"] + ".npy"))
            if ok and self.pre2:
                ok = os.path.exists(os.path.join(self.pre2, "xhat",
                                                 r["key"] + ".npy"))
            if ok:
                keep.append(r)
            else:
                missing += 1
        if missing:
            print(f"[warn] {missing} precomputed x_hat files missing, skipped")
        self.rows = keep

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _np2t(a):
        return torch.from_numpy(
            np.ascontiguousarray(a.transpose(2, 0, 1))).float() * 2 - 1

    def __getitem__(self, i):
        r = self.rows[i]
        x_hat = np.load(os.path.join(self.pre, "xhat", r["key"] + ".npy")
                        ).astype(np.float32)
        x_hat2 = (np.load(os.path.join(self.pre2, "xhat", r["key"] + ".npy")
                          ).astype(np.float32) if self.pre2 else None)
        full = np.load(os.path.join(self.data_dir, r["target_rel"])
                       ).astype(np.float32)
        t_lo, t_hi = float(r["t_lo"]), float(r["t_hi"])
        l0, _, _ = make_bracket(full, t_lo, t_hi)

        arrs = [x_hat, full, l0] + ([x_hat2] if x_hat2 is not None else [])
        if self.augment:
            k = np.random.randint(4)
            ax = np.random.randint(2)
            if k:
                arrs = [np.rot90(a, k) for a in arrs]
            if np.random.rand() < 0.5:
                arrs = [np.flip(a, ax) for a in arrs]
            arrs = [np.ascontiguousarray(a) for a in arrs]
        x_hat, full, l0 = arrs[0], arrs[1], arrs[2]
        if x_hat2 is not None:
            x_hat2 = arrs[3]

        out = {
            "x_hat": self._np2t(x_hat),
            "target_data": self._np2t(full),
            "guidance_data": self._np2t(l0),
            "stops": torch.tensor([t_lo, t_hi], dtype=torch.float32),
            "path": r["key"],
        }
        if x_hat2 is not None:
            out["x_hat2"] = self._np2t(x_hat2)
        return out
