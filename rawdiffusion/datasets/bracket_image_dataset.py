"""Dataset for the two-specialist (LEDiff-inspired) pipeline.

Two single-head models are trained on the same symmetric guidance but
different targets:

    Model H (highlight specialist):  L0 -> L-   shadows clipped, highlights intact
    Model S (shadow specialist):     L0 -> L+   highlights clipped, shadows intact

All three variants are synthesised on the fly from the unclipped target patch
`L` on disk; the stored guidance is not used:

    L0 = (clip(L, t_lo, t_hi) - t_lo) / (t_hi - t_lo)     # guidance / input
    L- = (clip(L, t_lo, 1.0)  - t_lo) / (1.0 - t_lo)      # H target
    L+ =  clip(L, 0.0, t_hi)  / t_hi                      # S target

Training re-draws the stops per patch per epoch. Validation/test uses the
deterministic midpoint stops, matching what the preprocessors write.

`target_variant` selects which array lands in `target_data`, which is the only
key train.py reads, so Model H and Model S differ by one config value.
`return_all=True` additionally exposes every variant, for the fusion module.
"""
import os

import numpy as np
import torch

from .bracket_ops import ENCODINGS, MIN_USABLE_RANGE, draw_stops, make_bracket
from .raw_image_dataset import RGBImageDataset


class BracketRGBImageDataset(RGBImageDataset):
    """RGBImageDataset that synthesises the exposure bracket from the target.

    Args:
        shadow_stops: (min, max) exponents for t_lo = 2**X_shadow.
        highlight_stops: (min, max) exponents for t_hi = 2**X_highlight.
        target_variant: which array becomes `target_data`:
            "highlight" -> L-   (train Model H)
            "shadow"    -> L+   (train Model S)
            "full"      -> L    (the single-head target)
        is_train: True draws stops uniformly per patch; False uses the
            deterministic midpoint of each range.
        return_all: also return target_full / target_highlight / target_shadow
            and the stops, for the fusion stage.
    """

    def __init__(self, dataset_path, file_list, transforms=None,
                 shadow_stops=(-12.0, -6.0), highlight_stops=(-4.0, 0.0),
                 target_variant="highlight", is_train=True, return_all=False,
                 absolute_scale=True, target_encoding="none", seed=None):
        super().__init__(dataset_path, file_list, transforms=transforms)
        if target_variant not in ("highlight", "shadow", "full"):
            raise ValueError(f"bad target_variant: {target_variant}")
        self.shadow_stops = tuple(float(v) for v in shadow_stops)
        self.highlight_stops = tuple(float(v) for v in highlight_stops)
        self.target_variant = target_variant
        self.is_train = is_train
        self.return_all = return_all
        self.absolute_scale = absolute_scale
        if target_encoding not in ENCODINGS:
            raise ValueError(f"bad target_encoding: {target_encoding}")
        self.target_encoding = target_encoding
        self.seed = seed

    # ------------------------------------------------------------------ #
    def _draw_stops(self, idx):
        rng = (np.random.default_rng(self.seed + idx)
               if (self.seed is not None and self.is_train) else None)
        return draw_stops(self.shadow_stops, self.highlight_stops,
                          self.is_train, rng)

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int):
        target_path, guidance_path = self.data[idx]
        full = self._load_rgb(target_path)          # unclipped L, float32 [0,1]

        # Crop/augment BEFORE synthesising, so every variant is pixel-aligned
        # by construction. ImageTransforms applies identical ops to both of its
        # arguments, so passing (full, full) keeps the randomness in sync.
        if self.transforms is not None:
            full, _ = self.transforms(full, full)
        full = np.ascontiguousarray(full, dtype=np.float32)

        t_lo, t_hi = self._draw_stops(idx)
        l0, l_minus, l_plus = make_bracket(full, t_lo, t_hi,
                                           self.absolute_scale)

        chosen = {"highlight": l_minus, "shadow": l_plus, "full": full}[
            self.target_variant]
        # Encode the target only, and only after the bracket is built: clipping
        # must happen in linear radiance. Decode with
        # bracket_ops.pu21_decode_np at inference.
        chosen = ENCODINGS[self.target_encoding](chosen)

        out = {
            "target_data": self.np2tensor(chosen).float() * 2 - 1,
            "guidance_data": self.np2tensor(l0).float() * 2 - 1,
            "path": os.path.relpath(guidance_path, self.dataset_path),
        }
        if self.return_all:
            # same encoding as `chosen`
            enc = ENCODINGS[self.target_encoding]
            out["target_full"] = self.np2tensor(enc(full)).float() * 2 - 1
            out["target_highlight"] = self.np2tensor(enc(l_minus)).float() * 2 - 1
            out["target_shadow"] = self.np2tensor(enc(l_plus)).float() * 2 - 1
            out["stops"] = torch.tensor([t_lo, t_hi], dtype=torch.float32)
        return out
