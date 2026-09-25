"""
Linear-RGB preprocessing pipeline for HDR+.

Both target and guidance are stored in *linear* sRGB (sRGB EOTF reversed),
as float values in [0, 1]. The model trained on these will predict linear
sRGB.

    target   = final.jpg, half-resized, linear-encoded
               (float in [0, 1])
    guidance = same image with shadows + highlights clipped + renormalized
               in linear space, NO sRGB gamma re-applied
               (float in [0, 1])

The model learns: linear-clipped RGB  ->  linear-unclipped RGB.

Per burst folder (must contain final.jpg):
    1. Load final.jpg as uint8 sRGB and resize to half resolution.
    2. Convert to linear via the sRGB EOTF.
    3. Burst-level train/test split:
         - TRAIN burst: tile into non-overlapping patch_size x patch_size
           patches; for each patch, sample shadow_stops ~ U[shadow_min, shadow_max]
           and highlight_stops ~ U[hl_min, hl_max]. Apply linear clip + normalize.
         - TEST  burst: no tiling. Use 2^E[shadow_stops], 2^E[highlight_stops]
           (the mean of the range).
    4. Save target (linear) and guidance (linear) as float32 .npy
       (values in [0, 1]). Train: {burst}_pNNN.*. Test: {burst}.*.
    5. Append to HDRPlus_train.txt / HDRPlus_test.txt in the form
           {target_rel_path},{guidance_rel_path}

Configurable stops ranges (via CLI flags):
    --shadow_stops <min> <max>     (e.g. --shadow_stops -8.1 -7.9)
    --highlight_stops <min> <max>  (e.g. --highlight_stops -2.1 -1.9)

Usage:
    python current_pipeline_linear_RGB.py \\
        --dataset_root <hdrplus>/results_20171023 \\
        --output_dir   data/<split> \\
        --patch_size   256 \\
        --shadow_stops -7.1 -6.9 \\
        --highlight_stops -1.1 -0.9 \\
        --test_ratio   0.165 \\
        --seed         42
"""

import argparse
import cv2
import numpy as np
import random
from pathlib import Path


# ── sRGB gamma utilities ─────────────────────────────────────────────────────

def srgb_to_linear(img_uint8):
    """sRGB uint8 -> linear float64 in [0, 1]."""
    x = img_uint8.astype(np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def clip_and_normalize_linear(linear, threshold_low, threshold_high):
    """Clip shadows + highlights in linear space, normalize to [0, 1] linear.

    NO sRGB gamma re-applied. Output is linear float in [0, 1].
    """
    if threshold_high - threshold_low < 0.05:
        threshold_high = threshold_low + 0.05

    upper = threshold_high
    usable = upper - threshold_low

    clipped = np.clip(linear, threshold_low, upper)
    normalized = (clipped - threshold_low) / usable
    return normalized


def compute_clipmasks(target_lin, guidance_lin, eps=0.0):
    """Compute per-pixel clip-mask channels distinguishing
    artificial vs source clipping AND shadow vs highlight clipping.

    Inputs are (H, W, 3) float linear arrays in [0, 1].

    Returns an (8, H, W) uint8 array with channels:
        [0] orig_shadow_any     -- any channel already crushed to 0 in source
        [1] orig_shadow_all     -- all channels already crushed to 0 in source
        [2] orig_highlight_any  -- any channel already blown to 1 in source
        [3] orig_highlight_all  -- all channels already blown to 1 in source
        [4] art_shadow_any      -- any channel artificially crushed to 0
        [5] art_shadow_all      -- all channels artificially crushed to 0
        [6] art_highlight_any   -- any channel artificially blown to 1
        [7] art_highlight_all   -- all channels artificially blown to 1
    """
    t = target_lin
    g = guidance_lin

    # Already clipped in the source (per channel)
    orig_low_pc = t <= eps                 # (H, W, 3)
    orig_high_pc = t >= 1.0 - eps          # (H, W, 3)

    # Clipped by us (per channel) -- exclude pixels already at the limit
    art_low_pc = (g <= eps) & ~orig_low_pc
    art_high_pc = (g >= 1.0 - eps) & ~orig_high_pc

    return np.stack([
        orig_low_pc.any(axis=-1),
        orig_low_pc.all(axis=-1),
        orig_high_pc.any(axis=-1),
        orig_high_pc.all(axis=-1),
        art_low_pc.any(axis=-1),
        art_low_pc.all(axis=-1),
        art_high_pc.any(axis=-1),
        art_high_pc.all(axis=-1),
    ], axis=0).astype(np.uint8)


# ── Load + resize the final.jpg ─────────────────────────────────────────────

def load_target_linear(final_jpg_path, downsample=2):
    """Load final.jpg, resize, convert sRGB->linear. Returns float64 (H,W,3) in [0,1]."""
    rgb = cv2.imread(str(final_jpg_path))
    if rgb is None:
        raise IOError(f"could not read {final_jpg_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if downsample > 1:
        h, w = rgb.shape[:2]
        rgb = cv2.resize(
            rgb, (w // downsample, h // downsample), interpolation=cv2.INTER_AREA
        )
    return srgb_to_linear(rgb)


# ── Patch extraction ─────────────────────────────────────────────────────────

def extract_patches(arr, patch_size):
    """Non-overlapping patches on a centered grid. Returns list of patches."""
    H, W = arr.shape[:2]
    n_h = H // patch_size
    n_w = W // patch_size
    if n_h == 0 or n_w == 0:
        return []
    off_h = (H - n_h * patch_size) // 2
    off_w = (W - n_w * patch_size) // 2
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            h0 = off_h + i * patch_size
            w0 = off_w + j * patch_size
            patches.append(arr[h0:h0 + patch_size, w0:w0 + patch_size])
    return patches


# ── Main ────────────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser(
        description="HDR+ clip+normalize patch pipeline (linear RGB target + linear guidance)"
    )
    parser.add_argument("--dataset_root", type=str,
                        default="datasets/20171106_subset/results_20171023",
                        help="Path to results_20171023 (burst subfolders with final.jpg)")
    parser.add_argument("--output_dir", type=str,
                        default="datasets/processed/hdrplus_linrgb")
    parser.add_argument("--patch_size", type=int, default=256,
                        help="Patch size in resized pixels (must be a multiple of 64 for the U-Net)")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Integer divisor for resizing the source JPG (1=full, 2=1/2, 4=1/4, 8=1/8). Doesn't need to be a power of 2.")
    parser.add_argument("--shadow_stops", type=float, nargs=2, required=True,
                        help="Range for shadow stops (e.g. -8.1 -7.9)")
    parser.add_argument("--highlight_stops", type=float, nargs=2, required=True,
                        help="Range for highlight stops (e.g. -2.1 -1.9)")
    parser.add_argument("--test_ratio", type=float, default=0.165)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    shadow_stops_min, shadow_stops_max = args.shadow_stops
    highlight_stops_min, highlight_stops_max = args.highlight_stops

    for split in ["train", "test"]:
        (output_dir / f"HDRPlus_{split}_target").mkdir(parents=True, exist_ok=True)
        (output_dir / f"HDRPlus_{split}_guidance").mkdir(parents=True, exist_ok=True)

    burst_folders = sorted([
        d for d in dataset_root.iterdir()
        if d.is_dir() and (d / "final.jpg").exists()
    ])
    print(f"Found {len(burst_folders)} valid burst folders")

    if not burst_folders:
        print("ERROR: no valid bursts.")
        return

    indices = list(range(len(burst_folders)))
    random.shuffle(indices)
    n_test = int(len(burst_folders) * args.test_ratio)
    test_indices = set(indices[:n_test])

    train_lines = []
    test_lines = []
    processed_bursts = 0
    total_patches = 0
    failed = []

    for idx, bf in enumerate(burst_folders):
        try:
            target_lin = load_target_linear(bf / "final.jpg", downsample=args.downsample)
            split = "test" if idx in test_indices else "train"

            if split == "test":
                name = bf.name
                target_out = output_dir / "HDRPlus_test_target" / f"{name}.npy"
                guidance_out = output_dir / "HDRPlus_test_guidance" / f"{name}.npy"
                clipmasks_out = output_dir / "HDRPlus_test_guidance" / f"{name}_clipmasks.npy"

                E_shadow = 0.5 * (shadow_stops_min + shadow_stops_max)
                E_highlight = 0.5 * (highlight_stops_min + highlight_stops_max)
                threshold_low = float(2 ** E_shadow)
                threshold_high = float(2 ** E_highlight)

                guidance_lin = clip_and_normalize_linear(
                    target_lin, threshold_low, threshold_high
                )
                clipmasks = compute_clipmasks(target_lin, guidance_lin)

                np.save(target_out, target_lin.astype(np.float32))
                np.save(guidance_out, guidance_lin.astype(np.float32))
                np.save(clipmasks_out, clipmasks)
                test_lines.append(
                    f"HDRPlus_test_target/{name}.npy,HDRPlus_test_guidance/{name}.npy"
                )
                total_patches += 1

                processed_bursts += 1
                if processed_bursts % 10 == 0 or processed_bursts == len(burst_folders):
                    print(f"  [{processed_bursts}/{len(burst_folders)}] {bf.name} "
                          f"(test, full image {target_lin.shape}) total={total_patches}")
                continue

            patches = extract_patches(target_lin, args.patch_size)
            if not patches:
                print(f"  SKIP (too small for patch): {bf.name}")
                continue

            for p_idx, target_p in enumerate(patches):
                shadow_stops = rng.uniform(shadow_stops_min, shadow_stops_max)
                highlight_stops = rng.uniform(highlight_stops_min, highlight_stops_max)
                threshold_low = float(2 ** shadow_stops)
                threshold_high = float(2 ** highlight_stops)
                guidance_p = clip_and_normalize_linear(
                    target_p, threshold_low, threshold_high
                )

                name = f"{bf.name}_p{p_idx:03d}"
                target_out = output_dir / "HDRPlus_train_target" / f"{name}.npy"
                guidance_out = output_dir / "HDRPlus_train_guidance" / f"{name}.npy"

                np.save(target_out, target_p.astype(np.float32))
                np.save(guidance_out, guidance_p.astype(np.float32))

                train_lines.append(
                    f"HDRPlus_train_target/{name}.npy,HDRPlus_train_guidance/{name}.npy"
                )
                total_patches += 1

            processed_bursts += 1
            if processed_bursts % 10 == 0 or processed_bursts == len(burst_folders):
                print(f"  [{processed_bursts}/{len(burst_folders)}] {bf.name} "
                      f"(train) {len(patches)} patches  "
                      f"target={target_lin.shape} total={total_patches}")

        except Exception as e:
            print(f"  FAILED: {bf.name}: {e}")
            failed.append(bf.name)

    with open(output_dir / "HDRPlus_train.txt", "w") as f:
        f.write("\n".join(sorted(train_lines)) + "\n")
    with open(output_dir / "HDRPlus_test.txt", "w") as f:
        f.write("\n".join(sorted(test_lines)) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Done.")
    print(f"  Bursts processed: {processed_bursts}")
    print(f"  Total patches:    {total_patches}")
    print(f"  Train patches:    {len(train_lines)}")
    print(f"  Test patches:     {len(test_lines)}")
    print(f"  Failed bursts:    {len(failed)}")
    if failed:
        print(f"  Failed names:     {failed}")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
