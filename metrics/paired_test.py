#!/usr/bin/env python3
"""Paired per-image significance test between two prediction directories.

Over the images both directories have in common, reports the mean difference,
its standard error, a 95% confidence interval, the per-image win rate and the
median. Prints TIE when the interval straddles zero, and warns when the mean and
the median disagree in sign.

  python metrics/paired_test.py --a <dir> --b <dir> \
      --target_dir <split>/SIHDR_test_target \
      [--guidance_dir <split>/SIHDR_test_guidance --direction highlight|shadow] \
      [--label_a ours --label_b theirs] [--curve pu21|aydin]

Both directories must hold `<name>_generated_linear.tiff`, already decoded and
aligned. Without --direction the comparison is whole-image; with it, only pixels
clipped at that end count, and the two bases are not comparable to each other.

Requires only numpy and tifffile.
"""
import argparse
import glob
import os

import numpy as np
import tifffile

_PU_DENOM = float(np.log10(319.0))
EPS = 1e-6


def hwc(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[2] != 3:
        a = np.transpose(a, (1, 2, 0))
    return a


def pu(x):
    """Aydin et al. (2008) PU encoding, matching hdr_metrics.pu_encode."""
    return np.log10(318.0 * np.clip(x, 1e-8, 1.0) + 1.0) / _PU_DENOM


_PU21_P = [0.353487901, 0.3735252458, 8.277049286e-05, 0.9062562627,
           0.09150803491, 0.9099517204, 596.3148142]


def pu21(x, l_peak=1000.0):
    """Mantiuk 2021 PU21 encoding.

    The PSNR peak constant is omitted: a peak term enters both scores of a pair
    identically and cancels in the paired difference A-B.
    """
    p = _PU21_P
    Y = np.clip(x, 1e-8, 1.0) * l_peak
    Ym = Y ** p[3]
    return p[6] * (((p[0] + p[1] * Ym) / (1.0 + p[2] * Ym)) ** p[4] - p[5])


CURVES = {"aydin": pu, "pu21": pu21}


def score_one(pred, gt, mask3=None, curve=pu):
    if mask3 is None:
        d = (curve(pred) - curve(gt)) ** 2
    else:
        d = (curve(pred)[mask3] - curve(gt)[mask3]) ** 2
    return -10.0 * np.log10(max(float(np.mean(d)), 1e-12))


def collect(pred_dir, target_dir, guidance_dir, direction, curve=pu):
    out = {}
    for tf in sorted(glob.glob(os.path.join(target_dir, "*.npy"))):
        name = os.path.basename(tf)[:-4]
        pf = os.path.join(pred_dir, name + "_generated_linear.tiff")
        if not os.path.exists(pf):
            continue
        gt = hwc(np.load(tf))
        pred = hwc(tifffile.imread(pf))
        if pred.shape[:2] != gt.shape[:2]:
            print(f"[warn] shape mismatch {name}, skipped")
            continue
        mask3 = None
        if direction:
            g = hwc(np.load(os.path.join(guidance_dir, name + ".npy")))
            m = (g >= 1 - EPS) if direction == "highlight" else (g <= EPS)
            m = m.any(axis=-1)
            if not m.any():
                continue
            mask3 = np.repeat(m[:, :, None], 3, axis=2)
        out[name] = score_one(pred, gt, mask3, curve)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="pred dir for arm A")
    ap.add_argument("--b", required=True, help="pred dir for arm B")
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--guidance_dir")
    ap.add_argument("--direction", choices=["highlight", "shadow"],
                    help="omit for whole-image (the 2.1c-blend basis)")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--curve", default="aydin", choices=["aydin", "pu21"],
                    help="perceptual encoding before PSNR. 'aydin' is what every "
                         "existing number in the docs uses; 'pu21' is what SI-HDR "
                         "recommends and what their 3.5 dB threshold assumes.")
    args = ap.parse_args()

    if args.direction and not args.guidance_dir:
        ap.error("--direction requires --guidance_dir")

    A = collect(args.a, args.target_dir, args.guidance_dir, args.direction, CURVES[args.curve])
    B = collect(args.b, args.target_dir, args.guidance_dir, args.direction, CURVES[args.curve])
    keys = sorted(set(A) & set(B))
    if len(keys) < 3:
        print(f"only {len(keys)} common images -- nothing to test")
        return 1

    d = np.array([A[k] - B[k] for k in keys])
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    tie = lo * hi <= 0
    basis = f"{args.direction} region" if args.direction else "whole image"

    print(f"\n{args.label_a} vs {args.label_b}   ({basis}, PU-PSNR, n={len(d)})")
    print(f"  {args.label_a:22s} {np.mean([A[k] for k in keys]):7.3f} dB")
    print(f"  {args.label_b:22s} {np.mean([B[k] for k in keys]):7.3f} dB")
    print(f"  mean difference       {mean:+7.3f} dB   SE {se:.3f}")
    print(f"  95% CI                [{lo:+.3f}, {hi:+.3f}]")
    print(f"  {args.label_a} wins on          {int((d > 0).sum())}/{len(d)} "
          f"({100 * (d > 0).mean():.0f}%)")
    print(f"  median difference     {np.median(d):+7.3f} dB")
    if tie:
        print(f"  -> TIE: the CI straddles zero. Do not claim a winner.")
    else:
        w = args.label_a if mean > 0 else args.label_b
        print(f"  -> SIGNIFICANT in favour of {w}.")
    if np.sign(mean) != np.sign(np.median(d)) and np.median(d) != 0:
        print("  !! mean and median disagree in sign -- heavy-tailed, so the "
              "mean is a poor summary here. Quote the win rate and median.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())