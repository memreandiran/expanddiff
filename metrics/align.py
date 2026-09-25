#!/usr/bin/env python3
"""Fit a reconstruction to the reference scale and save it in the layout the
scoring tools expect.

  python metrics/align.py --raw_dir <model output> --split_dir <split> \
      --align scale --out_dir <arm>_scale/pred

  --align none          no calibration
  --align scale         one parameter: s minimising ||s*pred - gt|| over pixels
                        the input did not clip
  --align gamma_scale   two parameters, (pred^g)*s
  --align guidance_scale  fits the same gain against the guidance, using no
                        ground truth

--out_dir must be the `pred` directory itself, not its parent.

Outputs are `<name>_generated_linear.tiff`, float32 linear, which the scoring
scripts read.
"""
import argparse
import glob
import os

import numpy as np
import tifffile

# Only used to resolve RELATIVE --raw_dir / --split_dir / --out_dir arguments.
# Pass absolute paths and it is never consulted.
ROOT = os.environ.get("EXPANDIFF_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RD = ROOT


def hwc(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[2] != 3:
        a = np.transpose(a, (1, 2, 0))
    return a


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(x, 1.0 / 2.4) - 0.055)


BLEND_PRESETS = {
    "code": {"thr": 0.05, "g_init": 2.4, "g_lo": 2.4, "g_hi": 2.6},
    "supp": {"thr": 0.10, "g_init": 2.0, "g_lo": 1.8, "g_hi": 2.4},
}


def soft_overexposed_mask(ldr_srgb, thr=0.05):
    """LEDiff's own mask (test_hdr_itm.py:generate_soft_mask): 1 where the LDR is
    saturated, with a soft ramp over the top `thr` of the range."""
    m = np.max(ldr_srgb, axis=2)
    m = np.minimum(1.0, np.maximum(0.0, (m - 1.0 + thr) / thr))
    return np.repeat(m[:, :, None], 3, axis=2)


def lediff_blend(pred_hdr, ldr_srgb, gt=None, preset="code"):
    """LEDiff's published post-process (examples/text_to_image/test_hdr_itm.py).

    The network output is used ONLY in over-exposed regions. Everywhere else the
    output is the input LDR raised to a fitted gamma and scaled by a fitted
    exposure. The fit targets the network's own HDR on non-clipped pixels
    (optimize_gamma_exp), so no ground truth is needed.

    Constants come from `preset` (see BLEND_PRESETS): "code" reproduces their
    released script, "supp" reproduces their supplementary.
    """
    from scipy.optimize import least_squares

    P = BLEND_PRESETS[preset]
    over = soft_overexposed_mask(ldr_srgb, thr=P["thr"])
    non_over = 1.0 - over
    l = ldr_srgb[non_over == 1]
    h = pred_hdr[non_over == 1]
    if l.size < 100:
        return pred_hdr

    def resid(params):
        g, e = params
        return (np.clip(l, 1e-8, None) ** g) * (2.0 ** e) - h

    try:
        g, e = least_squares(resid, [P["g_init"], 0.0],
                             bounds=([P["g_lo"], -np.inf],
                                     [P["g_hi"], np.inf])).x
    except Exception:                                        # noqa: BLE001
        return pred_hdr
    ldr_adj = (np.clip(ldr_srgb, 1e-8, None) ** g) * (2.0 ** e)
    return (non_over * ldr_adj + over * pred_hdr).astype(np.float32)


def _square_resize(arr, out_h, out_w):
    """Centre-crop to square then resize, matching sihdr_preprocess.square_resize.
    BOX for downscale (area average, no aliasing), BICUBIC for upscale."""
    from PIL import Image

    h, w = arr.shape[:2]
    side = min(h, w)
    oy, ox = (h - side) // 2, (w - side) // 2
    sq = arr[oy:oy + side, ox:ox + side]
    resample = Image.BOX if side > out_h else Image.BICUBIC
    return np.stack([
        np.asarray(Image.fromarray(sq[:, :, c], mode="F")
                   .resize((out_w, out_h), resample))
        for c in range(sq.shape[2])
    ], axis=-1).astype(np.float32)


def fit_align(pred, gt, guidance, mode):
    """-> (aligned, params). Fitted only on pixels the guidance did NOT clip.

    `scale` and `gamma_scale` fit against the TARGET, so they use ground truth.

    `guidance_scale` is the same one-parameter least-squares scale, fitted
    against the GUIDANCE -- the model's own input -- instead of the target, so
    no ground truth is used. Its results land in guidance units, not target
    units, so they are NOT comparable to the `scale` numbers.
    """
    m = ((guidance > 1e-6) & (guidance < 1 - 1e-6)).all(axis=-1)
    if m.sum() < 100:
        m = np.ones(gt.shape[:2], bool)
    p, g = pred[m], gt[m]
    if mode == "none":
        return pred, {}
    if mode == "guidance_scale":
        r = guidance[m]
        s = float((p * r).sum() / max((p * p).sum(), 1e-12))
        return pred * s, {"scale": s, "oracle_free": True}
    if mode == "scale":
        s = float((p * g).sum() / max((p * p).sum(), 1e-12))
        return pred * s, {"scale": s}
    if mode == "gamma_scale":
        best = (None, None, np.inf)
        for gam in np.linspace(0.5, 2.5, 41):
            pg = np.power(np.clip(p, 1e-8, None), gam)
            s = float((pg * g).sum() / max((pg * pg).sum(), 1e-12))
            err = float(np.mean((pg * s - g) ** 2))
            if err < best[2]:
                best = (gam, s, err)
        gam, s, _ = best
        return np.power(np.clip(pred, 1e-8, None), gam) * s, {"gamma": float(gam),
                                                              "scale": s}
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True,
                    help="LEDiff .npy output, or a dir of our *.tiff")
    ap.add_argument("--split_dir", required=True,
                    help="the 512 pseudo-split with GT and guidance")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--align", default="scale",
                    choices=["none", "scale", "gamma_scale", "guidance_scale"],
                    help="guidance_scale fits the same one-parameter gain "
                         "against the guidance instead of the target (no "
                         "ground truth); its scores are NOT comparable to the "
                         "scale/gamma_scale numbers.")
    ap.add_argument("--file_list", default="HDRPlus_test.txt")
    ap.add_argument("--blend_preset", default="code",
                    choices=["code", "supp"],
                    help="which blend constants to use. 'code' = their released "
                         "test_hdr_itm.py (thr 0.05, gamma in [2.4,2.6]) and is "
                         "the default and what every published number here "
                         "used. 'supp' = their supplementary S3 (thr 0.1, gamma "
                         "in [1.8,2.4], init 2.0). The two disagree; see "
                         "BLEND_PRESETS.")
    ap.add_argument("--blend", action="store_true",
                    help="apply LEDiff's published post-process: keep the "
                         "network output only in over-exposed regions and use a "
                         "gamma/exposure-fitted copy of the input LDR "
                         "elsewhere. REQUIRED to score LEDiff as published -- "
                         "without it ~92%% of pixels are raw network output "
                         "that their own results never use.")
    args = ap.parse_args()

    raw = args.raw_dir if os.path.isabs(args.raw_dir) else os.path.join(RD, args.raw_dir)
    spl = args.split_dir if os.path.isabs(args.split_dir) else os.path.join(RD, args.split_dir)
    out = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(RD, args.out_dir)
    os.makedirs(out, exist_ok=True)

    items = []
    with open(os.path.join(spl, args.file_list)) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t_rel, g_rel = line.split(",")[:2]
            items.append((os.path.splitext(os.path.basename(g_rel))[0],
                          t_rel, g_rel))

    stats, n = [], 0
    for name, t_rel, g_rel in items:
        cand = [os.path.join(raw, name + e) for e in
                (".npy", "_generated_linear.tiff", ".tiff", ".exr", ".hdr")]
        src = next((c for c in cand if os.path.exists(c)), None)
        if src is None:
            print(f"[warn] no raw output for {name}")
            continue
        if src.endswith(".npy"):
            pred = hwc(np.load(src))
        elif src.endswith((".exr", ".hdr")):
            import os as _os
            _os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            import cv2
            a = cv2.imread(src, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if a is None:
                print(f"[warn] cannot read {src}")
                continue
            pred = hwc(np.nan_to_num(
                cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32),
                nan=0.0, posinf=0.0, neginf=0.0))
        else:
            pred = hwc(tifffile.imread(src))
        gt = hwc(np.load(os.path.join(spl, t_rel)))
        gd = hwc(np.load(os.path.join(spl, g_rel)))
        if pred.shape[:2] != gt.shape[:2]:
            pred = _square_resize(pred, gt.shape[0], gt.shape[1])
            if pred.shape[:2] != gt.shape[:2]:
                print(f"[warn] shape mismatch {name}: {pred.shape} vs {gt.shape}")
                continue

        if args.blend:
            lp = os.path.join(spl, "ldr_input",
                              os.path.splitext(os.path.basename(g_rel))[0] + ".png")
            if os.path.exists(lp):
                from PIL import Image
                ls = np.asarray(Image.open(lp).convert("RGB"),
                                dtype=np.float32) / 255.0
                if ls.shape[:2] != pred.shape[:2]:
                    ls = _square_resize(ls, pred.shape[0], pred.shape[1])
                pred = lediff_blend(pred, ls, preset=args.blend_preset)
            else:
                print(f"[warn] --blend but no ldr_input for {name}")

        aligned, par = fit_align(pred, gt, gd, args.align)
        aligned = np.clip(aligned, 0.0, 1.0).astype(np.float32)

        tifffile.imwrite(os.path.join(out, name + "_generated_linear.tiff"),
                         np.ascontiguousarray(aligned), photometric="rgb")
        tifffile.imwrite(os.path.join(out, name + "_guide_linear.tiff"),
                         np.ascontiguousarray(gd), photometric="rgb")
        import imageio

        imageio.imwrite(os.path.join(out, name + "_generated_srgb.png"),
                        np.rint(linear_to_srgb(aligned) * 255).astype(np.uint8))
        stats.append(par)
        n += 1

    print(f"\n{n} images -> {out}   align={args.align}")
    if stats and args.align != "none":
        for k in stats[0]:
            v = np.array([s[k] for s in stats])
            print(f"  {k}: median {np.median(v):.4f}  "
                  f"range [{v.min():.4f}, {v.max():.4f}]")
        print("  (a scale near 1.0 means the method was already in the target's "
              "scale and the calibration did little)")


if __name__ == "__main__":
    main()