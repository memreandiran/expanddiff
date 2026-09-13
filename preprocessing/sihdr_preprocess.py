#!/usr/bin/env python3
"""SI-HDR (Hanji & Mantiuk, SIGGRAPH 2022) -> a 512x512 evaluation split.

This is the split the paper evaluates on. Download SI-HDR from its authors,
then:

    python preprocessing/sihdr_preprocess.py --dataset_root <SI-HDR> \
        --output_dir data/sihdr_clip95_512 --clip_level clip_95 --size 512

The LDR inputs are the benchmark's OWN simulated captures, `input/clip_95/*.png`
(or `clip_97`), not something we synthesise, so every method is scored on the
images the benchmark itself defines. Each reference is centre-cropped to a
square and resized to `--size`, and anchored the same way training targets are:
the median luminance at `--median_nits`, `--peak_nits` mapping to 1. That
anchoring is what every reported number uses; `--normalization percentile`
builds a different, incomparable target set.

Output is an ordinary split, which every script here reads unchanged:

    <out>/SIHDR_test.txt             index, one "target,guidance" per line
    <out>/SIHDR_test_target/*.npy    float32 linear [0,1], from their .exr
    <out>/SIHDR_test_guidance/*.npy  their capture, linearised
    <out>/ldr_input/*.png            their capture, 8-bit and untouched

Two guidance forms are written because methods differ in what they accept: the
`.npy` is linear for models that want radiance, the `.png` is the original 8-bit
file for those that want an image. Build the 8-bit twin with make_q8_split.py
before scoring, so every method is fed the same precision.

For the doubly-clipped condition, see sihdr_pctclip_preprocess.py.
"""
import argparse
import glob
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def load_hdr(path):
    """HDR/EXR -> float32 RGB, negatives clipped. cv2 needs the OpenEXR backend,
    enabled by the env var set at import."""
    import cv2

    arr = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if arr is None:
        return None
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.clip(arr, 0.0, None)


def normalize_median_anchor(arr, median_nits=20.0, peak_nits=1000.0):
    """Anchor the scene MEDIAN at a fixed display luminance; keep highlights.

    This is the unbiased alternative to `normalize_hdr`. The percentile variant
    truncates each image's own top 0.1% -- precisely the highlight detail an ITM
    method is asked to reconstruct -- and lands the target in display-referred
    [0,1], which happens to be the output space of our HDR+-trained models. Both
    effects flatter us and penalise a method that emits scene-referred radiance.

    Here every image gets the SAME absolute mapping: median -> median_nits,
    ceiling at peak_nits. The clip at 1.0 is a display limit applied equally to
    both methods rather than a content-dependent cut, and it matches the space
    `general.target_encoding=pu21` trains in (peak_nits == PU21's l_peak).
    """
    y = arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    nz = y[y > 0]
    if not nz.size:
        return None
    med = float(np.median(nz))
    if not np.isfinite(med) or med <= 0:
        return None
    return np.clip(arr * (median_nits / peak_nits) / med, 0.0, 1.0).astype(np.float32)


def normalize_hdr(arr, top_percentile=99.9):
    """Scale so the top percentile maps to 1.0. SI-HDR references are absolute
    radiance with no fixed upper bound, so a percentile is the standard way to
    put them on a relative [0,1] scale without letting one specular highlight
    dominate."""
    v = np.percentile(arr, top_percentile)
    if not np.isfinite(v) or v <= 0:
        v = max(float(arr.max()), 1e-8)
    return np.clip(arr / v, 0.0, 1.0)


def square_resize(arr, size):
    h, w = arr.shape[:2]
    s = min(h, w)
    oy, ox = (h - s) // 2, (w - s) // 2
    sq = arr[oy:oy + s, ox:ox + s]
    resample = Image.BOX if s > size else Image.BICUBIC
    return np.stack([
        np.asarray(Image.fromarray(sq[:, :, c], mode="F")
                   .resize((size, size), resample))
        for c in range(sq.shape[2])
    ], axis=-1).astype(np.float32)


def jpeg_roundtrip(lin, quality):
    """sRGB-encode, JPEG, decode, re-linearise. Models the 8-bit camera path;
    LEDiff's inputs are ordinary photographs, not float buffers."""
    import cv2

    srgb = np.rint(linear_to_srgb(lin) * 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(srgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return lin
    dec = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    x = dec.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92,
                    ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", required=True,
                    help="unpacked SI-HDR root containing reference/ and input/")
    ap.add_argument("--clip_level", default="clip_95",
                    choices=["clip_95", "clip_97"],
                    help="which of their two simulated exposures to use. The "
                         "names give the percent of HDR pixels RETAINED, so "
                         "clip_95 clips 5%% and clip_97 clips 3%% -- both much "
                         "milder than our HDR+ regimes (~18%% blown), which is "
                         "worth stating when comparing across the two.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--degradation", default="dataset",
                    choices=["dataset", "lediff_minus", "lediff_zero",
                             "lediff_plus"],
                    help="'dataset' uses SI-HDR's own simulated captures, which "
                         "clip HIGHLIGHTS ONLY (measured: 0.00%% of pixels at "
                         "zero) -- so LEDiff's shadow arm, which assumes a "
                         "C- input with crushed shadows, is being handed an "
                         "image with none. The lediff_* modes synthesise the "
                         "input from the reference EXR using LEDiff's OWN "
                         "protocol (HDR-FLIP exposures + their randomised CRF), "
                         "so each arm can be tested on the input it assumes: "
                         "plus -> C+ (blown highlights, their highlight arm), "
                         "minus -> C- (crushed shadows, their shadow arm), "
                         "zero -> the middle exposure, clipped at BOTH ends, "
                         "which is the realistic single-photo case and the one "
                         "neither of their arms is designed for.")
    ap.add_argument("--normalization", default="median_anchor",
                    choices=["median_anchor", "percentile"],
                    help="How the reference is put on a [0,1] scale. "
                         "'median_anchor' (default, and what every reported "
                         "number uses) applies ONE fixed photometric mapping to "
                         "every image: the median luminance to --median_nits, "
                         "--peak_nits to 1.0. 'percentile' instead scales each "
                         "image so its own 99.9th percentile hits 1.0, which "
                         "deletes the highlights an expansion method is asked "
                         "to reconstruct and makes each image's scale depend on "
                         "its own content. Splits built the two ways are NOT "
                         "comparable to each other.")
    ap.add_argument("--median_nits", type=float, default=20.0)
    ap.add_argument("--peak_nits", type=float, default=1000.0)
    ap.add_argument("--top_percentile", type=float, default=99.9)
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = args.output_dir
    d_t = os.path.join(out, "SIHDR_test_target")
    d_g = os.path.join(out, "SIHDR_test_guidance")
    d_l = os.path.join(out, "ldr_input")
    for d in (d_t, d_g, d_l):
        os.makedirs(d, exist_ok=True)

    ref_dir = os.path.join(args.dataset_root, "reference")
    in_dir = os.path.join(args.dataset_root, "input", args.clip_level)
    for d in (ref_dir, in_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"[error] missing {d}. Unpack reference.zip and "
                             f"input.zip under {args.dataset_root}")

    refs = sorted(glob.glob(os.path.join(ref_dir, "*.exr"))
                  + glob.glob(os.path.join(ref_dir, "*.EXR")))
    if args.max_images:
        refs = refs[:args.max_images]
    if not refs:
        raise SystemExit(f"[error] no .exr in {ref_dir}")
    print(f"{len(refs)} references, inputs from {args.clip_level}")

    rng = np.random.RandomState(args.seed)
    lines, n, missing = [], 0, 0
    for f in refs:
        name = os.path.splitext(os.path.basename(f))[0]
        png = os.path.join(in_dir, name + ".png")
        if not os.path.exists(png):
            missing += 1
            continue
        arr = load_hdr(f)
        if arr is None:
            print(f"[warn] could not read {name}.exr (OpenEXR backend?)")
            continue

        if args.normalization == "median_anchor":
            norm = normalize_median_anchor(arr, args.median_nits, args.peak_nits)
            if norm is None:
                print(f"[warn] {name}: no positive luminance, skipping")
                continue
        else:
            norm = normalize_hdr(arr, args.top_percentile)
        tgt = square_resize(norm, args.size)

        if args.degradation != "dataset":
            # LEDiff's own degradation, applied to the reference radiance.
            # Reusing scenehdr_preprocess so there is exactly one
            # implementation of their protocol in the tree.
            import importlib.util as _ilu
            _sp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "scenehdr_preprocess.py")
            _sc = _ilu.module_from_spec(_ilu.spec_from_file_location("_sc", _sp))
            _ilu.spec_from_file_location("_sc", _sp).loader.exec_module(_sc)
            ex = _sc.flip_exposures(arr)
            if ex is None:
                print(f"[warn] {name}: no usable luminance, skipping")
                continue
            e_lo, e0, e_hi, _dr = ex
            expo = {"lediff_minus": e_lo, "lediff_zero": e0,
                    "lediff_plus": e_hi}[args.degradation]
            lin_g, _b, _g = _sc.lediff_capture(arr, expo, rng)
            gui = square_resize(lin_g, args.size)
            np.save(os.path.join(d_t, name + ".npy"), tgt.astype(np.float32))
            np.save(os.path.join(d_g, name + ".npy"),
                    np.clip(gui, 0, 1).astype(np.float32))
            Image.fromarray(
                np.rint(np.clip(linear_to_srgb(gui), 0, 1) * 255).astype(np.uint8)
            ).save(os.path.join(d_l, name + ".png"))
            lines.append(f"SIHDR_test_target/{name}.npy,"
                         f"SIHDR_test_guidance/{name}.npy")
            n += 1
            if n % 25 == 0:
                print(f"  {n}/{len(refs)}", flush=True)
            continue

        # their capture, kept 8-bit for LEDiff and linearised for ours.
        # NOTE: the dataset applies a CUSTOM CRF, not sRGB, so the sRGB inverse
        # here is an approximation. It affects only our model's input; LEDiff
        # gets the untouched PNG.
        sdr = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32) / 255.0
        x = np.where(sdr <= 0.04045, sdr / 12.92,
                     ((sdr + 0.055) / 1.055) ** 2.4).astype(np.float32)
        gui = square_resize(x, args.size)

        np.save(os.path.join(d_t, name + ".npy"), tgt.astype(np.float32))
        np.save(os.path.join(d_g, name + ".npy"),
                np.clip(gui, 0, 1).astype(np.float32))
        # crop+resize the PNG identically so LEDiff sees the same pixels
        Image.fromarray(
            np.rint(np.clip(linear_to_srgb(gui), 0, 1) * 255).astype(np.uint8)
        ).save(os.path.join(d_l, name + ".png"))

        lines.append(f"SIHDR_test_target/{name}.npy,"
                     f"SIHDR_test_guidance/{name}.npy")
        n += 1
        if n % 25 == 0:
            print(f"  {n}/{len(refs)}", flush=True)

    with open(os.path.join(out, "SIHDR_test.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n{n} images at {args.size}x{args.size} -> {out}")
    if missing:
        print(f"  {missing} references had no matching {args.clip_level} input")
    print("next: build the 8-bit twin every method is scored on, then sample:\n"
          f"  python preprocessing/make_q8_split.py --src {out} --dst {out}_q8\n"
          "  python training/sample.py checkpoint_path=<ckpt> ... "
          f"dataset.val.data_dir={out}_q8\n"
          "see metrics/README.md for alignment and scoring")


if __name__ == "__main__":
    main()
