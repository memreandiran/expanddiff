#!/usr/bin/env python3
"""Turn scene-referred HDR sources into the linear-RGB training pairs the model
is trained on: a target image and the clipped LDR guidance synthesised from it.

Build the paper's two splits through the wrapper scripts in this folder rather
than calling this directly — `run_scenehdr_pct_build.sh` (ExpandDiff-P) and
`run_scenehdr_c95_build.sh` (ExpandDiff-B) pass the settings each split needs.
Call this directly to build a split of your own.

DEGRADATION, chosen with `--degradation`:

  percentile  clip the target at randomly sampled percentiles and renormalise,
              the ranges given by --clip_pct_low / --clip_pct_high. This is what
              both released splits use.
  stops       clip at fixed stop thresholds, --shadow_stops / --highlight_stops.
  lediff      a per-scene clip point derived from the scene's own dynamic range
              (HDR-FLIP exposures), highlights only, no explicit shadow clip.

CAMERA RESPONSE, chosen with `--capture_model`:

  none        no response curve; the guidance stays in linear radiance.
  lediff      push the clipped image through a randomised two-parameter
              response and an 8-bit round trip, then re-linearise with the sRGB
              inverse:

                  I = (1 + b) * min(E,1)^g / (b + min(E,1)^g)
                  b ~ N(0.6, 0.1),  g ~ N(0.9, 0.1)

              Randomising per image trains for an unknown response curve, and
              re-linearising with sRGB rather than the true curve reproduces
              what happens at inference, where an 8-bit photo arrives with its
              response unknown. The ExpandDiff-B split uses this.

TARGET SCALE. Scene radiance is unbounded, so it needs an anchor. Percentile
normalisation is the usual trick and is wrong here: it clips the top fraction
of a percent, which is precisely the highlights we are asking the model to
reconstruct. Instead the scene MEDIAN is placed at a fixed display luminance
(--median_nits, default 20 cd/m^2 -- photographic mid-grey on a 1000-nit
display) and everything above runs freely up to --peak_nits. Highlights are
kept, the scale is consistent across scenes, and it lines up with
general.target_encoding=pu21, whose [0,1] range means exactly [0, peak_nits].
Scenes brighter than the ceiling (direct sun) are clipped there; raise
--peak_nits to 10000 to keep them, at the cost of spending code values on
luminances no display reproduces.

Panoramas are projected to regular images from random camera viewpoints, 5 per
panorama by default; ordinary HDR images are centre-cropped and resized.

Run one source per invocation, into the same output directory. The first writes
the index, every later one adds `--append`:

    # panoramas (Poly Haven, Laval)
    python scenehdr_preprocess.py --dataset_root data/polyhaven --panorama \\
        --views_per_image 5 --output_dir data/scenehdr

    # ordinary HDR images (HDR-Real, Fairchild)
    python scenehdr_preprocess.py --dataset_root data/hdr_real \\
        --output_dir data/scenehdr --append

Output is the split layout train.py and sample.py expect. Point
`dataset.train.data_dir` at `<out>` and `dataset.train.file_list` at the index:

    <out>/SceneHDR_train.txt              index, one "target,guidance" per line
    <out>/SceneHDR_train_target/*.npy     float32 linear, median at median_nits
    <out>/SceneHDR_train_guidance/*.npy   float32 linear, the clipped LDR input
"""
import argparse
import glob
import json
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np  # noqa: E402

# ACES response at t=0.85, i.e. FLIP.h computeExposures(). Hard-coded rather
# than re-derived so this file states the constant LEDiff's pipeline uses.
FLIP_XMAX = 2.118874
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
EXTS = ("*.exr", "*.EXR", "*.hdr", "*.HDR", "*.pfm", "*.PFM")


def load_hdr(path):
    import cv2

    arr = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if arr is None:
        return None
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, 0.0, None)


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92,
                    ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def clip_and_normalize_linear(linear, threshold_low, threshold_high):
    """Clip shadows + highlights in linear space, normalize to [0, 1] linear.

    Copied verbatim from current_pipeline_linear_RGB.py so the `--degradation
    stops` path is bit-identical to the HDR+ pipeline. If that one changes,
    change this one too.
    """
    if threshold_high - threshold_low < 0.05:
        threshold_high = threshold_low + 0.05

    upper = threshold_high
    usable = upper - threshold_low

    clipped = np.clip(linear, threshold_low, upper)
    normalized = (clipped - threshold_low) / usable
    return normalized


def luminance(rgb):
    return rgb @ LUMA


# --------------------------------------------------------------------------- #
# equirectangular -> perspective
# --------------------------------------------------------------------------- #
def perspective_from_equirect(pano, out_h, out_w, yaw, pitch, fov_deg):
    """Gnomonic projection of an equirectangular panorama.

    Bilinear, wrapping in longitude and clamping in latitude. Done in numpy so
    this stays runnable without a GPU; a 4k panorama to 512x512 is ~10 ms.
    """
    H, W = pano.shape[:2]
    f = 0.5 * out_w / np.tan(np.radians(fov_deg) * 0.5)
    xs = np.arange(out_w, dtype=np.float32) - (out_w - 1) * 0.5
    ys = np.arange(out_h, dtype=np.float32) - (out_h - 1) * 0.5
    x, y = np.meshgrid(xs, ys)
    d = np.stack([x, -y, np.full_like(x, f)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    d = d @ rx.T @ ry.T

    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    u = (lon / (2 * np.pi) + 0.5) * W - 0.5
    v = (0.5 - lat / np.pi) * H - 0.5

    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    du, dv = (u - u0)[..., None], (v - v0)[..., None]
    u0m, u1m = u0 % W, (u0 + 1) % W                    # wrap in longitude
    v0m = np.clip(v0, 0, H - 1)
    v1m = np.clip(v0 + 1, 0, H - 1)                    # clamp at the poles

    top = pano[v0m, u0m] * (1 - du) + pano[v0m, u1m] * du
    bot = pano[v1m, u0m] * (1 - du) + pano[v1m, u1m] * du
    return (top * (1 - dv) + bot * dv).astype(np.float32)


def center_crop_resize(arr, out_h, out_w):
    """For non-panorama sources: crop to the output aspect, then box-average."""
    h, w = arr.shape[:2]
    ar = out_w / out_h
    ch, cw = (h, int(round(h * ar))) if w / h > ar else (int(round(w / ar)), w)
    ch, cw = min(ch, h), min(cw, w)
    oy, ox = (h - ch) // 2, (w - cw) // 2
    arr = arr[oy:oy + ch, ox:ox + cw]
    fy, fx = max(1, ch // out_h), max(1, cw // out_w)
    if fy > 1 or fx > 1:                               # area-average, no alias
        arr = arr[:(ch // fy) * fy, :(cw // fx) * fx]
        arr = arr.reshape(ch // fy, fy, cw // fx, fx, 3).mean(axis=(1, 3))
    h2, w2 = arr.shape[:2]
    yi = np.clip((np.arange(out_h) + 0.5) * h2 / out_h - 0.5, 0, h2 - 1)
    xi = np.clip((np.arange(out_w) + 0.5) * w2 / out_w - 0.5, 0, w2 - 1)
    y0, x0 = np.floor(yi).astype(int), np.floor(xi).astype(int)
    y1, x1 = np.minimum(y0 + 1, h2 - 1), np.minimum(x0 + 1, w2 - 1)
    wy, wx = (yi - y0)[:, None, None], (xi - x0)[None, :, None]
    top = arr[y0][:, x0] * (1 - wx) + arr[y0][:, x1] * wx
    bot = arr[y1][:, x0] * (1 - wx) + arr[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


# --------------------------------------------------------------------------- #
# LEDiff's degradation
# --------------------------------------------------------------------------- #
def flip_exposures(lin):
    """(E-, E0, E+, D) in stops, per HDR-FLIP / LEDiff sec. 3.2."""
    y = luminance(lin)
    y_max = float(y.max())
    nz = y[y > 0]
    y_med = float(np.median(nz)) if nz.size else 0.0
    y_med = max(y_med, np.finfo(np.float32).eps)
    if y_max <= 0:
        return None
    e_lo = np.log2(FLIP_XMAX / y_max)
    e_hi = np.log2(FLIP_XMAX / y_med)
    return e_lo, 0.5 * (e_lo + e_hi), e_hi, np.log2(y_max / y_med)


def lediff_capture(lin, exposure, rng, quantize=True):
    """HDR radiance -> 8-bit LDR -> back to linear, LEDiff eq. (1).

    Highlights clip at 1.0; shadows are lost to the response curve and 8-bit
    quantisation rather than to an explicit threshold.
    """
    beta = float(np.clip(rng.normal(0.6, 0.1), 0.05, 2.0))
    gamma = float(np.clip(rng.normal(0.9, 0.1), 0.3, 2.0))
    e = np.minimum(lin * (2.0 ** exposure), 1.0) ** gamma
    ldr = (1.0 + beta) * e / (beta + e)
    ldr = np.clip(ldr, 0.0, 1.0)
    if quantize:
        ldr = np.rint(ldr * 255.0) / 255.0
    # invert sRGB, not the true CRF: at deployment the curve is unknown and
    # this same inverse is what gets applied, so train through the same gap
    return srgb_to_linear(ldr.astype(np.float32)), beta, gamma


def normalize_target(lin, median_nits, peak_nits):
    """Anchor the scene median at a fixed display luminance; keep highlights."""
    y = luminance(lin)
    nz = y[y > 0]
    if not nz.size:
        return None
    med = float(np.median(nz))
    if not np.isfinite(med) or med <= 0:
        return None
    out = lin * (median_nits / peak_nits) / med
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", required=True,
                    help="directory of .exr/.hdr/.pfm, searched recursively")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--panorama", action="store_true",
                    help="treat inputs as equirectangular and project random "
                         "perspective views, as LEDiff does for its panorama "
                         "sources")
    ap.add_argument("--views_per_image", type=int, default=5,
                    help="LEDiff's 36k images over ~5.4k panoramas plus ~10k "
                         "regular images implies roughly this many")
    ap.add_argument("--size", type=int, nargs=2, default=[512, 512],
                    metavar=("H", "W"))
    ap.add_argument("--fov", type=float, nargs=2, default=[55.0, 95.0],
                    metavar=("MIN", "MAX"), help="horizontal FOV range")
    ap.add_argument("--max_pitch", type=float, default=35.0,
                    help="degrees. Keeps the camera off the poles, where an "
                         "equirect panorama is most distorted and where HDRIs "
                         "carry tripod and sky-cap artefacts.")
    ap.add_argument("--median_nits", type=float, default=20.0)
    ap.add_argument("--peak_nits", type=float, default=1000.0,
                    help="matches general.target_encoding=pu21's default "
                         "l_peak; 10000 keeps the sun at the cost of code "
                         "values no display shows")
    ap.add_argument("--min_dr", type=float, default=3.0,
                    help="skip scenes whose median-to-max range is under this "
                         "many stops -- nothing to reconstruct in them")
    ap.add_argument("--clip_repeats", type=int, default=1,
                    help="--degradation stops/percentile only: emit N guidance "
                         "images per target, each with an independently sampled "
                         "clip. The target is written once and shared, so this "
                         "costs guidance storage only. Use it instead of "
                         "re-running with --append: with --exposures e0 the "
                         "guidance filename equals the target base name, so a "
                         "second pass would OVERWRITE the first and leave the "
                         "index with duplicate lines pointing at one file. "
                         "N=3 restores the pair count scenehdr_512 got from "
                         "E-/E0/E+, but as three random clipping amounts rather "
                         "than three fixed exposures.")
    ap.add_argument("--capture_model", default="none",
                    choices=["none", "lediff"],
                    help="applies to --degradation stops/percentile only "
                         "('lediff' already has it built in). 'none' = our "
                         "HDR+ behaviour: the clipped linear guidance is stored "
                         "as-is, no response curve and no quantisation. "
                         "'lediff' additionally pushes it through the "
                         "randomised CRF of their eq. (1) and an 8-bit round "
                         "trip, then linearises back with the sRGB inverse -- "
                         "which is what happens at test time, since SI-HDR's "
                         "inputs are 8-bit PNGs made with a custom CRF and we "
                         "invert sRGB not knowing their curve. Training "
                         "through the same gap is the point. Does NOT model "
                         "sensor noise; SI-HDR's inputs also carry Canon 5D3 "
                         "noise, which remains an unmodelled axis.")
    ap.add_argument("--clip_pct_high", type=float, nargs=2, default=[0.0, 30.0],
                    help="--degradation percentile only: percent of pixels to "
                         "BLOW, sampled U[lo, hi] per image. 0 = untouched. "
                         "SI-HDR's own test inputs sit at 5%% (clip_95) and 3%% "
                         "(clip_97), so a range starting at 0 covers them and "
                         "everything up to our HDR+ regimes (~18%%).")
    ap.add_argument("--clip_pct_low", type=float, nargs=2, default=[0.0, 10.0],
                    help="--degradation percentile only: percent of pixels to "
                         "CRUSH, sampled U[lo, hi] per image. 0 = untouched.")
    ap.add_argument("--degradation", default="lediff",
                    choices=["lediff", "stops", "percentile"],
                    help="how the LDR guidance is made from the HDR target. "
                         "'lediff' = their capture model: per-scene E-/E0/E+ "
                         "from HDR-FLIP, randomised CRF, 8-bit quantisation, "
                         "no explicit shadow clip. This is what scenehdr_512 "
                         "used and what made 2.1i's falsification test valid. "
                         "'stops' = OUR HDR+ recipe: hard clip to "
                         "[2^shadow, 2^highlight] with both stops sampled "
                         "uniformly per image, then linear rescale to [0,1]. "
                         "The HDR+ models trained this way cover no-clip "
                         "through heavy-clip and transfer to clipping rules "
                         "they never saw, which is why they still beat the "
                         "scene generalist on clip_95/clip_97.")
    ap.add_argument("--shadow_stops", type=float, nargs=2, default=[-12.0, -6.0],
                    help="--degradation stops only: U[lo, hi] per image, as in "
                         "current_pipeline_linear_RGB.py. -12 is effectively "
                         "no shadow clip.")
    ap.add_argument("--highlight_stops", type=float, nargs=2, default=[-4.0, 0.0],
                    help="--degradation stops only: U[lo, hi] per image. 0 is "
                         "exactly no highlight clip, which is what puts "
                         "unclipped examples in the training set.")
    ap.add_argument("--exposure_jitter", type=float, default=0.0,
                    help="stops of uniform jitter added to each capture "
                         "exposure, U(-j, +j). LEDiff pins the capture to "
                         "exactly E-/E0/E+, so the model sees only three "
                         "clipping amounts; our HDR+ pipeline samples stops "
                         "over a range instead, which is what lets it handle "
                         "clipping rules it never trained on (clip_95/clip_97 "
                         "threshold a luminance percentile, not an exposure). "
                         "0.0 = off and bit-exact with the existing splits. "
                         "1.0 is a reasonable first try: it spans 2 stops "
                         "around each anchor without letting E- and E+ overlap "
                         "on typical scenes (median bracket span is ~7 stops).")
    ap.add_argument("--exposures", default="e0", choices=["e0", "all"],
                    help="'e0' uses only the middle exposure, which is what a "
                         "normal photo looks like and what arrives at "
                         "deployment. 'all' additionally emits E- and E+ as "
                         "separate pairs against the SAME target: LEDiff needs "
                         "the three because its method IS bracket generation, "
                         "but for us each is just a valid (LDR, HDR) pair at a "
                         "different difficulty -- E- near-unclipped, E+ with "
                         "~50%% of pixels blown. Free 3x augmentation that "
                         "spreads difficulty the way our stop sweeps do.")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--prefix", default="SceneHDR")
    ap.add_argument("--append", action="store_true",
                    help="add to an existing index instead of replacing it, "
                         "so several sources land in one split")
    ap.add_argument("--tag", default=None,
                    help="filename prefix for this source, defaults to the "
                         "basename of --dataset_root")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = args.output_dir
    d_t = os.path.join(out, f"{args.prefix}_{args.split}_target")
    d_g = os.path.join(out, f"{args.prefix}_{args.split}_guidance")
    for d in (d_t, d_g):
        os.makedirs(d, exist_ok=True)
    index = os.path.join(out, f"{args.prefix}_{args.split}.txt")
    tag = args.tag or os.path.basename(os.path.normpath(args.dataset_root))

    files = []
    for e in EXTS:
        files += glob.glob(os.path.join(args.dataset_root, "**", e),
                           recursive=True)
    files = sorted(set(files))
    if args.max_images:
        files = files[:args.max_images]
    if not files:
        raise SystemExit(f"[error] no HDR files under {args.dataset_root}")

    n_views = args.views_per_image if args.panorama else 1
    print(f"{len(files)} HDR files x {n_views} view(s) "
          f"-> up to {len(files) * n_views} images")
    print(f"target scale: median -> {args.median_nits:g} cd/m^2, "
          f"peak {args.peak_nits:g} cd/m^2")

    h, w = args.size
    rng = np.random.RandomState(args.seed)
    lines, n, skipped, drs = [], 0, 0, []

    for fi, path in enumerate(files):
        pano = load_hdr(path)
        if pano is None:
            print(f"[warn] unreadable (OpenEXR backend?): {path}")
            skipped += 1
            continue
        stem = os.path.splitext(os.path.basename(path))[0]

        for v in range(n_views):
            if args.panorama:
                view = perspective_from_equirect(
                    pano, h, w,
                    yaw=rng.uniform(-np.pi, np.pi),
                    pitch=np.radians(rng.uniform(-args.max_pitch,
                                                 args.max_pitch)),
                    fov_deg=rng.uniform(*args.fov))
            else:
                view = center_crop_resize(pano, h, w)

            ex = flip_exposures(view)
            if ex is None:
                skipped += 1
                continue
            e_lo, e0, e_hi, dr = ex
            if dr < args.min_dr:
                skipped += 1
                continue

            tgt = normalize_target(view, args.median_nits, args.peak_nits)
            if tgt is None:
                skipped += 1
                continue

            base = f"{tag}_{stem}_{v:02d}" if n_views > 1 else f"{tag}_{stem}"
            # one target, one guidance per requested exposure. The target is
            # written once and reused, so 'all' costs guidance storage only.
            np.save(os.path.join(d_t, base + ".npy"), tgt)
            picks = [("e0", e0)] if args.exposures == "e0" \
                else [("em", e_lo), ("e0", e0), ("ep", e_hi)]
            # N independently-sampled clips per target. Only meaningful for the
            # randomised degradations -- the lediff capture is already pinned to
            # its three exposures, so repeating it would just resample the CRF.
            if args.clip_repeats > 1 and args.degradation in ("percentile",
                                                              "stops"):
                picks = [(f"{sfx}c{i}", exp)
                         for sfx, exp in picks
                         for i in range(args.clip_repeats)]
            for sfx, exp in picks:
                # Jitter the exposure, HDR+-style. LEDiff's protocol pins the
                # capture to exactly E-/E0/E+ per scene, so the model only ever
                # sees three clipping amounts and the CRF (beta, gamma) is the
                # sole source of variation. Our HDR+ pipeline instead samples
                # stops uniformly over a RANGE per patch, which is why that
                # model covers no-clip through heavy-clip and transfers to
                # clipping rules it never saw -- e.g. clip_95/clip_97, which
                # threshold a luminance PERCENTILE rather than an exposure, and
                # where the scene generalist currently loses to the HDR+ one.
                #
                # jitter=0.0 is the default and reproduces the existing splits
                # bit-exactly, so this cannot silently change anything already
                # built.
                if args.exposure_jitter > 0.0:
                    exp = exp + float(rng.uniform(-args.exposure_jitter,
                                                  args.exposure_jitter))
                # guidance comes from the SAME radiance the target came from,
                # so the pair stays consistent
                if args.degradation == "percentile":
                    # Clip a random PERCENTAGE of pixels rather than at a fixed
                    # radiance. Scale-invariant by construction, so it does not
                    # care that the scene-referred median sits at 0.02 while
                    # the HDR+ median sits at 0.1-0.3 -- the same setting means
                    # the same thing on both. And it is the test protocol:
                    # SI-HDR's clip_95/clip_97 blow 5% / 3% of pixels
                    # (sihdr_preprocess.py:140), while our HDR+ regimes blow
                    # ~18%, so a range from 0 covers the whole span.
                    # Thresholds come from the MAX-CHANNEL percentile, not the
                    # luminance percentile. clip_and_normalize_linear clips per
                    # channel and a pixel reads as blown if ANY channel hits the
                    # ceiling, so a luminance-derived threshold overshoots badly
                    # -- measured, asking for 3% blew 12.9% and 5% blew 19.0%.
                    # Off max-channel the requested percentage is exact.
                    mx = tgt.max(axis=-1)
                    mn = tgt.min(axis=-1)
                    p_hi = float(rng.uniform(*args.clip_pct_high))
                    p_lo = float(rng.uniform(*args.clip_pct_low))
                    t_hi = (float(np.percentile(mx, 100.0 - p_hi))
                            if p_hi > 0.0 else float(mx.max()))
                    t_lo = (float(np.percentile(mn, p_lo))
                            if p_lo > 0.0 else 0.0)
                    gui = clip_and_normalize_linear(tgt, t_lo, t_hi)
                    if args.capture_model == "lediff":
                        # exposure 0: the clip already happened above, so this
                        # only applies the CRF + 8-bit round trip.
                        gui, _, _ = lediff_capture(gui, 0.0, rng)
                elif args.degradation == "stops":
                    # OUR HDR+ recipe, applied to the normalised scene-referred
                    # target rather than a linearised JPEG. Stops are sampled
                    # per image so the model spans no-clip to heavy-clip, which
                    # is the property that makes the HDR+ models transfer.
                    s_lo = float(rng.uniform(*args.shadow_stops))
                    s_hi = float(rng.uniform(*args.highlight_stops))
                    gui = clip_and_normalize_linear(tgt, 2.0 ** s_lo,
                                                    2.0 ** s_hi)
                    if args.capture_model == "lediff":
                        gui, _, _ = lediff_capture(gui, 0.0, rng)
                else:
                    gui, _, _ = lediff_capture(view, exp, rng)
                gname = (base
                         if (args.exposures == "e0" and args.clip_repeats == 1)
                         else f"{base}_{sfx}")
                np.save(os.path.join(d_g, gname + ".npy"),
                        np.clip(gui, 0.0, 1.0).astype(np.float32))
                lines.append(
                    f"{args.prefix}_{args.split}_target/{base}.npy,"
                    f"{args.prefix}_{args.split}_guidance/{gname}.npy")
                n += 1
            drs.append(dr)

        if (fi + 1) % 50 == 0:
            print(f"  {fi + 1}/{len(files)} files, {n} images", flush=True)

    mode = "a" if (args.append and os.path.exists(index)) else "w"
    with open(index, mode) as fh:
        fh.write("\n".join(lines) + "\n")
    total = sum(1 for ln in open(index) if ln.strip())

    stats = {
        "source": args.dataset_root, "tag": tag, "images": n,
        "skipped": skipped, "panorama": args.panorama,
        "views_per_image": n_views, "size": [h, w],
        "median_nits": args.median_nits, "peak_nits": args.peak_nits,
        "dynamic_range_stops": {
            "mean": float(np.mean(drs)) if drs else None,
            "p10": float(np.percentile(drs, 10)) if drs else None,
            "p90": float(np.percentile(drs, 90)) if drs else None,
        },
    }
    with open(os.path.join(out, f"stats_{tag}_{args.split}.json"), "w") as fh:
        json.dump(stats, fh, indent=1)

    print(f"\n{n} images written ({skipped} skipped) -> {out}")
    print(f"index now holds {total} pairs: {index}")
    if drs:
        d = np.array(drs)
        print(f"scene dynamic range: mean {d.mean():.1f} stops "
              f"(p10 {np.percentile(d, 10):.1f}, p90 {np.percentile(d, 90):.1f})")
        print(f"  -> LEDiff E0 clip point equals highlight_stops "
              f"{-(d.mean() / 2 + 1.083):+.2f} on average")
    print("\ntrain with (PU21 targets, since these are scene-referred):\n"
          f"  python train.py dataset=hdrplus_linrgb general.is_linear=true \\\n"
          f"    general.target_encoding=pu21 general.weight_logl1=0.0 \\\n"
          f"    dataset.train.data_dir={out} "
          f"dataset.train.file_list={args.prefix}_train.txt \\\n"
          f"    general.suffix=scenehdr_pu21")


if __name__ == "__main__":
    main()
