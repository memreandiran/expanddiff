#!/usr/bin/env python3
"""SI-HDR -> a condition clipped at BOTH ends, from the same references.

Applies the training degradation of Eq. (1) at fixed percentages rather than
sampled ones:

    python preprocessing/sihdr_pctclip_preprocess.py --dataset_root <SI-HDR> \
        --output_dir data/sihdr_cp_512 --clip_pct_low 5 --clip_pct_high 15

The defaults 5 and 15 are the means of the training distribution (U[0,10] and
U[0,30]); 10 and 30 give a harder condition. Thresholds are percentiles of the
per-pixel MAX and MIN channel, not of luminance.

The target, the anchoring and the resize are identical to sihdr_preprocess.py,
so the two conditions share a target set and differ only in the input. Output is
the same split layout, including `ldr_input/*.png`. Build the 8-bit twin with
make_q8_split.py before scoring.
"""
import argparse, glob, os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np
from PIL import Image

import importlib.util as _ilu
_HERE = os.path.dirname(os.path.abspath(__file__))

def _load(mod, fn):
    spec = _ilu.spec_from_file_location(mod, os.path.join(_HERE, fn))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_sh = _load("_sh", "sihdr_preprocess.py")          # load_hdr, square_resize, ...
_sc = _load("_sc", "scenehdr_preprocess.py")       # clip_and_normalize_linear

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_root", required=True)
ap.add_argument("--output_dir", required=True)
ap.add_argument("--clip_pct_low", type=float, default=5.0,
                help="percent of pixels crushed (default: midpoint of ExpandDiff-P's U[0,10])")
ap.add_argument("--clip_pct_high", type=float, default=15.0,
                help="percent of pixels blown (default: midpoint of ExpandDiff-P's U[0,30])")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--median_nits", type=float, default=20.0)
ap.add_argument("--peak_nits", type=float, default=1000.0)
ap.add_argument("--max_images", type=int, default=0)
a = ap.parse_args()

d_t = os.path.join(a.output_dir, "SIHDR_test_target")
d_g = os.path.join(a.output_dir, "SIHDR_test_guidance")
d_l = os.path.join(a.output_dir, "ldr_input")
for d in (d_t, d_g, d_l):
    os.makedirs(d, exist_ok=True)

ref_dir = os.path.join(a.dataset_root, "reference")
refs = sorted(glob.glob(os.path.join(ref_dir, "*.exr")))
if a.max_images:
    refs = refs[:a.max_images]
if not refs:
    raise SystemExit(f"[error] no .exr in {ref_dir}")
print(f"{len(refs)} references  |  clip {a.clip_pct_low}% low / {a.clip_pct_high}% high", flush=True)

lines, n, bl, cr = [], 0, [], []
for f in refs:
    name = os.path.splitext(os.path.basename(f))[0]
    arr = _sh.load_hdr(f)
    if arr is None:
        print(f"[warn] unreadable {name}"); continue
    norm = _sh.normalize_median_anchor(arr, a.median_nits, a.peak_nits)
    if norm is None:
        print(f"[warn] {name}: no positive luminance"); continue
    tgt = _sh.square_resize(norm, a.size)

    mx, mn = tgt.max(axis=-1), tgt.min(axis=-1)
    t_hi = float(np.percentile(mx, 100.0 - a.clip_pct_high)) if a.clip_pct_high > 0 else float(mx.max())
    t_lo = float(np.percentile(mn, a.clip_pct_low)) if a.clip_pct_low > 0 else 0.0
    gui = _sc.clip_and_normalize_linear(tgt, t_lo, t_hi)

    bl.append((gui >= 1.0).mean()); cr.append((gui <= 0.0).mean())
    np.save(os.path.join(d_t, name + ".npy"), tgt.astype(np.float32))
    np.save(os.path.join(d_g, name + ".npy"), np.clip(gui, 0, 1).astype(np.float32))
    # 8-bit sRGB for the methods that read a PNG (LEDiff, DITM, ExpandNet,
    # MaskHDR, Refusion-HDR).
    Image.fromarray(
        np.rint(np.clip(_sh.linear_to_srgb(gui), 0, 1) * 255).astype(np.uint8)
    ).save(os.path.join(d_l, name + ".png"))
    lines.append(f"SIHDR_test_target/{name}.npy,SIHDR_test_guidance/{name}.npy")
    n += 1
    if n % 25 == 0:
        print(f"  {n}/{len(refs)}", flush=True)

with open(os.path.join(a.output_dir, "SIHDR_test.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{n} scenes -> {a.output_dir}")
print(f"measured coverage: {100*np.mean(bl):.2f}% blown, {100*np.mean(cr):.2f}% crushed")
