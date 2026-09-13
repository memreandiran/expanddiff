#!/usr/bin/env python
"""Write CRF-corrected copies of a prediction directory, so that HDR-VDP-3 can be
run on the corrected basis.

  python metrics/crf_apply_dir_ref.py --data_dir <split> \
      --pred_dir <arm>/pred --file_list SIHDR_test.txt --out_dir <arm>_crf/pred

The correction is applied with no clamping on either side, matching
crf_ref2_cells.py and the reference implementation.
"""
# Make the repository root importable no matter where this is run from:
# Python puts the SCRIPT's directory on sys.path, not the working directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_ref_metrics import hwc, read_pairs  # noqa: E402
from sihdr_crf_correct_ref import correct as _crf  # noqa: E402  REFERENCE basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--file_list", default="SIHDR_test.txt")
    ap.add_argument("--max_images", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = read_pairs(args.data_dir, args.file_list)
    if args.max_images:
        pairs = pairs[: args.max_images]

    n = missing = mismatch = 0
    for t_rel, _g in pairs:
        name = os.path.splitext(os.path.basename(t_rel))[0]
        p = os.path.join(args.pred_dir, name + "_generated_linear.tiff")
        t = os.path.join(args.data_dir, t_rel)
        if not (os.path.exists(p) and os.path.exists(t)):
            missing += 1
            continue
        pred = hwc(tifffile.imread(p))
        gt = hwc(np.load(t))
        if pred.shape != gt.shape:
            mismatch += 1
            continue
        out = _crf(pred, gt)[0]
        tifffile.imwrite(
            os.path.join(args.out_dir, name + "_generated_linear.tiff"),
            np.asarray(out, dtype=np.float32),
        )
        n += 1
        if n % 40 == 0:
            print(f"  {n}/{len(pairs)}", flush=True)

    print(f"wrote {n} corrected tiffs (missing {missing}, shape mismatch {mismatch})")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())