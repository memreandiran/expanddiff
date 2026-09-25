#!/bin/bash
# Build the percentile-clipped training split ExpandDiff-P is trained on, into
# OUT (default data/scenehdr_pct_512).
#
# Each scene is rendered to 512x512 -- panoramas become VIEWS random perspective
# views, ordinary images are centre-cropped -- and anchored so its median
# luminance is 20 cd/m^2 with a 1000 cd/m^2 ceiling. The LDR guidance is then
# produced by clipping that target at randomly sampled percentiles and
# renormalising.
#
# RANGES, in percent of pixels, sampled U[lo, hi] per image:
#   PCT_HIGH  (--clip_pct_high, default "0 30")  blown
#   PCT_LOW   (--clip_pct_low,  default "0 10")  crushed
#
# CLIP_REPEATS is the number of independently sampled clips per target. The
# default, 3, is the setting of the released models.
#
# CAPTURE=none (default) applies no camera response curve. CAPTURE=lediff adds a
# randomised response and an 8-bit round trip; that is what the ExpandDiff-B
# split uses, via run_scenehdr_c95_build.sh.
#
# Point DATASETS_DIR at a directory holding the HDR sources, one folder each:
#   PolyHaven/  Laval_photometric/extracted/  HDR-Real/extracted/  Fairchild_HDR/
# The repository README says where to obtain them. Missing sources are skipped.

set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
DS=${DATASETS_DIR:-$ROOT/datasets}
PY=${PY:-python}

VIEWS=${VIEWS:-5}
EXPOSURES=${EXPOSURES:-e0}
PCT_HIGH=${PCT_HIGH:-0 30}
PCT_LOW=${PCT_LOW:-0 10}
# N independently sampled clips per target.
CLIP_REPEATS=${CLIP_REPEATS:-3}
# Capture model applied AFTER the percentile clip: "none" (default), or
# "lediff" for the randomised CRF + 8-bit round trip.
CAPTURE=${CAPTURE:-none}
OUT=${OUT:-$ROOT/data/scenehdr_pct_512}

cd "$HERE" || exit 1
command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ] || { echo "no interpreter: $PY"; exit 1; }

# Refuse to write into an existing split. Point OUT at a fresh directory, or
# delete the old one on purpose.
if [ -f "$OUT/SceneHDR_train.txt" ]; then
  echo "REFUSING: $OUT/SceneHDR_train.txt already exists."
  echo "  It holds $(wc -l < "$OUT/SceneHDR_train.txt") pairs from a previous build."
  echo "  Set OUT=<fresh dir>, or remove that directory if you mean to replace it."
  exit 1
fi
mkdir -p "$OUT"

echo "building $OUT"
echo "  degradation=percentile  high=U[$PCT_HIGH]%  low=U[$PCT_LOW]%  capture=$CAPTURE"
echo "  clip_repeats=$CLIP_REPEATS  (independent clip per repeat, shared target)"
echo "  views/pano=$VIEWS  exposures=$EXPOSURES"
echo

n=0; fail=0
run () {   # $1 dir, $2 tag, $3 extra flags, $4 append?
  [ -d "$1" ] || { echo "SKIP $2 (missing $1)"; return; }
  echo "--- $2 ---"
  # shellcheck disable=SC2086
  "$PY" scenehdr_preprocess.py --dataset_root "$1" --output_dir "$OUT" \
    --tag "$2" --views_per_image "$VIEWS" --exposures "$EXPOSURES" \
    --size 512 512 \
    --degradation percentile \
    --clip_pct_high $PCT_HIGH --clip_pct_low $PCT_LOW \
    --clip_repeats "$CLIP_REPEATS" --capture_model "$CAPTURE" \
    $3 $4 \
    && n=$((n + 1)) || { echo "$2 preprocess FAILED"; fail=$((fail + 1)); }
}

# First source writes the index, the rest append to it.
run "$DS/PolyHaven"                    polyhaven "--panorama" ""
run "$DS/Laval_photometric/extracted"  lavalphoto "--panorama" "--append"
run "$DS/HDR-Real/extracted"           hdrreal   ""           "--append"
run "$DS/Fairchild_HDR"                fairchild ""           "--append"

echo
echo "########## DONE ($n sources ok, $fail failed) ##########"
idx=$OUT/SceneHDR_train.txt
if [ -f "$idx" ]; then
  echo "pairs: $(wc -l < "$idx")"
  echo
  echo "--- actual clipping achieved (sanity, 60 images) ---"
  "$PY" - "$OUT" <<'PY'
import glob, os, sys
import numpy as np
out = sys.argv[1]
gs = sorted(glob.glob(os.path.join(out, "SceneHDR_train_guidance", "*.npy")))[:60]
hi, lo = [], []
for f in gs:
    g = np.load(f).astype(np.float32)
    hi.append((g >= 1 - 1e-6).any(-1).mean() * 100)
    lo.append((g <= 1e-6).any(-1).mean() * 100)
hi, lo = np.array(hi), np.array(lo)
print(f"  blown   mean {hi.mean():5.2f}%  median {np.median(hi):5.2f}%")
print(f"  crushed mean {lo.mean():5.2f}%  median {np.median(lo):5.2f}%")
print(f"  in the SI-HDR regime (<=5% blown): {(hi <= 5).sum()}/{len(hi)}")
PY
else
  echo "NO INDEX WRITTEN -- every source failed"
  exit 1
fi
