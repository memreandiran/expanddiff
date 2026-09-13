#!/bin/bash
# Run one checkpoint over one test split, end to end, and print the scores.
#
#   bash scripts/evaluate_checkpoint.sh <checkpoint> <split> <index> <encoding> <out dir>
#
#   <checkpoint>  e.g. checkpoints/expanddiff_p_150k.ckpt
#   <split>       a preprocessed split: an index file beside <prefix>_target/,
#                 <prefix>_guidance/ and ldr_input/
#   <index>       the index file inside it, e.g. SIHDR_test.txt
#   <encoding>    the checkpoint's target encoding: pu21 or none
#                 (checkpoints/MANIFEST.md lists it per file)
#   <out dir>     where to put the 8-bit twin, the predictions and the yamls
#
# What it does, in order:
#   1  build the 8-bit twin of the split, so the model is fed what other
#      methods are fed  (skipped if it already exists)
#   2  sample from the twin
#   3  fit the per-image brightness gain against the ORIGINAL split
#   4  score: PU21-PSNR and the rest
#   5  optionally PU21-VSI, the corrected columns, PU21-PIQE, FID-R and
#      HDR-VDP-3, each skipped with a message if its dependency is absent
#
# Run it from the repository root. Exits nonzero if any stage fails.

set -u
if [ $# -lt 5 ]; then
  sed -n 2,20p "$0"; exit 2
fi
CKPT=$1; SPLIT=$2; INDEX=$3; ENC=$4; OUT=$5
PY=${PY:-python}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREFIX=${INDEX%.txt}
ARM=$(basename "${CKPT%.ckpt}")
FAIL=0

cd "$REPO" || exit 1
export PYTHONPATH=$REPO
mkdir -p "$OUT"

echo "########## 1. the 8-bit input twin ##########"
Q8=$OUT/$(basename "$SPLIT")_q8
if [ -d "$Q8" ] && [ -n "$(ls -A "$Q8" 2>/dev/null)" ]; then
  echo "already built: $Q8"
else
  "$PY" preprocessing/make_q8_split.py --src "$SPLIT" --dst "$Q8" \
      --file_list "$INDEX" || { echo "FAILED: 8-bit twin"; exit 1; }
fi

echo
echo "########## 2. sampling ##########"
mkdir -p "$OUT/logs"
# A distinct suffix per (checkpoint, split) keeps two runs out of one
# experiment directory: the directory is named from the config, not from the
# checkpoint, so without this a second run writes alongside the first and the
# predictions of the two become impossible to tell apart.
SUFFIX=eval_${ARM}_$(basename "$SPLIT")
"$PY" training/sample.py "checkpoint_path=$CKPT" \
    general.is_linear=true "general.target_encoding=$ENC" \
    general.weight_logl1=0.0 general.weight_pul1=0.0 \
    "general.suffix=$SUFFIX" \
    "dataset.val.data_dir=$Q8" "dataset.val.file_list=$INDEX" \
    2>&1 | tee "$OUT/logs/sample_$ARM.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAILED: sampling"; exit 1; }

# sample.py prints the experiment directory it resolved; the predictions land
# under inference_sampling/<val split>_<index>_<respacing>/visualizations, one
# level deeper than you might expect.
EXP=$(grep -m1 "^experiment_folder: " "$OUT/logs/sample_$ARM.log" | cut -d" " -f2-)
[ -n "${EXP:-}" ] || { echo "FAILED: could not read the experiment directory"; exit 1; }
# Name the directory rather than taking the first one found: sample.py names it
# after the val split, and an experiment directory can hold several.
VALNAME=$(basename "$Q8")_${INDEX%.txt}
SAMPDIR=$(find "$EXP/inference_sampling" -mindepth 1 -maxdepth 1 -type d -name "${VALNAME}*" 2>/dev/null | head -1)
[ -n "${SAMPDIR:-}" ] || SAMPDIR=$(find "$EXP/inference_sampling" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
VIS=$(find "$SAMPDIR" -type d -name visualizations 2>/dev/null | head -1)
RAW=$VIS
if ! ls "$RAW"/*_generated_linear.tiff >/dev/null 2>&1; then
  RAW=$(find "$VIS" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
fi
ls "$RAW"/*_generated_linear.tiff >/dev/null 2>&1 \
  || { echo "FAILED: no predictions under $VIS"; exit 1; }
echo "predictions: $RAW"

echo
echo "########## 3. alignment (against the ORIGINAL split) ##########"
"$PY" metrics/align.py --raw_dir "$RAW" --split_dir "$SPLIT" \
    --file_list "$INDEX" --align scale --out_dir "$OUT/${ARM}_scale/pred" \
  || { echo "FAILED: alignment"; exit 1; }

echo
echo "########## 4. reference metrics ##########"
"$PY" metrics/compute_ref_metrics.py --device "${DEVICE:-cpu}" --linear \
    --data_dir "$SPLIT" --file_list "$INDEX" \
    --pred_dir "$OUT/${ARM}_scale/pred" --out "$OUT/yaml/psnr_$ARM.yaml" \
  || { echo "FAILED: reference metrics"; FAIL=$((FAIL+1)); }

echo
echo "########## 5. the optional metric families ##########"
if command -v octave-cli >/dev/null 2>&1 || command -v octave >/dev/null 2>&1; then
  "$PY" metrics/vsi_ref_cells.py --split_dir "$SPLIT" --file_list "$INDEX" \
      --condition "$(basename "$SPLIT")" --arms "${ARM}_scale" \
      --out_dir "$OUT/yaml/vsi" || { echo "FAILED: PU21-VSI"; FAIL=$((FAIL+1)); }
  "$PY" metrics/crf_ref2_cells.py --split_dir "$SPLIT" --file_list "$INDEX" \
      --condition "$(basename "$SPLIT")" --arms "${ARM}_scale" \
      --out_dir "$OUT/yaml/crf" || { echo "FAILED: corrected columns"; FAIL=$((FAIL+1)); }
else
  echo "SKIP PU21-VSI and the corrected columns: Octave not on PATH"
fi

if [ -n "${VDP_ROOT:-}" ]; then
  DIAG_IN=${DIAG_IN:-24} RES_W=${RES_W:-1920} RES_H=${RES_H:-1080} DIST_M=${DIST_M:-1.0} \
  "$PY" metrics/hdrvdp3_bridge.py --data_dir "$SPLIT" --file_list "$INDEX" \
      --pred_dir "$OUT/${ARM}_scale/pred" --out "$OUT/yaml/vdp3_$ARM.yaml" \
    || { echo "FAILED: HDR-VDP-3"; FAIL=$((FAIL+1)); }
else
  echo "SKIP HDR-VDP-3: set VDP_ROOT to an hdrvdp-3.0.7 install"
fi

"$PY" metrics/compute_fid.py --device "${DEVICE:-cpu}" \
    --data_dir "$SPLIT" --file_list "$INDEX" \
    --pred_dir "$OUT/${ARM}_scale/pred" --out "$OUT/yaml/fid_$ARM.yaml" \
  || { echo "FAILED: FID-R"; FAIL=$((FAIL+1)); }

"$PY" metrics/piqe_ref_cells.py --split_dir "$SPLIT" --file_list "$INDEX" \
    --condition "$(basename "$SPLIT")" --arms "${ARM}_scale" \
    --out_dir "$OUT/yaml/piqe" || { echo "SKIP or FAILED: PU21-PIQE (needs pyiqa)"; }

echo
echo "########## results ##########"
grep -H -E "^(pu21_psnr_ref|psnr|ssim|lpips|n_images):" "$OUT/yaml/psnr_$ARM.yaml" 2>/dev/null
find "$OUT/yaml" -name "*.yaml" | sort | sed "s|^|  |"
echo
echo "########## DONE, $FAIL failures ##########"
[ "$FAIL" -eq 0 ] || exit 1
