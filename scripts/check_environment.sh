#!/bin/bash
# Report which parts of this repository can run here, and what is missing.
#
#   bash scripts/check_environment.sh
#
# Every metric family has its own dependencies. This prints one line per family.

PY=${PY:-python}
ok=0; miss=0
say () { printf "  %-22s %s\n" "$1" "$2"; }
have_mod () { "$PY" -c "import $1" >/dev/null 2>&1; }

echo "interpreter"
say "python" "$("$PY" --version 2>&1)"

echo
echo "core (training, sampling, alignment, PU21-PSNR)"
for m in torch lightning hydra omegaconf numpy tifffile PIL yaml; do
  if have_mod "$m"; then say "$m" "ok"; ok=$((ok+1)); else say "$m" "MISSING"; miss=$((miss+1)); fi
done
if have_mod torch; then
  say "cuda" "$("$PY" -c "import torch;print('available' if torch.cuda.is_available() else 'NOT available -- sampling needs a GPU')" 2>/dev/null)"
fi

echo
echo "metric families"
if have_mod lpips; then say "lpips" "ok"; else say "lpips" "MISSING (compute_ref_metrics)"; miss=$((miss+1)); fi
if have_mod piq; then say "piq" "ok"; else say "piq" "MISSING (SSIM/MS-SSIM)"; miss=$((miss+1)); fi

if have_mod torch_fidelity; then
  say "torch-fidelity" "ok (FID-R)"
else
  say "torch-fidelity" "MISSING -- compute_fid.py falls back to a DIFFERENT"
  say "" "  Inception and writes an incomparable score. Install it,"
  say "" "  and always check the yaml records"
  say "" "  feature_extractor: torch-fidelity-inception-v3-compat"
  miss=$((miss+1))
fi
if have_mod pyiqa; then say "pyiqa" "ok (PU21-PIQE)"; else say "pyiqa" "MISSING (PU21-PIQE only)"; miss=$((miss+1)); fi

OCT=${OCT_BIN:-octave-cli}
if command -v "$OCT" >/dev/null 2>&1 || [ -x "$OCT" ]; then
  say "octave" "ok ($OCT)"
  [ -n "${OCTAVE_HOME:-}" ] || say "" "  note: some builds also need OCTAVE_HOME set"
else
  say "octave" "MISSING -- PU21-VSI, +CRF PU21-PSNR/VSI and HDR-VDP-3 need it"
  miss=$((miss+1))
fi

PU=${PU21_M:-}
if [ -n "$PU" ] && [ -f "$PU/m_vsi.m" ] && [ -f "$PU/crf_correction.m" ]; then
  say "PU21_M" "ok ($PU)"
else
  say "PU21_M" "NOT SET or incomplete -- needs m_vsi.m and crf_correction.m"
  say "" "  from https://github.com/gfxdisp/pu21 ; PU21-VSI and +CRF"
  miss=$((miss+1))
fi

if [ -n "${VDP_ROOT:-}" ] && [ -d "$VDP_ROOT" ]; then
  say "VDP_ROOT" "ok ($VDP_ROOT)"
else
  say "VDP_ROOT" "NOT SET -- HDR-VDP-3 3.0.7 from https://hdrvdp.sourceforge.net"
  miss=$((miss+1))
fi

echo
echo "checkpoints"
n=$(ls checkpoints/*.ckpt 2>/dev/null | wc -l)
if [ "$n" -gt 0 ]; then say "checkpoints/" "$n present"
else say "checkpoints/" "none -- run: bash checkpoints/download.sh"; fi

echo
if [ "$miss" -eq 0 ]; then
  echo "everything needed is present."
else
  echo "$miss dependency check(s) failed; the families named above will not run."
  echo "Training, sampling, alignment and PU21-PSNR need only the core list."
fi
