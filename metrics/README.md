# Metrics

How to score a set of predictions. `scripts/evaluate_checkpoint.sh` runs all of
this in one command; the steps below are what it does.

You need predictions as `<name>_generated_linear.tiff`, float32 linear radiance
where `1.0` means 1000 cd/m², and the split they came from. `training/sample.py`
and `training/inference_custom.py` both write that form.

Pass absolute paths. `--device` defaults to `cuda`; pass `--device cpu` off-GPU.

## 1. Align

Single-image HDR expansion determines radiance only up to a global scale, so
every method is calibrated the same way before scoring:

    python metrics/align.py --raw_dir <predictions> --split_dir <split> \
        --file_list <prefix>.txt --align scale --out_dir <arm>_scale/pred

`--out_dir` must be the `pred` directory itself. The gain is fitted per image
over pixels the input did not clip. Other modes: `gamma_scale` (two parameters),
`guidance_scale` (fitted against the input, no ground truth), `none`. For LEDiff,
add `--blend --blend_preset supp`, the post-process its paper describes.

⚠ Scores written by `training/sample.py` itself come before this step and are several dB
lower. Start from its tiffs, not its yaml.

## 2. Score

    python metrics/compute_ref_metrics.py --device cpu --linear \
        --data_dir <split> --file_list <prefix>.txt \
        --pred_dir <arm>_scale/pred --out psnr.yaml

| script | produces | read this field |
|---|---|---|
| `compute_ref_metrics.py` | PU21-PSNR, PSNR, SSIM, LPIPS | `pu21_psnr_ref` |
| `vsi_ref_cells.py` | PU21-VSI (Octave) | `pu21_vsi` |
| `crf_ref2_cells.py` | +CRF PU21-PSNR and PU21-VSI | `pu21_psnr_ref`, `pu21_vsi` |
| `compute_fid.py` | FID-R, Reinhard-tone-mapped crops | `fid_all_reinhard.fid` |
| `hdrvdp3_bridge.py` | HDR-VDP-3 in JOD | `hdrvdp3_Q_JOD` |
| `piqe_ref_cells.py` | PU21-PIQE | `pu21_piqe_ref` |

The cell scripts take `--split_dir`, `--condition`, `--arms <arm>_scale` and
`--out_dir`, and read the predictions from `<pred_root>/<arm>_scale/pred`, where
`--pred_root` defaults to `--split_dir`. The reported FID-R is the mean over
`--seed 1234`, `7` and `99`.

Higher is better for PU21-PSNR, PU21-VSI and HDR-VDP-3; lower for PIQE and
FID-R. Check `n_images` matches your split size, and `feature_extractor` reads
`torch-fidelity-inception-v3-compat` for FID.

HDR-VDP-3 needs `VDP_ROOT` and the viewing geometry, which has no default:
`DIAG_IN=24 RES_W=1920 RES_H=1080 DIST_M=1.0` is what every reported number used.
For the corrected basis, write corrected tiffs first:

    python metrics/crf_apply_dir_ref.py --data_dir <split> --file_list <prefix>.txt \
        --pred_dir <arm>_scale/pred --out_dir <arm>_scale_crf/pred

## 3. Compare two methods

    python metrics/paired_test.py --curve pu21 \
        --a <arm_a>_scale/pred --b <arm_b>_scale/pred \
        --target_dir <split>/<prefix>_target

Use this rather than comparing means: per-image differences are heavy-tailed, so
sub-dB gaps between means are routinely not significant. It prints the mean, a
95% confidence interval, the win rate and the median.

## Note on the input

Predictions compared against other methods should be sampled from the 8-bit twin
of the split (`preprocessing/make_q8_split.py`) and aligned and scored against
the original. Otherwise a model reading float32 guidance gets precision the
methods fed the `ldr_input` PNGs never had.
