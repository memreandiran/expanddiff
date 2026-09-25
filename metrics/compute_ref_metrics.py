#!/usr/bin/env python3
"""Reference metrics for a directory of saved predictions.

Scores PU21-PSNR, PU21-VSI, PU21-SSIM/MS-SSIM, PSNR, SSIM, LPIPS, MSE, Pearson
and cosine distance for a folder of `<name>_generated_linear.tiff` against a
preprocessed split, and writes them as yaml.

  python metrics/compute_ref_metrics.py --device cpu --linear \
      --data_dir <split> --file_list SIHDR_test.txt \
      --pred_dir <split>/<arm>/pred --out metrics.yaml

--linear enables the PU21 metrics, which assume linear luminance and would
double-warp sRGB input.
--device defaults to cuda; pass cpu explicitly on a machine without a GPU.

`pu21_psnr_ref` is PU21-PSNR on the gfxdisp/pu21 reference convention. For
CRF-corrected PU21-PSNR and PU21-VSI use metrics/crf_ref2_cells.py; for PU21-VSI
from the reference m_vsi.m use metrics/vsi_ref_cells.py.

Relative paths are resolved against $EXPANDIFF_ROOT; absolute paths bypass that.
"""
# Make the repository root importable no matter where this is run from.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import glob
import os
import sys

import numpy as np
import torch
import yaml

from rawdiffusion.evaluation.collection import CollectionMetric
from rawdiffusion.evaluation.metrics import (
    CosineDistanceMetric,
    LPIPSMetric,
    MSEMetric,
    PearsonMetric,
    PSNRMetric,
    PUMSSSIMMetric,
    PU21PIQEMetric,
    MuLawPSNRMetric,
    MuLawSSIMMetric,
    PU21PSNRMetric,
    PU21SSIMMetric,
    PU21VSIMetric,
    PUPSNRMetric,
    PUSSIMMetric,
    SSIMMetric,
)

# Only used to resolve RELATIVE --pred_dir / --data_dir / --out arguments.
# Pass absolute paths and it is never consulted.
ROOT = os.environ.get("EXPANDIFF_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RD = ROOT


def hwc(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[2] != 3:
        a = np.transpose(a, (1, 2, 0))
    return a


def to_t(a, device):
    return torch.from_numpy(
        np.ascontiguousarray(a.transpose(2, 0, 1))[None]).float().to(device)


def read_pairs(data_dir, file_list):
    out = []
    with open(os.path.join(data_dir, file_list)) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(tuple(line.split(",")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--file_list", default="HDRPlus_test.txt")
    ap.add_argument("--out", default=None,
                    help="output yaml (default: metric_ref.yaml beside the run)")
    ap.add_argument("--linear", action="store_true", default=True,
                    help="inputs are linear-RGB, so emit the PU metrics")
    ap.add_argument("--srgb", dest="linear", action="store_false")
    ap.add_argument("--l_peak", type=float, default=1000.0,
                    help="display peak luminance in nits assumed by the PU21 "
                         "encoding. Shifts all PU21-PIQE scores by a constant, "
                         "so it must be identical across everything compared.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data_dir = args.data_dir if os.path.isabs(args.data_dir) \
        else os.path.join(RD, args.data_dir)
    pred_dir = args.pred_dir if os.path.isabs(args.pred_dir) \
        else os.path.join(RD, args.pred_dir)

    metrics_dict = {
        "mse": MSEMetric(),
        "psnr": PSNRMetric(),
        "ssim": SSIMMetric(),
        "pearson": PearsonMetric(),
        "lpips": LPIPSMetric(input_is_linear=args.linear),
        "cosine_distance": CosineDistanceMetric(),
    }
    if args.linear:
        metrics_dict["pu_psnr"] = PUPSNRMetric()
        metrics_dict["pu21_psnr"] = PU21PSNRMetric()
        metrics_dict["pu21_ssim"] = PU21SSIMMetric()
        metrics_dict["pu21_psnr_ref"] = PU21PSNRMetric(convention="reference")
        metrics_dict["pu21_ssim_ref"] = PU21SSIMMetric(convention="reference")
        try:
            _vsi = PU21VSIMetric()
            _d = torch.rand(1, 3, 64, 64)
            _vsi.update(_d, _d)
            assert abs(float(_vsi.compute()) - 1.0) < 1e-4, "VSI(x,x) != 1"
            _vsi.reset()
            metrics_dict["pu21_vsi"] = _vsi
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] PU21-VSI skipped: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        metrics_dict["mu_psnr"] = MuLawPSNRMetric()
        metrics_dict["mu_ssim"] = MuLawSSIMMetric()
        metrics_dict["pu_ssim"] = PUSSIMMetric()
        metrics_dict["pu_ms_ssim"] = PUMSSSIMMetric()
        metrics_dict["pu21_piqe"] = PU21PIQEMetric(l_peak=args.l_peak)
    coll = CollectionMetric(metrics_dict)
    coll.reset()

    import tifffile

    pairs = read_pairs(data_dir, args.file_list)
    n, skipped = 0, 0
    for t_rel, _ in pairs:
        name = os.path.splitext(os.path.basename(t_rel))[0]
        p = os.path.join(pred_dir, name + "_generated_linear.tiff")
        t = os.path.join(data_dir, t_rel)
        if not (os.path.exists(p) and os.path.exists(t)):
            skipped += 1
            continue
        pred = hwc(tifffile.imread(p))
        gt = hwc(np.load(t))
        if pred.shape[:2] != gt.shape[:2]:
            print(f"[warn] shape mismatch {name}: {pred.shape} vs {gt.shape}")
            skipped += 1
            continue
        pt = to_t(pred, args.device).clamp(0, 1)
        gtt = to_t(gt, args.device).clamp(0, 1)
        if torch.isnan(pt).any():
            print(f"[warn] NaN in {name}, skipping")
            skipped += 1
            continue
        coll.update(pt, gtt)
        n += 1
        if n % 10 == 0:
            print(f"  {n}/{len(pairs)}", flush=True)

    if n == 0:
        raise SystemExit(f"[error] no usable pairs in {pred_dir}")

    res = {k: float(v.item() if hasattr(v, "item") else v)
           for k, v in coll.compute().items()}
    res["n_images"] = n
    if skipped:
        res["n_skipped"] = skipped
    if "pu21_piqe" in metrics_dict:
        res["pu21_piqe_backend"] = metrics_dict["pu21_piqe"].backend
        res["pu21_l_peak"] = args.l_peak

    out = args.out
    if out is None:
        d = pred_dir
        for _ in range(3):
            d = os.path.dirname(d)
            if os.path.exists(os.path.join(d, "config.yaml")) or \
               os.path.basename(os.path.dirname(d)) == "inference_sampling":
                break
        out = os.path.join(d, "metric_ref.yaml")
    elif not os.path.isabs(out):
        out = os.path.join(RD, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        yaml.dump(res, fh, sort_keys=True)

    print(f"\n{n} images" + (f", {skipped} skipped" if skipped else ""))
    for k in ("pu_psnr", "pu_ssim", "pu_ms_ssim", "pu21_psnr", "pu21_ssim",
              "pu21_psnr_ref", "pu21_ssim_ref", "pu21_vsi",
              "mu_psnr", "mu_ssim",
              "pu21_piqe", "psnr", "ssim",
              "lpips", "mse", "pearson", "cosine_distance"):
        if k in res and isinstance(res[k], (int, float)):
            print(f"  {k:16s} {res[k]:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()