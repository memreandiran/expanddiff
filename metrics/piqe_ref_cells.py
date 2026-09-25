#!/usr/bin/env python3
"""PU21-PIQE, no-reference, through the pyiqa implementation, on CPU.

  python metrics/piqe_ref_cells.py --split_dir <split> --condition <name> \
      --arms <arm>[,<arm>...] [--out_dir <dir>] \
      [--check <arm>=<value>]

Writes one yaml per arm, recording the backend and the device beside the score.

pyiqa must be importable; the script aborts otherwise. PIQE values from other
implementations are not comparable.

Runs on CPU only. Compare only scores computed the same way.

--check <arm>=<value> rescores <arm> first and aborts unless it reproduces
<value> within 0.01.
"""
import argparse, os, re, sys
import numpy as np, tifffile, torch

_PU21 = (0.353487901, 0.3735252458, 8.277049286e-05, 0.9062562627,
         0.09150803491, 0.9099517204, 596.3148142)
_YMIN = 0.005


def pu21_encode(lin, l_peak=1000.0):
    p1, p2, p3, p4, p5, p6, p7 = _PU21
    y = torch.clamp(lin * l_peak, min=_YMIN)
    yp = torch.pow(y, p4)
    v = torch.clamp(p7 * (torch.pow((p1 + p2 * yp) / (1 + p3 * yp), p5) - p6), min=0.0)
    yq = float(l_peak) ** p4
    vmax = max(p7 * (((p1 + p2 * yq) / (1 + p3 * yq)) ** p5 - p6), 1e-8)
    return torch.clamp(v / vmax, 0.0, 1.0)


def hwc(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[2] != 3:
        a = np.transpose(a, (1, 2, 0))
    return a


def names(data_dir, file_list):
    out = []
    with open(os.path.join(data_dir, file_list)) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(os.path.splitext(os.path.basename(line.split(",")[0]))[0])
    return out


def score(metric, pred_dir, ns, l_peak=1000.0):
    tot, n = 0.0, 0
    for nm in ns:
        p = os.path.join(pred_dir, nm + "_generated_linear.tiff")
        if not os.path.exists(p):
            continue
        t = torch.from_numpy(np.ascontiguousarray(hwc(tifffile.imread(p)).transpose(2, 0, 1))[None]).float()
        s = float(metric(pu21_encode(t, l_peak)).item())
        if np.isfinite(s):
            tot += s; n += 1
    return (tot / n if n else float("nan")), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--arms", required=True, help="comma list, e.g. pct3lin_scale,pct3notanh_scale")
    ap.add_argument("--file_list", default="SIHDR_test.txt")
    ap.add_argument("--out_dir", default="fid_logs/piqe_ref")
    ap.add_argument("--check", default="", help="arm=expected, aborts unless it reproduces")
    ap.add_argument("--pred_root", default=None,
                    help="directory holding <arm>/pred (default: --split_dir)")
    a = ap.parse_args()
    root = a.pred_root or a.split_dir

    try:
        import pyiqa
    except Exception as e:                                   # noqa: BLE001
        sys.exit(f"ABORT: pyiqa not importable ({e}). Set PYTHONPATH to the "
                 f"staged package; do NOT let the local fallback run.")
    torch.set_grad_enabled(False)
    metric = pyiqa.create_metric("piqe", as_loss=False, device="cpu")
    ns = names(a.split_dir, a.file_list)
    print(f"{len(ns)} images in {a.file_list}")

    if a.check:
        arm, exp = a.check.split("=")
        v, n = score(metric, os.path.join(root, arm, "pred"), ns)
        print(f"BASIS CHECK {arm}: {v:.4f} (n={n}), expected {float(exp):.4f}")
        if abs(v - float(exp)) > 0.01:
            sys.exit("ABORT: basis check failed; the new cells would not be comparable.")
        print("basis OK")

    os.makedirs(a.out_dir, exist_ok=True)
    failed = []
    for arm in a.arms.split(","):
        out = os.path.join(a.out_dir, f"piqe_{a.condition}_{arm}.yaml")
        if os.path.exists(out):
            print(f"have {out}"); continue
        v, n = score(metric, os.path.join(root, arm, "pred"), ns)
        if n == 0:
            print(f"  !! no images for {arm} under {os.path.join(root, arm, 'pred')}")
            failed.append(arm)
            continue
        with open(out, "w") as fh:
            fh.write(f"arm: {arm}\nbackend: pyiqa-piqe\ncondition: {a.condition}\n"
                     f"device: cpu\nn_images: {n}\npu21_l_peak: 1000.0\n"
                     f"pu21_piqe_ref: {v}\n")
        print(f"  {arm}: {v:.4f} (n={n}) -> {out}")
    if failed:
        sys.exit(f"FAILED: no score for {', '.join(failed)}")


if __name__ == "__main__":
    main()