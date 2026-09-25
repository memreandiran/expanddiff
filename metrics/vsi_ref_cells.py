#!/usr/bin/env python3
"""PU21-VSI computed by the reference implementation, gfxdisp/pu21's m_vsi.m,
driven through Octave.

  python metrics/vsi_ref_cells.py --split_dir <split> --condition <name> \
      --arms <arm>[,<arm>...] [--out_dir <dir>] [--l_peak 1000]

Requires Octave and a gfxdisp/pu21 checkout: $PU21_M must contain m_vsi.m, and
$OCT_BIN must point at an octave-cli binary. One Octave process per arm.

m_vsi.m is fed RAW PU21 units, as pu21_metric.m does.
rawdiffusion/evaluation/metrics/vsi_ref.py is a dependency-free approximation
of m_vsi.m; prefer this script.

For CRF-corrected PU21-VSI use metrics/crf_ref2_cells.py.

Writes <out_dir>/vsiref_<cond>_<arm>.yaml.
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np, tifffile, yaml
from scipy.io import savemat

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("EXPANDIFF_ROOT", os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, ROOT)          # so `rawdiffusion` imports from the repo root
from rawdiffusion.utils import pu21_encode_metric   # noqa: E402
import torch                                        # noqa: E402

PU21_M = os.environ.get("PU21_M", os.path.join(ROOT, "tools", "pu21_matlab"))
# Octave: $OCT_BIN, else whatever is on PATH. Some builds (conda-forge
# among them) also need $OCTAVE_HOME; when it is not already set we derive
# it from the binary, which is correct for a prefix install.
OCT = os.environ.get("OCT_BIN", "octave-cli")
_OCT_HOME = os.environ.get(
    "OCTAVE_HOME",
    os.path.dirname(os.path.dirname(os.path.abspath(OCT)))
    if os.path.sep in OCT else "")

OCT_SCRIPT = r"""
warning('off','all');
pkg load image;
addpath(getenv('PU21_M'));
d = getenv('PAIRS'); f = dir(fullfile(d,'*.mat'));
fid = fopen(getenv('OUT_CSV'),'w'); fprintf(fid,'name,vsi\n');
for i=1:numel(f)
  s = load(fullfile(d,f(i).name));
  v = m_vsi(s.P, s.T);
  fprintf(fid,'%s,%.10f\n', strrep(f(i).name,'.mat',''), v);
end
fclose(fid);
"""


def hwc(a):
    a = np.asarray(a, np.float32)
    return a.transpose(1, 2, 0) if a.ndim == 3 and a.shape[0] == 3 else a


def read_pairs(split_dir, file_list):
    out = []
    with open(os.path.join(split_dir, file_list)) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(tuple(line.split(",")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--file_list", default="SIHDR_test.txt")
    ap.add_argument("--out_dir", default="fid_logs/vsi_ref")
    ap.add_argument("--l_peak", type=float, default=1000.0)
    ap.add_argument("--pred_root", default=None,
                    help="directory holding <arm>/pred (default: --split_dir)")
    a = ap.parse_args()

    if not os.path.isfile(os.path.join(PU21_M, "m_vsi.m")):
        raise SystemExit(f"[error] m_vsi.m not found under PU21_M={PU21_M}")
    correct = None

    pairs = read_pairs(a.split_dir, a.file_list)
    os.makedirs(a.out_dir, exist_ok=True)
    tag = "vsiref"
    failed = []

    for arm in a.arms.split(","):
        arm = arm.strip()
        out = os.path.join(a.out_dir, f"{tag}_{a.condition}_{arm}.yaml")
        if os.path.exists(out):
            print(f"  have {out}")
            continue
        pdir = os.path.join(a.pred_root or a.split_dir, arm, "pred")
        tmp = tempfile.mkdtemp()
        n = miss = 0
        for t_rel, _g in pairs:
            name = os.path.splitext(os.path.basename(t_rel))[0]
            pp = os.path.join(pdir, name + "_generated_linear.tiff")
            tp = os.path.join(a.split_dir, t_rel)
            if not (os.path.exists(pp) and os.path.exists(tp)):
                miss += 1
                continue
            pred, gt = hwc(tifffile.imread(pp)), hwc(np.load(tp))
            if pred.shape[:2] != gt.shape[:2]:
                miss += 1
                continue
            if correct is not None:
                pred = correct(pred, gt)[0]
            P = pu21_encode_metric(torch.from_numpy(np.ascontiguousarray(pred)).clamp(0, 1),
                                   a.l_peak).numpy().astype(np.float64)
            T = pu21_encode_metric(torch.from_numpy(np.ascontiguousarray(gt)).clamp(0, 1),
                                   a.l_peak).numpy().astype(np.float64)
            savemat(os.path.join(tmp, f"{name}.mat"), {"P": P, "T": T}, format="5")
            n += 1
        if n == 0:
            print(f"  !! no images for {arm} under {pdir}")
            failed.append(arm)
            continue
        csv = os.path.join(tmp, "out.csv")
        mfile = os.path.join(tmp, "run.m")
        with open(mfile, "w") as fh:
            fh.write(OCT_SCRIPT)
        env = dict(os.environ, PU21_M=PU21_M, PAIRS=tmp, OUT_CSV=csv,
                   OCTAVE_HOME=_OCT_HOME)
        r = subprocess.run([OCT, mfile], env=env, capture_output=True, text=True)
        if not os.path.exists(csv):
            print(f"  !! octave produced nothing for {arm}\n{r.stderr[-600:]}")
            failed.append(arm)
            continue
        vals = [float(l.split(",")[1]) for l in open(csv).read().splitlines()[1:] if l.strip()]
        res = {"arm": arm, "condition": a.condition, "backend": "reference-m_vsi.m-octave",
               "crf_corrected": False, "n_images": len(vals), "n_missing": miss,
               "pu21_l_peak": a.l_peak, "pu21_vsi": float(np.mean(vals))}
        with open(out, "w") as fh:
            yaml.dump(res, fh, sort_keys=True, default_flow_style=False)
        print(f"  {arm}: pu21_vsi {res['pu21_vsi']:.6f} (n={len(vals)}) -> {out}")
    if failed:
        sys.exit(f"FAILED: no score for {', '.join(failed)}")


if __name__ == "__main__":
    main()