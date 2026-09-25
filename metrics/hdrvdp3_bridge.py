#!/usr/bin/env python3
"""Score one prediction directory with HDR-VDP-3 (3.0.7) by driving Octave.

  VDP_ROOT=<hdrvdp-3.0.7> DIAG_IN=24 RES_W=1920 RES_H=1080 DIST_M=1.0 \
  python metrics/hdrvdp3_bridge.py --data_dir <split> --pred_dir <arm>/pred \
      --file_list SIHDR_test.txt --out vdp3.yaml

Requires Octave (on PATH, or at $OCT_BIN) and an HDR-VDP-3 3.0.7 installation
at $VDP_ROOT. Some Octave builds, conda-forge among them, also need
$OCTAVE_HOME set to the install prefix; the whole environment is passed
through, so exporting it is enough.

DIAG_IN, RES_W, RES_H and DIST_M have no defaults on the Octave side; unset
gives NaN for every image. They are written into the output yaml. The metric
depends strongly on angular resolution, so results at different geometries are
not comparable; the values above give ~63 pixels per degree.

`rgb-native` expects absolute linear cd/m^2, so both sides are multiplied by
$PEAK_NITS (default 1000).
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import yaml

OCT_SCRIPT = r"""
pkg load statistics
pkg load image
root = getenv('VDP_ROOT');
addpath(root);
addpath(fullfile(root,'utils'));
addpath(fullfile(root,'matlabPyrTools'));
% MEX deliberately NOT on the path. matlabPyrTools ships pure-.m upConv/corrDn
% in the parent dir; the compiled .mex shadows them and returns nothing in its
% 7-argument accumulator form under Octave, killing reconSpyrLevs. Verified:
% with MEX excluded, Q_JOD(x,x)=10.0000 exactly and a noisy pair gives 7.4573.

ppd = hdrvdp_pix_per_deg( str2double(getenv('DIAG_IN')), ...
        [str2double(getenv('RES_W')) str2double(getenv('RES_H'))], ...
        str2double(getenv('DIST_M')) );

listfile = getenv('PAIR_LIST');
outfile  = getenv('OUT_CSV');
fid_in  = fopen(listfile, 'r');
fid_out = fopen(outfile, 'w');
fprintf(fid_out, 'name,Q,Q_JOD\n');
while true
  line = fgetl(fid_in);
  if ~ischar(line), break; end
  parts = strsplit(line, ',');
  name = parts{1};
  S = load(parts{2});          % .mat with L_test and L_ref, already in cd/m^2
  try
    res = hdrvdp3('quality', double(S.L_test), double(S.L_ref), ...
                  'rgb-native', ppd, {'use_gpu', false});
    fprintf(fid_out, '%s,%.6f,%.6f\n', name, res.Q, res.Q_JOD);
  catch err
    fprintf(fid_out, '%s,NaN,NaN\n', name);
    fprintf(2, 'ERROR on %s: %s\n', name, err.message);
  end
  fflush(fid_out);
end
fclose(fid_in); fclose(fid_out);
"""


def hwc(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[2] != 3:
        a = np.transpose(a, (1, 2, 0))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--file_list", default="SIHDR_test.txt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import scipy.io
    import tifffile

    peak = float(os.environ.get("PEAK_NITS", 1000))
    maximg = int(os.environ.get("MAXIMG", 0))
    octave = os.environ.get("OCT_BIN", "octave-cli")

    pairs = []
    with open(os.path.join(args.data_dir, args.file_list)) as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(line.split(",")[:2])
    if maximg:
        pairs = pairs[:maximg]

    tmp = tempfile.mkdtemp(prefix="vdp3_")
    listpath = os.path.join(tmp, "pairs.txt")
    outcsv = os.path.join(tmp, "res.csv")
    n_written = n_missing = 0
    with open(listpath, "w") as lf:
        for t_rel, _g in pairs:
            name = os.path.splitext(os.path.basename(t_rel))[0]
            p = os.path.join(args.pred_dir, name + "_generated_linear.tiff")
            t = os.path.join(args.data_dir, t_rel)
            if not (os.path.exists(p) and os.path.exists(t)):
                n_missing += 1
                continue
            pred = hwc(tifffile.imread(p))
            ref = hwc(np.load(t))
            if pred.shape != ref.shape:
                n_missing += 1
                continue
            mp = os.path.join(tmp, name + ".mat")
            scipy.io.savemat(mp, {
                "L_test": np.maximum(pred, 1e-4) * peak,
                "L_ref": np.maximum(ref, 1e-4) * peak,
            }, do_compression=False)
            lf.write(f"{name},{mp}\n")
            n_written += 1

    if not n_written:
        print("no usable pairs", file=sys.stderr)
        return 1

    scriptpath = os.path.join(tmp, "run.m")
    with open(scriptpath, "w") as fh:
        fh.write(OCT_SCRIPT)

    env = dict(os.environ, PAIR_LIST=listpath, OUT_CSV=outcsv)
    r = subprocess.run([octave, "-q", scriptpath], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], file=sys.stderr)
        print(r.stderr[-2000:], file=sys.stderr)
        return 1
    if r.stderr.strip():
        print(r.stderr[-1000:], file=sys.stderr)

    Q, J, bad = [], [], 0
    with open(outcsv) as fh:
        for row in csv.DictReader(fh):
            q, j = float(row["Q"]), float(row["Q_JOD"])
            if np.isfinite(q) and np.isfinite(j):
                Q.append(q); J.append(j)
            else:
                bad += 1

    if not J:
        print("octave produced no finite scores", file=sys.stderr)
        return 1

    res = {
        "hdrvdp3_Q": float(np.mean(Q)),
        "hdrvdp3_Q_JOD": float(np.mean(J)),
        "hdrvdp3_Q_JOD_std": float(np.std(J)),
        "n_images": len(J),
        "n_missing": n_missing,
        "n_failed": bad,
        "peak_nits": peak,
        "display_diagonal_in": float(os.environ.get("DIAG_IN", 24)),
        "display_resolution": [int(os.environ.get("RES_W", 1920)),
                               int(os.environ.get("RES_H", 1080))],
        "viewing_distance_m": float(os.environ.get("DIST_M", 1.0)),
        "hdrvdp_version": "3.0.7",
        "engine": "octave",
        "color_encoding": "rgb-native",
        "task": "quality",
        "pred_dir": args.pred_dir,
    }
    with open(args.out, "w") as fh:
        yaml.safe_dump(res, fh, sort_keys=True)
    print(f"  Q_JOD {res['hdrvdp3_Q_JOD']:.4f}  Q {res['hdrvdp3_Q']:.4f}  "
          f"n={len(J)} missing={n_missing} failed={bad} -> {args.out}")

    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())