#!/usr/bin/env python3
"""Corrected (+CRF) PU21-PSNR and PU21-VSI, on the benchmark's own convention.

  python metrics/crf_ref2_cells.py --self_check
  python metrics/crf_ref2_cells.py --split_dir <split> --condition <name> \
      --arms a,b,c [--out_dir <dir>] [--no_vsi]

--self_check verifies PSNR and VSI against stored MATLAB R2024b values and
exits nonzero on any mismatch.

VSI is computed by the reference m_vsi.m under Octave, so $PU21_M must point at
a gfxdisp/pu21 checkout and $OCT_BIN at an octave-cli binary.

Writes <out_dir>/crfref2_<cond>_<arm>.yaml.
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np, tifffile, torch, yaml
from scipy.io import savemat

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("EXPANDIFF_ROOT", os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, ROOT)          # so `rawdiffusion` imports from the repo root
from rawdiffusion.utils import PU21_PEAK, pu21_encode_metric  # noqa: E402
from sihdr_crf_correct_ref import correct as crf_correct      # noqa: E402

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
warning('off','all'); pkg load image; addpath(getenv('PU21_M'));
d = getenv('PAIRS'); f = dir(fullfile(d,'*.mat'));
fid = fopen(getenv('OUT_CSV'),'w'); fprintf(fid,'name,vsi\n');
for i=1:numel(f)
  s = load(fullfile(d,f(i).name));
  fprintf(fid,'%s,%.10f\n', strrep(f(i).name,'.mat',''), m_vsi(s.P, s.T));
end
fclose(fid);
"""

MATLAB_REF = {"001": (42.59198934, 0.99951212),
              "008": (33.57275559, 0.99371936),
              "175": (27.75958496, 0.98524220)}


def hwc(a):
    a = np.asarray(a, np.float32)
    return a.transpose(1, 2, 0) if a.ndim == 3 and a.shape[0] == 3 else a


def enc(nits_over_peak):
    """Raw PU21 units. No [0,1] clamp: the encoder's own [0.005, 10000] applies."""
    return pu21_encode_metric(torch.from_numpy(np.ascontiguousarray(nits_over_peak)), 1000.0)


def read_pairs(split_dir, file_list):
    return [tuple(l.strip().split(",")) for l in open(os.path.join(split_dir, file_list)) if l.strip()]


def corrected_pairs(split_dir, arm, file_list, pred_root=None):
    """Yield (name, corrected_pred_over_peak, target_over_peak)."""
    pdir = os.path.join(pred_root or split_dir, arm, "pred")
    for t_rel, _g in read_pairs(split_dir, file_list):
        name = os.path.splitext(os.path.basename(t_rel))[0]
        pp = os.path.join(pdir, name + "_generated_linear.tiff")
        tp = os.path.join(split_dir, t_rel)
        if not (os.path.exists(pp) and os.path.exists(tp)):
            yield name, None, None
            continue
        pred, gt = hwc(tifffile.imread(pp)), hwc(np.load(tp))
        if pred.shape[:2] != gt.shape[:2] or np.isnan(pred).any():
            yield name, None, None
            continue
        It, _ = crf_correct(pred, gt)     # both already relative to the 1000-nit peak
        yield name, It, gt


def score(split_dir, arm, file_list, want_vsi=True, pred_root=None):
    tmp = tempfile.mkdtemp()
    psnrs, n, miss, over = [], 0, 0, []
    for name, It, gt in corrected_pairs(split_dir, arm, file_list, pred_root):
        if It is None:
            miss += 1
            continue
        P, T = enc(It), enc(gt)
        mse = float(((P - T) ** 2).mean())
        psnrs.append(10.0 * np.log10(PU21_PEAK ** 2 / max(mse, 1e-12)))
        over.append(float((It > 1.0).mean()) * 100.0)
        if want_vsi:
            savemat(os.path.join(tmp, f"{name}.mat"),
                    {"P": P.numpy().astype(np.float64), "T": T.numpy().astype(np.float64)},
                    format="5")
        n += 1
    vsi = None
    if want_vsi and n:
        csv = os.path.join(tmp, "out.csv")
        mf = os.path.join(tmp, "run.m")
        open(mf, "w").write(OCT_SCRIPT)
        env = dict(os.environ, PU21_M=PU21_M, PAIRS=tmp, OUT_CSV=csv,
                   OCTAVE_HOME=_OCT_HOME)
        r = subprocess.run([OCT, mf], env=env, capture_output=True, text=True)
        if os.path.exists(csv):
            vsi = float(np.mean([float(l.split(",")[1])
                                 for l in open(csv).read().splitlines()[1:] if l.strip()]))
        else:
            print(f"  !! octave failed for {arm}\n{r.stderr[-500:]}", file=sys.stderr)
    return (float(np.mean(psnrs)) if psnrs else None, vsi, n, miss,
            float(np.mean(over)) if over else None)


def self_check(split_dir):
    print("SELF-CHECK against MATLAB R2024b (arm scpct150k_q8_scale at C_p)")
    tmp = tempfile.mkdtemp()
    ok = True
    for sid, (want_p, want_v) in MATLAB_REF.items():
        pp = os.path.join(split_dir, "scpct150k_q8_scale", "pred", f"{sid}_generated_linear.tiff")
        tp = os.path.join(split_dir, "SIHDR_test_target", f"{sid}.npy")
        pred, gt = hwc(tifffile.imread(pp)), hwc(np.load(tp))
        It, _ = crf_correct(pred, gt)
        P, T = enc(It), enc(gt)
        got_p = 10.0 * np.log10(PU21_PEAK ** 2 / float(((P - T) ** 2).mean()))
        savemat(os.path.join(tmp, f"{sid}.mat"),
                {"P": P.numpy().astype(np.float64), "T": T.numpy().astype(np.float64)}, format="5")
        dp = abs(got_p - want_p)
        print(f"  {sid} PSNR got {got_p:.8f} want {want_p:.8f}  d {dp:.2e}  "
              f"{'OK' if dp < 1e-5 else 'FAIL'}")
        ok &= dp < 1e-5
    csv = os.path.join(tmp, "out.csv"); mf = os.path.join(tmp, "run.m")
    open(mf, "w").write(OCT_SCRIPT)
    env = dict(os.environ, PU21_M=PU21_M, PAIRS=tmp, OUT_CSV=csv,
               OCTAVE_HOME=_OCT_HOME)
    subprocess.run([OCT, mf], env=env, capture_output=True, text=True)
    if os.path.exists(csv):
        for l in open(csv).read().splitlines()[1:]:
            sid, v = l.split(",")
            want = MATLAB_REF[sid][1]
            d = abs(float(v) - want)
            print(f"  {sid} VSI  got {float(v):.8f} want {want:.8f}  d {d:.2e}  "
                  f"{'OK' if d < 5e-5 else 'FAIL'}")
            ok &= d < 5e-5
    else:
        print("  !! octave produced no VSI"); ok = False
    print("SELF-CHECK PASS" if ok else "SELF-CHECK FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir")
    ap.add_argument("--condition")
    ap.add_argument("--arms")
    ap.add_argument("--file_list", default="SIHDR_test.txt")
    ap.add_argument("--out_dir", default="fid_logs/crf_ref2")
    ap.add_argument("--no_vsi", action="store_true")
    ap.add_argument("--self_check", action="store_true")
    ap.add_argument("--pred_root", default=None,
                    help="directory holding <arm>/pred (default: --split_dir)")
    a = ap.parse_args()

    if a.self_check:
        sd = a.split_dir or os.path.join(ROOT, "lediff_eval/sihdr512_pctclip/cp")
        return self_check(sd)
    for req in ("split_dir", "condition", "arms"):
        if not getattr(a, req):
            raise SystemExit(f"[error] --{req} required")

    os.makedirs(a.out_dir, exist_ok=True)
    failed = []
    for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
        out = os.path.join(a.out_dir, f"crfref2_{a.condition}_{arm}.yaml")
        if os.path.exists(out):
            print(f"  have {out}")
            continue
        p, v, n, miss, over = score(a.split_dir, arm, a.file_list, not a.no_vsi, a.pred_root)
        if not n:
            print(f"  !! no images for {arm} under {os.path.join(a.pred_root or a.split_dir, arm, 'pred')}")
            failed.append(arm)
            continue
        res = {"arm": arm, "condition": a.condition, "crf_corrected": True,
               "crf_basis": "reference-port-gfxdisp-pu21",
               "convention": "reference-no-anchor-clamp",
               "vsi_backend": "reference-m_vsi.m-octave" if v is not None else None,
               "n_images": n, "n_missing": miss, "pu21_l_peak": 1000.0,
               "pct_pixels_above_anchor": over, "pu21_psnr_ref": p, "pu21_vsi": v}
        yaml.dump(res, open(out, "w"), sort_keys=True, default_flow_style=False)
        print(f"  {arm}: PSNR {p:.4f}  VSI {'--' if v is None else f'{v:.6f}'}  "
              f"(n={n}, {over:.2f}% px above anchor) -> {out}")
    if failed:
        print(f"FAILED: no score for {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())