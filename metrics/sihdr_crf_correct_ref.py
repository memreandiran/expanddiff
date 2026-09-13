#!/usr/bin/env python3
"""The benchmark's 20-parameter CRF correction: a port of `crf_correction.m` from
the reference PU21 release (gfxdisp/pu21).

Two per-image regularised least-squares fits of the prediction against the
reference: a cubic in PQ-encoded luma (4 coefficients) and a cubic with cross
terms in CIE u'v' chromaticity (16), shrunk toward the identity mapping. The
benchmark requires this correction for every metric it recommends except
PU21-PIQE.

  from sihdr_crf_correct_ref import correct
  corrected, info = correct(pred, ref)          # both linear RGB, HWC, [0,1]

Called with the reference defaults: deg=3, lambda=0.01, ptf='pq', cspace='luv',
normalize=0, i.e. absolute-luminance input and no pixel subsampling.

  python metrics/sihdr_crf_correct_ref.py --self_test

verifies the basis ordering, the placement of the identity prior, and that
correcting an image against itself returns it exactly.
"""
import argparse
import sys

import numpy as np

RGB2XYZ = np.array([[0.412424, 0.357579, 0.180464],
                    [0.212656, 0.715158, 0.072186],
                    [0.019332, 0.119193, 0.950444]], dtype=np.float64)
XYZ2RGB = np.linalg.inv(RGB2XYZ)

_M = 78.8438
_N = 0.1593
_C1 = 0.8359
_C2 = 18.8516
_C3 = 18.6875

UV_SCALE = 410.0 / 255.0     # rgb2luv / luv2rgb


def ptf(x, ptf_type, L_max, forw):
    """crf_correction.m:102-124. 'lin' (anything but pq/log) is the identity."""
    x = np.asarray(x, dtype=np.float64)
    if forw:
        if ptf_type == "pq":
            Lp = (x / L_max) ** _N
            return ((_C1 + _C2 * Lp) / (1.0 + _C3 * Lp)) ** _M
        if ptf_type == "log":
            return np.log10(x)
        return x
    if ptf_type == "pq":
        xm = x ** (1.0 / _M)
        Lp = (_C1 - xm) / (_C3 * xm - _C2)
        return L_max * Lp ** (1.0 / _N)
    if ptf_type == "log":
        return 10.0 ** x
    return x


def rgb2luv(rgb):
    """crf_correction.m:127-148. Returns (H,W,3) = (Y, u*410/255, v*410/255)."""
    xyz = np.empty_like(rgb)
    for c in range(3):
        xyz[..., c] = np.clip(RGB2XYZ[c, 0] * rgb[..., 0]
                              + RGB2XYZ[c, 1] * rgb[..., 1]
                              + RGB2XYZ[c, 2] * rgb[..., 2], 0.0001, 100000000.0)
    s = xyz.sum(axis=-1)
    x = xyz[..., 0] / s
    y = xyz[..., 1] / s
    luv = np.empty_like(rgb)
    luv[..., 0] = xyz[..., 1]
    den = -2.0 * x + 12.0 * y + 3.0
    luv[..., 1] = 4.0 * x / den * UV_SCALE
    luv[..., 2] = 9.0 * y / den * UV_SCALE
    return luv


def luv2rgb(luv):
    """crf_correction.m:151-179."""
    L = luv[..., 0]
    u = luv[..., 1] * (1.0 / UV_SCALE)
    v = luv[..., 2] * (1.0 / UV_SCALE)
    den = 6.0 * u - 16.0 * v + 12.0
    x = 9.0 * u / den
    y = 4.0 * v / den
    Y = np.clip(L, 0.0001, 100000000.0)
    X = np.clip((x / y) * L, 0.0001, 100000000.0)
    Z = np.clip(((1.0 - x - y) / y) * L, 0.0001, 100000000.0)
    rgb = np.empty_like(luv)
    for c in range(3):
        rgb[..., c] = np.maximum(0.0, XYZ2RGB[c, 0] * X + XYZ2RGB[c, 1] * Y
                                 + XYZ2RGB[c, 2] * Z)
    return rgb


def lin_matrix(I, deg):
    """crf_correction.m:184-207, unrolled.

    zz=2, deg=3 -> [u^3, v^3, u^2, v^2, u*v, u, v, 1]
    zz=1, deg=3 -> [P^3, P^2, P, 1]
    """
    if I.ndim == 2:
        I = I[..., None]
    s = I.shape[0] * I.shape[1]
    zz = I.shape[2]
    M_ = I.reshape(s, zz)

    cols = []
    if zz > 1:                                  # cross-channel terms first
        for i in range(zz - 1):
            for j in range(i + 1, zz):
                cols.append(M_[:, i] * M_[:, j])
    M = np.stack(cols, axis=1) if cols else np.empty((s, 0))
    M = np.concatenate([M, M_, np.ones((s, 1))], axis=1)   # + linear + const
    for d in range(2, deg + 1):                 # PREPENDED, hence descending
        M = np.concatenate([M_ ** d, M], axis=1)
    return M


def corr_opt(Ir, Igt, deg, lam, ptf_type, L_min, L_max):
    """crf_correction.m:78-100."""
    zz = 1 if Ir.ndim == 2 else Ir.shape[2]
    y = ptf(Igt, ptf_type, L_max, 1)
    x = ptf(Ir, ptf_type, L_max, 1)

    Y = y.reshape(-1, zz)
    X = lin_matrix(x, deg)

    W0 = np.zeros((X.shape[1], zz))
    W0[X.shape[1] - 1 - zz:X.shape[1] - 1, :] = np.eye(zz)

    sc = lam * X.shape[0] / X.shape[1]
    W = np.linalg.solve(X.T @ X + sc * np.eye(X.shape[1]), X.T @ Y + sc * W0)

    It = (X @ W).reshape(Ir.shape)
    It = np.maximum(It, ptf(L_min, ptf_type, L_max, 1))
    return ptf(It, ptf_type, L_max, 0), W


def crf_correction(Ir, Igt, deg=3, lam=0.01, ptf_type="pq", cspace="luv",
                   normalize=0):
    """crf_correction.m:17-76. Both images ABSOLUTE linear RGB, HWC."""
    L_min, L_max, SC = 0.005, 1e4, 500.0
    Ir = np.asarray(Ir, dtype=np.float64).copy()
    Igt = np.asarray(Igt, dtype=np.float64).copy()

    scale_gt = None
    if normalize:
        scale_gt = np.median(Igt)
        Igt = SC * Igt / scale_gt
        Ir = SC * Ir / np.median(Ir)
    Igt = np.maximum(Igt, L_min)
    Ir = np.maximum(Ir, L_min)

    if cspace == "rgb":
        It, x = corr_opt(Ir, Igt, deg, lam, ptf_type, L_min, L_max)
    elif cspace == "luv":
        Ir_luv = rgb2luv(Ir)
        Igt_luv = rgb2luv(Igt)
        It_l, x1 = corr_opt(Ir_luv[..., 0], Igt_luv[..., 0],
                            deg, 0.0, ptf_type, L_min, L_max)
        It_uv, x2 = corr_opt(Ir_luv[..., 1:3], Igt_luv[..., 1:3],
                             deg, lam, "lin", L_min, L_max)
        x = (x1, x2)
        It_luv = np.empty_like(Ir_luv)
        It_luv[..., 0] = It_l
        It_luv[..., 1:3] = It_uv
        It = luv2rgb(It_luv)
    else:
        raise ValueError(f"bad cspace: {cspace}")

    if normalize:
        It = scale_gt * It / SC
    return It, x


def correct(pred_rgb, ref_rgb, l_peak=1000.0):
    """Drop-in for sihdr_crf_correct.correct, on the reference basis.

    Our data is display-referred [0,1] with 1.0 = l_peak cd/m^2 (the l_peak
    every PU21 metric here uses). pu21_metric.m demands ABSOLUTE luminance and
    calls crf_correction with normalize=0, so scale in and back out.
    """
    pred = np.asarray(pred_rgb, dtype=np.float64)
    ref = np.asarray(ref_rgb, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {ref.shape}")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out, x = crf_correction(pred * l_peak, ref * l_peak)
    out = np.nan_to_num(out, nan=0.0, posinf=l_peak, neginf=0.0) / l_peak
    return out.astype(np.float32), {
        "luma_coeffs": np.asarray(x[0]).ravel().tolist(),
        "chroma_coeffs": np.asarray(x[1]).ravel().tolist(),
        "n_fit_pixels": int(pred.shape[0] * pred.shape[1]),
        "n_parameters": 20,
        "basis": "reference port of gfxdisp/pu21 crf_correction.m",
    }


def _self_test():
    rng = np.random.default_rng(0)
    ok = True

    Xc = lin_matrix(rng.random((4, 4, 2)), 3)
    Xl = lin_matrix(rng.random((4, 4)), 3)
    print(f"basis     chroma K={Xc.shape[1]} (want 8)   luma K={Xl.shape[1]} (want 4)")
    ok &= Xc.shape[1] == 8 and Xl.shape[1] == 4
    u = np.array([[[0.3, 0.5]]])
    B = lin_matrix(u, 3)[0]
    want = [0.3 ** 3, 0.5 ** 3, 0.3 ** 2, 0.5 ** 2, 0.15, 0.3, 0.5, 1.0]
    print(f"basis order {'OK' if np.allclose(B, want) else 'WRONG'}  "
          f"[u^3,v^3,u^2,v^2,uv,u,v,1]")
    ok &= bool(np.allclose(B, want))
    W0 = np.zeros((8, 2)); W0[8 - 1 - 2:8 - 1, :] = np.eye(2)
    print(f"W0 chroma rows {list(np.flatnonzero(W0.any(axis=1)))} (want [5, 6] "
          f"= u, v -> the paper's c6,1 and c7,2)")
    ok &= list(np.flatnonzero(W0.any(axis=1))) == [5, 6]

    ref = rng.random((64, 64, 3)).astype(np.float32) * 0.5 + 0.001
    out, _ = correct(ref, ref)
    e = np.abs(out - ref).max()
    print(f"identity  max|out-ref| = {e:.3e}")
    ok &= e < 5e-3

    out, _ = correct(ref * 0.37, ref)
    e = np.abs(out - ref).max()
    print(f"gain 0.37 max|out-ref| = {e:.3e}")
    ok &= e < 5e-3

    b = np.abs(ref ** 1.6 - ref).mean()
    out, _ = correct(ref ** 1.6, ref)
    a = np.abs(out - ref).mean()
    print(f"gamma 1.6     mean|err| {b:.4f} -> {a:.4f}")
    ok &= a < b / 2

    cast = ref * np.array([1.25, 1.0, 0.8], np.float32)
    b = np.abs(cast - ref).mean()
    out, _ = correct(cast, ref)
    a = np.abs(out - ref).mean()
    print(f"colour cast   mean|err| {b:.4f} -> {a:.4f}")
    ok &= a < b / 2

    N, K = 1000, 8
    print(f"lambda    0.01*N/K = {0.01 * N / K:g} for N={N}, K={K} "
          f"(the /K sihdr_crf_correct.py omits)")

    print("\nSELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(_self_test())
    ap.error("nothing to do; pass --self_test")