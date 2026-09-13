"""Exposure-bracket synthesis, numpy only.

Kept free of torch so it can be unit-tested and reused off-cluster (haas001
has no torch) and shared by the dataset, the fusion stage, and any evaluation
script that needs to rebuild the same variants.

Given an unclipped linear-RGB patch L in [0, 1] and thresholds t_lo < t_hi:

    L0 = (clip(L, t_lo, t_hi) - t_lo) / (t_hi - t_lo)    guidance: both ends clipped
    L- = (clip(L, t_lo, 1.0)  - t_lo) / (1.0 - t_lo)     highlights intact
    L+ =  clip(L, 0.0, t_hi)  / t_hi                     shadows intact

L0 is what the model sees. L- and L+ are the two specialist targets: each keeps
one end of the range intact so a single decoder only ever has to reconstruct one
kind of destroyed content.
"""
import numpy as np

MIN_USABLE_RANGE = 0.05  # same floor the preprocessors enforce


def draw_stops(shadow_stops, highlight_stops, is_train, rng=None):
    """-> (t_lo, t_hi) linear thresholds.

    Training draws uniformly inside each stop range, matching the per-patch
    randomisation the preprocessors use. Validation/test takes the deterministic
    midpoint, which is exactly what the preprocessed test splits contain, so
    results stay comparable to existing checkpoints.
    """
    s_lo, s_hi = (float(v) for v in shadow_stops)
    h_lo, h_hi = (float(v) for v in highlight_stops)
    if is_train:
        r = rng if rng is not None else np.random.default_rng()
        x_s, x_h = r.uniform(s_lo, s_hi), r.uniform(h_lo, h_hi)
    else:
        x_s, x_h = 0.5 * (s_lo + s_hi), 0.5 * (h_lo + h_hi)
    t_lo, t_hi = 2.0 ** x_s, 2.0 ** x_h
    if t_hi - t_lo < MIN_USABLE_RANGE:
        t_hi = t_lo + MIN_USABLE_RANGE
    return float(t_lo), float(t_hi)


def make_bracket(lin, t_lo, t_hi, absolute_scale=True):
    """-> (L0, L_minus, L_plus), each float32 in [0, 1], same shape as `lin`.

    L0 is always the rescaled guidance: it is the model INPUT and must span
    [0, 1] the way the preprocessed data does.

    `absolute_scale` controls the two TARGETS:

      True (default) -- targets are clip-only, already in L's own units:
          L- = clip(L, t_lo, 1.0)      spans [t_lo, 1]
          L+ = clip(L, 0.0,  t_hi)     spans [0, t_hi]
        Fusion is then a plain convex combination of the two predictions and
        needs NO stops at inference, so the pipeline stays deployable on
        arbitrary photos. Between them the pair covers the whole range: every
        pixel is either above t_lo (where L- == L) or below t_hi (where
        L+ == L). Caveat: L+ is compressed into [0, t_hi], so its L1/L2
        gradients are ~1/t_hi smaller than L-'s.

      False -- each target is additionally rescaled to fill [0, 1]. Trains more
        comfortably but the variants then live on three different affine axes,
        so any fusion must first un-normalise them (see unnormalize_* below),
        which requires knowing t_lo and t_hi at inference.
    """
    l0 = (np.clip(lin, t_lo, t_hi) - t_lo) / (t_hi - t_lo)
    if absolute_scale:
        l_minus = np.clip(lin, t_lo, 1.0)
        l_plus = np.clip(lin, 0.0, t_hi)
    else:
        l_minus = (np.clip(lin, t_lo, 1.0) - t_lo) / (1.0 - t_lo)
        l_plus = np.clip(lin, 0.0, t_hi) / t_hi
    return (l0.astype(np.float32), l_minus.astype(np.float32),
            l_plus.astype(np.float32))


# --------------------------------------------------------------------------- #
# Un-normalisation: map each variant back into the TARGET's scale.
#
# L0, L- and L+ are three DIFFERENT affine rescalings of L, so a convex
# combination of them cannot reconstruct L. Measured on a real test image with
# midpoint stops: |L0 - L| = 0.226 mean abs over unclipped pixels, while
# |unnormalize_L0(L0) - L| = 1.9e-9. Any fusion must therefore operate on
# un-normalised sources, all of which live in L's scale and are each EXACT over
# part of the range:
#
#   unnormalize_L0     exact wherever the guidance was not clipped
#   unnormalize_minus  exact wherever highlights survived (L >= t_lo)
#   unnormalize_plus   exact wherever shadows survived   (L <= t_hi)
#
# unnormalize_L0 is precisely the analytic inverse-rescale baseline.
# --------------------------------------------------------------------------- #
def unnormalize_L0(l0, t_lo, t_hi):
    """Guidance -> target scale. Exact on unclipped pixels, flat at the limits."""
    return l0 * (t_hi - t_lo) + t_lo


def unnormalize_minus(l_minus, t_lo, t_hi=None):
    """Highlight-specialist output -> target scale."""
    return l_minus * (1.0 - t_lo) + t_lo


def unnormalize_plus(l_plus, t_lo=None, t_hi=1.0):
    """Shadow-specialist output -> target scale."""
    return l_plus * t_hi


# --------------------------------------------------------------------------- #
# Target encoding, for training on SCENE-REFERRED radiance.
#
# HDR+ and FiveK targets are display-referred and already live in [0,1], so the
# tanh-bounded U-Net can predict them directly. True HDR (SI-HDR, Fairchild) is
# unbounded scene radiance: normalising it into [0,1] by percentile throws away
# the top of the range, which is precisely the highlight detail such a dataset
# exists for.
#
# PU21 encoding solves this without touching the architecture. It is bounded in
# [0,1] by construction, perceptually uniform, and monotonic, so the model can
# predict PU-encoded values with the same tanh head and the full dynamic range
# survives. Decode at inference. This is the same idea as LEDiff's log-space
# decoder.
#
# It also aligns training with reporting: PU-PSNR is exactly PSNR computed on
# these values, so the loss now optimises what the headline metric measures.
#
# NOTE: clipping must still happen in LINEAR radiance, because clipping is a
# physical sensor effect. Encode the TARGET only, after the bracket is built.
# --------------------------------------------------------------------------- #
_PU21_P = (0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627,
           0.09150303166, 0.9099517204, 596.3148142)


def pu21_encode_np(lin, l_peak=1000.0):
    """Linear [0,1] -> PU21, normalised so peak white maps to 1.0."""
    p1, p2, p3, p4, p5, p6, p7 = _PU21_P
    y = np.clip(np.asarray(lin, dtype=np.float64) * l_peak, 0.005, None)
    yp = y ** p4
    v = np.maximum(p7 * (((p1 + p2 * yp) / (1 + p3 * yp)) ** p5 - p6), 0.0)
    yq = float(l_peak) ** p4
    vmax = p7 * (((p1 + p2 * yq) / (1 + p3 * yq)) ** p5 - p6)
    return np.clip(v / vmax, 0.0, 1.0).astype(np.float32)


def pu21_decode_np(enc, l_peak=1000.0):
    """Inverse of pu21_encode_np, for turning predictions back into linear."""
    p1, p2, p3, p4, p5, p6, p7 = _PU21_P
    yq = float(l_peak) ** p4
    vmax = p7 * (((p1 + p2 * yq) / (1 + p3 * yq)) ** p5 - p6)
    v = np.clip(np.asarray(enc, dtype=np.float64), 0.0, 1.0) * vmax
    r = (v / p7 + p6) ** (1.0 / p5)          # = (p1 + p2*y^p4)/(1 + p3*y^p4)
    yp = np.clip((p1 - r) / (p3 * r - p2), 0.0, None)
    y = yp ** (1.0 / p4)
    return np.clip(y / l_peak, 0.0, 1.0).astype(np.float32)


ENCODINGS = {"none": lambda x: np.asarray(x, dtype=np.float32),
             "pu21": pu21_encode_np}
