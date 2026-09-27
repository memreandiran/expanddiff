"""Target encodings for the data loader, numpy only.

PU21 encoding is bounded in [0,1], perceptually uniform and monotonic, so the
model can predict PU-encoded values with the same tanh head. It is applied to
the target only, after any crop or flip; decode predictions at inference.
"""
import numpy as np

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
