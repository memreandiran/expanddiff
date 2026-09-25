"""PU21-PIQE: a no-reference HDR quality metric.

Lower is better. Roughly: <20 excellent, 20-35 good, 35-50 fair, >50 poor.

Two stages.

1. PU21 encoding (Mantiuk & Azimi 2021). It maps absolute luminance to a
   perceptually uniform scale on which an equal numerical difference is
   equally visible regardless of level.

2. PIQE (Venkatanath et al. 2015), a no-reference score built from block-wise
   distortion criteria on a mean-subtracted contrast-normalised (MSCN) map.

`pyiqa` ships PIQE and is used when importable; the local implementation below
is the fallback. The implementation used is stored in `backend`.
"""

import numpy as np
import torch

from .base_metric import BaseMetric

# PU21 "banding_glare" parameters, verbatim from the reference implementation
# (gfxdisp/pu21, matlab/pu21_encoder.m). banding_glare is the default variant.
#
#   V = max( p7 * ( ((p1 + p2*Y^p4) / (1 + p3*Y^p4))^p5 - p6 ), 0 )
#
# Y is luminance in nits, valid over roughly 0.005 to 10000, and V spans
# about 0 to 600.
_PU21 = (0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627,
         0.09150303166, 0.9099517204, 596.3148142)
_PU21_YMIN = 0.005     # nits; below this the fit is not defined


def pu21_encode(lin, l_peak=1000.0):
    """Linear RGB in [0,1] -> PU21 units, normalised to roughly [0,1].

    `l_peak` is the display peak luminance, in cd/m^2, that the relative values
    are taken to represent. It must be held fixed across everything being
    compared.
    """
    p1, p2, p3, p4, p5, p6, p7 = _PU21
    y = torch.clamp(lin * l_peak, min=_PU21_YMIN)
    yp = torch.pow(y, p4)
    v = torch.clamp(p7 * (torch.pow((p1 + p2 * yp) / (1 + p3 * yp), p5) - p6),
                    min=0.0)
    # Normalise so peak white maps to 1.0. The divisor depends only on l_peak.
    yq = float(l_peak) ** p4
    vmax = max(p7 * (((p1 + p2 * yq) / (1 + p3 * yq)) ** p5 - p6), 1e-8)
    return torch.clamp(v / vmax, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# PIQE (Venkatanath et al. 2015)
# --------------------------------------------------------------------------- #
def _mscn(x, eps=1e-8):
    """Mean-subtracted contrast-normalised coefficients, 7x7 Gaussian, sigma
    7/6, matching the reference implementation."""
    k = 7
    sigma = 7.0 / 6.0
    ax = torch.arange(k, dtype=x.dtype, device=x.device) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    # separable, applied as two conv2d passes over the (N, 1, H, W) input
    gh = g.view(1, 1, 1, k)
    gv = g.view(1, 1, k, 1)
    pad = k // 2

    def blur(t):
        t = torch.nn.functional.pad(t, (pad, pad, 0, 0), mode="reflect")
        t = torch.nn.functional.conv2d(t, gh)
        t = torch.nn.functional.pad(t, (0, 0, pad, pad), mode="reflect")
        return torch.nn.functional.conv2d(t, gv)

    mu = blur(x)
    sigma_map = torch.sqrt(torch.clamp(blur(x * x) - mu * mu, min=0.0))
    return (x - mu) / (sigma_map + 1.0), sigma_map


def piqe(img, block=16, activity_thr=0.1, deg_thr=0.5):
    """No-reference PIQE score for a single-channel image in [0,1].

    Blocks with enough local activity are scored for two artefact types,
    blockiness and Gaussian-noise-like distortion, and the final score is the
    mean over active blocks mapped to [0,100]. Blocks that are too flat carry
    no information and are excluded.
    """
    if img.dim() == 2:
        img = img[None, None]
    elif img.dim() == 3:
        img = img[None]
    h, w = img.shape[-2:]
    h2, w2 = h - h % block, w - w % block
    if h2 < block or w2 < block:
        return float("nan")
    img = img[..., :h2, :w2]

    mscn, _ = _mscn(img * 255.0)
    m = mscn[0, 0]
    scores, n_active = [], 0
    for i in range(0, h2, block):
        for j in range(0, w2, block):
            blk = m[i:i + block, j:j + block]
            var = blk.var(unbiased=False).item()
            if var <= activity_thr:
                continue                      # flat block, no information
            n_active += 1
            # blockiness: energy at the block borders vs the interior
            border = torch.cat([blk[0], blk[-1], blk[:, 0], blk[:, -1]])
            interior = blk[1:-1, 1:-1].reshape(-1)
            b = (border.abs().mean() /
                 (interior.abs().mean() + 1e-8)).item() if interior.numel() else 1.0
            # noise: how far the block deviates from the MSCN unit-variance
            # assumption a clean natural image satisfies
            nz = abs(var - 1.0)
            d = min(1.0, max(0.0, (b - 1.0) * deg_thr + nz * deg_thr))
            scores.append(d)
    if not scores:
        return float("nan")
    return float(100.0 * np.mean(scores))


class PU21PIQEMetric(BaseMetric):
    """No-reference: `target` is accepted and ignored, so it fits the
    CollectionMetric interface.

    Argument order matters for this metric: it scores the first argument.
    sample.py calls update(target_data, sample), which would make it score the
    ground truth. Callers must pass (pred, target)."""

    def __init__(self, l_peak=1000.0, min_value=0, max_value=1):
        super().__init__(min_value=min_value, max_value=max_value)
        self.l_peak = l_peak
        self.value = 0.0
        self.count = 0
        self.backend = "local-piqe"
        self._pyiqa = None
        try:                                   # prefer the reference impl
            import pyiqa

            self._pyiqa = pyiqa.create_metric("piqe", as_loss=False)
            self.backend = "pyiqa-piqe"
        except Exception:                      # noqa: BLE001
            pass

    def reset(self):
        self.value = 0.0
        self.count = 0

    def update(self, pred, target=None):
        pred = pred if pred.dim() == 4 else pred[None]
        enc = pu21_encode(pred, self.l_peak)
        for i in range(enc.shape[0]):
            if self._pyiqa is not None:
                s = float(self._pyiqa(enc[i:i + 1]).item())
            else:
                grey = (0.2126 * enc[i, 0] + 0.7152 * enc[i, 1]
                        + 0.0722 * enc[i, 2])
                s = piqe(grey)
            if np.isfinite(s):
                self.value += s
                self.count += 1

    def compute(self):
        return float("nan") if self.count == 0 else self.value / self.count
