"""HDR-aware quality metrics, intended for the linear-RGB pipeline.

Includes:

* ``PUPSNRMetric`` -- PSNR after Aydin et al. (2008) PU encoding.

* ``PUSSIMMetric`` -- SSIM computed on PU-encoded inputs.

* ``CosineDistanceMetric`` -- mean :math:`1 - \\cos(\\hat{x}_i, x_i)`
  over per-pixel RGB direction, as in ExpandNet (Marnerides et al.
  2018). Penalises colour casts, including in dark regions.

PU encoding here uses the Aydin et al. (2008) logarithmic
approximation:
    pu(L) = log10(318 * L + 1) / log10(319),    L in [0, 1].
which maps relative linear luminance to a perceptually uniform range
in [0, 1].
"""

import torch
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure,
)

from .base_metric import BaseMetric


_PU_DENOM = float(torch.log10(torch.tensor(319.0)).item())


def pu_encode(linear: torch.Tensor) -> torch.Tensor:
    """Aydin et al. (2008) PU-like log encoding.

    Maps a tensor of relative linear-RGB luminance values in
    :math:`[0, 1]` to a perceptually uniform range in :math:`[0, 1]`.
    """
    linear = linear.clamp(min=1e-8, max=1.0)
    return torch.log10(318.0 * linear + 1.0) / _PU_DENOM


def _pu_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    bs = pred.size(0)
    p = pu_encode(pred).reshape(bs, -1)
    t = pu_encode(target).reshape(bs, -1)
    mse = torch.mean((p - t) ** 2, dim=1).clamp(min=1e-12)
    return 20.0 * torch.log10(max_val / torch.sqrt(mse))


class PUPSNRMetric(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self) -> None:
        self.value = 0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = self.preprocess(pred, target)
        v = _pu_psnr(pred, target)
        self.value += v.sum()
        self.count += pred.size(0)

    def compute(self) -> float:
        return self.value / self.count


class PUSSIMMetric(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none")

    def reset(self) -> None:
        self.value = 0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = self.preprocess(pred, target)
        self.ssim = self.ssim.to(pred.device)
        v = self.ssim(pu_encode(pred), pu_encode(target))
        self.value += v.sum()
        self.count += pred.size(0)

    def compute(self) -> float:
        return self.value / self.count


class PUMSSSIMMetric(BaseMetric):
    """Multi-scale SSIM on PU-encoded inputs. Requires images of at least
    ~160 pixels per side."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset()
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0, reduction="none"
        )

    def reset(self) -> None:
        self.value = 0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = self.preprocess(pred, target)
        self.ms_ssim = self.ms_ssim.to(pred.device)
        v = self.ms_ssim(pu_encode(pred), pu_encode(target))
        self.value += v.sum()
        self.count += pred.size(0)

    def compute(self) -> float:
        return self.value / self.count


# --------------------------------------------------------------------------- #
# PU21 (Mantiuk & Azimi 2021) -- a different curve from the Aydin 2008 one
# used by the PU* metrics above. Both are called "PU"; they are not
# interchangeable.
# --------------------------------------------------------------------------- #
class PU21PSNRMetric(BaseMetric):
    """PSNR on PU21-encoded values, in either of two conventions.

    convention="normalized" (default) -- pu21_encode to [0,1] with l_peak,
        PSNR against peak 1.0.
    convention="reference" -- as gfxdisp/pu21 pu21_metric.m: absolute cd/m^2
        clamped to [0.005, 10000], raw encode, PSNR against peak 256. Use this
        when comparing to published PU21-PSNR values.

    The two cannot be converted into each other, since the normalised variant
    also clips at l_peak where the reference clips at 10000.
    """

    def __init__(self, *args, l_peak: float = 1000.0,
                 convention: str = "normalized", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if convention not in ("normalized", "reference"):
            raise ValueError(f"bad convention: {convention}")
        self.l_peak = l_peak
        self.convention = convention
        self.reset()

    def reset(self) -> None:
        self.value = 0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        from rawdiffusion.utils import PU21_PEAK, pu21_encode, pu21_encode_metric

        pred, target = self.preprocess(pred, target)
        enc, peak = ((pu21_encode_metric, PU21_PEAK)
                     if self.convention == "reference" else (pu21_encode, 1.0))
        p = enc(pred.clamp(0, 1), self.l_peak)
        t = enc(target.clamp(0, 1), self.l_peak)
        mse = ((p - t) ** 2).flatten(1).mean(dim=1).clamp(min=1e-12)
        self.value += (10.0 * torch.log10(peak**2 / mse)).sum()
        self.count += pred.size(0)

    def compute(self) -> float:
        return self.value / self.count


class PU21SSIMMetric(BaseMetric):
    """SSIM on PU21-encoded values."""

    def __init__(self, *args, l_peak: float = 1000.0,
                 convention: str = "normalized", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        if convention not in ("normalized", "reference"):
            raise ValueError(f"bad convention: {convention}")
        self.l_peak = l_peak
        self.convention = convention
        # data_range must match the encoder's output scale
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=256.0 if convention == "reference" else 1.0)
        self.reset()

    def reset(self) -> None:
        self.value = 0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        from rawdiffusion.utils import pu21_encode, pu21_encode_metric

        pred, target = self.preprocess(pred, target)
        enc = (pu21_encode_metric if self.convention == "reference"
               else pu21_encode)
        p = enc(pred.clamp(0, 1), self.l_peak)
        t = enc(target.clamp(0, 1), self.l_peak)
        self.ssim = self.ssim.to(p.device)
        self.value += self.ssim(p, t) * p.size(0)
        self.count += p.size(0)

    def compute(self) -> float:
        return self.value / self.count


class PU21VSIMetric(BaseMetric):
    """VSI (Zhang et al. 2014, Visual Saliency-Induced index) on PU21-encoded
    values.

    VSI is computed with `piq`'s internals (see vsi_ref.py). Higher is better,
    and the range is [0, 1] under both conventions: `convention` only changes
    the encoder's scale, not the output range.
    """

    def __init__(self, *args, l_peak: float = 1000.0,
                 convention: str = "normalized", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if convention not in ("normalized", "reference"):
            raise ValueError(f"bad convention: {convention}")
        try:
            import piq  # noqa: F401
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "PU21VSIMetric needs `piq` (pip install piq)."
            ) from exc
        self.l_peak = l_peak
        self.convention = convention
        self.reset()

    def reset(self) -> None:
        self.value = 0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        from rawdiffusion.utils import pu21_encode, pu21_encode_metric

        from .vsi_ref import vsi_raw_pu

        pred, target = self.preprocess(pred, target)
        enc = (pu21_encode_metric if self.convention == "reference"
               else pu21_encode)
        p = enc(pred.clamp(0, 1), self.l_peak)
        t = enc(target.clamp(0, 1), self.l_peak)
        # PU units are passed without rescaling, as `pu21_metric.m` calls
        # `m_vsi(P, T)` on the encoder's own output and `m_vsi.m` applies no
        # rescaling. `vsi_raw_pu` is piq's VSI without its input rescaling;
        # see vsi_ref.py.
        v = vsi_raw_pu(p, t, reduction="none")
        self.value += v.sum()
        self.count += p.size(0)

    def compute(self) -> float:
        return self.value / self.count


# --------------------------------------------------------------------------- #
# mu-law tonemapped PSNR/SSIM (Kalantari & Ramamoorthi 2017), also reported as
# PSNR-mu / SSIM-mu.
#
#     T(H) = log(1 + mu*H) / log(1 + mu),   mu = 5000,  H in [0, 1]
#
# A third curve alongside Aydin-PU and PU21; none of the three is
# interchangeable with the others.
#
# Comparing with published PSNR-mu / SSIM-mu values also requires the same
# target normalisation, not just the same metric.
# --------------------------------------------------------------------------- #
_MU = 5000.0


def mu_law(t, mu: float = _MU):
    t = t.clamp(0.0, 1.0)
    return torch.log1p(mu * t) / torch.log1p(torch.tensor(mu, dtype=t.dtype,
                                                          device=t.device))


class MuLawPSNRMetric(BaseMetric):
    def __init__(self, *args, mu: float = _MU, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.reset()

    def reset(self) -> None:
        self.value = 0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = self.preprocess(pred, target)
        p, t = mu_law(pred, self.mu), mu_law(target, self.mu)
        mse = ((p - t) ** 2).flatten(1).mean(dim=1).clamp(min=1e-12)
        self.value += (10.0 * torch.log10(1.0 / mse)).sum()
        self.count += pred.size(0)

    def compute(self) -> float:
        return self.value / self.count


class MuLawSSIMMetric(BaseMetric):
    def __init__(self, *args, mu: float = _MU, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        self.mu = mu
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.reset()

    def reset(self) -> None:
        self.value = 0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = self.preprocess(pred, target)
        p, t = mu_law(pred, self.mu), mu_law(target, self.mu)
        self.ssim = self.ssim.to(p.device)
        self.value += self.ssim(p, t) * p.size(0)
        self.count += p.size(0)

    def compute(self) -> float:
        return self.value / self.count


class CosineDistanceMetric(BaseMetric):
    def __init__(self, *args, eps: float = 1e-8, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.value = 0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = self.preprocess(pred, target)
        # pred, target: (B, 3, H, W)
        dot = (pred * target).sum(dim=1)
        p_norm = pred.pow(2).sum(dim=1).clamp(min=self.eps).sqrt()
        t_norm = target.pow(2).sum(dim=1).clamp(min=self.eps).sqrt()
        cos_sim = dot / (p_norm * t_norm)
        per_image = (1.0 - cos_sim).flatten(1).mean(dim=1)
        self.value += per_image.sum()
        self.count += pred.size(0)

    def compute(self) -> float:
        return self.value / self.count
