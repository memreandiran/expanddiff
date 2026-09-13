"""HDR-aware quality metrics.

Three new metrics intended for the linear-RGB pipeline:

* ``PUPSNRMetric`` -- PSNR after Aydin et al. (2008) PU encoding.
  Standard linear-space PSNR is dominated by errors in the bright
  regions because most linear-RGB values are tiny and the absolute
  squared error is largest where the magnitudes are largest. PU encoding
  approximates the human contrast sensitivity function so that an
  equal numerical error contributes roughly the same regardless of the
  underlying luminance level.

* ``PUSSIMMetric`` -- SSIM computed on PU-encoded inputs.

* ``CosineDistanceMetric`` -- mean :math:`1 - \\cos(\\hat{x}_i, x_i)`
  over per-pixel RGB direction. Designed (per ExpandNet,
  Marnerides et al. 2018) to penalise colour casts in dark regions
  that L1/L2 barely touch, because the absolute pixel magnitude is too
  small for the linear-space loss to notice.

PU encoding here uses the Aydin et al. (2008) logarithmic
approximation:
    pu(L) = log10(318 * L + 1) / log10(319),    L in [0, 1].
which maps relative linear luminance to a perceptually uniform range
in [0, 1]. This is the form most commonly cited in HDR vision/graphics
papers and matches what ExpandNet uses before reporting PSNR/SSIM.
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
    """Multi-scale SSIM on PU-encoded inputs. Matches ExpandNet's
    paper-reported MS-SSIM (computed after PU encoding). Requires images
    of at least ~160 pixels per side -- fine for the d2/d4 test
    resolutions used in this project."""

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
# PU21 (Mantiuk & Azimi 2021) -- the encoding the AIM 2025 ITM challenge ranks
# on, and a DIFFERENT curve from the Aydin 2008 one used by the PU* metrics
# above. Both are called "PU"; they are not interchangeable, so these are
# reported as separate columns rather than replacing anything.
#
# CONVENTION, and the caveat that goes with any comparison to AIM 2025: our
# pu21_encode normalises to [0, 1] with l_peak = 1000 cd/m^2, so PSNR is taken
# against a peak of 1.0. Mantiuk's reference pu21_metric uses its own peak
# convention on unnormalised PU units, so absolute values here will NOT line up
# with the challenge's 29.22 dB figure until that convention is confirmed and
# matched. What these ARE good for: comparing our own runs against each other
# on the curve the field now ranks with, and -- for target_encoding=pu21 runs
# -- reading almost exactly as PSNR on the training target.
# --------------------------------------------------------------------------- #
class PU21PSNRMetric(BaseMetric):
    """PSNR on PU21-encoded values, in either of two conventions.

    convention="normalized" (default) -- pu21_encode to [0,1] with l_peak,
        PSNR against peak 1.0. Self-consistent, and for target_encoding=pu21
        runs it reads almost exactly as PSNR on the training target. This is
        the basis to keep using for the LEDiff comparison so those numbers
        stay on one scale.
    convention="reference" -- matches gfxdisp/pu21 pu21_metric.m exactly:
        absolute cd/m^2 clamped to [0.005, 10000], RAW encode, PSNR against
        peak 256. Use this and only this when comparing to AIM 2025 or to any
        published PU21-PSNR.

    The two differ by roughly a constant offset but not exactly, since the
    normalised variant also clips at l_peak where the reference clips at
    10000 -- so they cannot be converted into each other post hoc.
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
    """SSIM on PU21-encoded values -- the challenge's second ranking metric."""

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

    Why this exists: the SI-HDR benchmark paper (Hanji et al., SIGGRAPH '22,
    §7.2) recommends exactly four metrics -- PU21-PSNR, **PU21-VSI**,
    HDR-VDP-3 and PU21-PIQE -- and names PU21-VSI and HDR-VDP-3 as the two
    best-performing against their subjective data. It also says explicitly
    "We do not recommend using PU21-SSIM", which is what we had been reporting.

    VSI comes from `piq`; it is not reimplemented here. Higher is better, and
    the range is [0, 1] for both conventions because VSI is a similarity index
    rather than an error measure -- so unlike PU21-PSNR/SSIM the `convention`
    only changes the encoder's scale, not the output range.
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
        # RAW PU units, NOT rescaled. `pu21_metric.m` calls `m_vsi(P, T)` on the
        # encoder's own output, which reaches 420 at a 1000 cd/m^2 peak, and
        # `m_vsi.m` applies no rescaling: its similarity constants (1.27, 386,
        # 130) are absolute. `piq.vsi` instead does `x = x * 255 / data_range`,
        # so calling it with data_range = enc(1.0) = 420.1 compressed the signal
        # by 0.607 while the constants stayed put -- pushing every similarity
        # ratio toward 1 and reading VSI HIGH by up to +0.0049, with the bias
        # GROWING with the error being measured. `vsi_raw_pu` is piq's body with
        # those two rescale lines removed. See vsi_ref.py; verified against
        # MATLAB R2024b running the reference m_vsi.m.
        v = vsi_raw_pu(p, t, reduction="none")
        self.value += v.sum()
        self.count += p.size(0)

    def compute(self) -> float:
        return self.value / self.count


# --------------------------------------------------------------------------- #
# mu-law tonemapped PSNR/SSIM -- the headline metric in the Kalantari lineage
# (Kalantari & Ramamoorthi 2017) and what ExpoCM reports as PSNR-mu / SSIM-mu.
#
#     T(H) = log(1 + mu*H) / log(1 + mu),   mu = 5000,  H in [0, 1]
#
# A THIRD curve alongside Aydin-PU and PU21. All three are "perceptual
# encodings before PSNR" and none is interchangeable with the others, so they
# are reported as separate columns.
#
# NOTE on comparability: ExpoCM applies NO alignment before scoring. Matching
# their numbers therefore requires matching their TARGET NORMALISATION too, not
# just the metric -- our median-anchored targets are on a different absolute
# scale than whatever HDR-EYE/HDR-REAL ship. Metric parity is necessary but not
# sufficient.
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
