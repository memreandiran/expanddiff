"""VSI on raw PU21 units, following `pu21_metric.m`'s calling convention.

`pu21_metric.m` does `m_vsi(P_test, P_reference)` where P comes straight out of
`pu21_encoder.encode`, i.e. raw PU units that reach 420 at a 1000 cd/m^2 peak
and ~596 at 10,000. `m_vsi.m` applies no rescaling: its three similarity maps
carry absolute additive constants (constForVS 1.27, constForGM 386,
constForChrom 130).

`vsi_raw_pu` is `piq.vsi`'s body without its `x = x * 255 / data_range` input
rescaling, and uses piq's internals. piq's default constants equal m_vsi.m's.

The result is not identical to `m_vsi.m`: the 256x256 downsample inside SDSP
uses torch's bilinear `interpolate` instead of `imresize`. For reference values
use `vsi_ref_cells.py`, which runs `m_vsi.m` itself.
"""
from __future__ import annotations

import torch
from torch.nn.functional import avg_pool2d, pad


def vsi_raw_pu(x: torch.Tensor, y: torch.Tensor, reduction: str = "mean",
               c1: float = 1.27, c2: float = 386.0, c3: float = 130.0,
               alpha: float = 0.4, beta: float = 0.02, omega_0: float = 0.021,
               sigma_f: float = 1.34, sigma_d: float = 145.0,
               sigma_c: float = 0.001) -> torch.Tensor:
    """VSI of two (N,3,H,W) tensors already in raw PU21 units.

    No data_range: the values are passed to the similarity maps unscaled, which
    is what `pu21_metric.m` does. Do not normalise the inputs first.
    """
    from piq.vsi import (gradient_map, rgb2lmn, scharr_filter, sdsp,
                         similarity_map)
    from piq.utils.common import _reduce

    if x.dim() != 4 or y.dim() != 4 or x.shape != y.shape:
        raise ValueError(f"expected matching (N,C,H,W); got {tuple(x.shape)} "
                         f"and {tuple(y.shape)}")
    if x.size(1) == 1:
        x, y = x.repeat(1, 3, 1, 1), y.repeat(1, 3, 1, 1)

    # piq.vsi rescales to [0,255] here; the reference does not, so that step
    # is omitted.
    vs_x = sdsp(x, data_range=255, omega_0=omega_0, sigma_f=sigma_f,
                sigma_d=sigma_d, sigma_c=sigma_c)
    vs_y = sdsp(y, data_range=255, omega_0=omega_0, sigma_f=sigma_f,
                sigma_d=sigma_d, sigma_c=sigma_c)

    x_lmn, y_lmn = rgb2lmn(x), rgb2lmn(y)

    kernel_size = max(1, round(min(vs_x.size()[-2:]) / 256))
    padding = kernel_size // 2
    if padding:
        pad_to_use = [padding, (kernel_size - 1) // 2] * 2
        vs_x = pad(vs_x, pad=pad_to_use, mode="replicate")
        vs_y = pad(vs_y, pad=pad_to_use, mode="replicate")
        x_lmn = pad(x_lmn, pad=pad_to_use, mode="replicate")
        y_lmn = pad(y_lmn, pad=pad_to_use, mode="replicate")
    vs_x = avg_pool2d(vs_x, kernel_size=kernel_size)
    vs_y = avg_pool2d(vs_y, kernel_size=kernel_size)
    x_lmn = avg_pool2d(x_lmn, kernel_size=kernel_size)
    y_lmn = avg_pool2d(y_lmn, kernel_size=kernel_size)

    sch = scharr_filter(device=x_lmn.device, dtype=x_lmn.dtype)
    kernels = torch.stack([sch, sch.transpose(1, 2)])
    gm_x = gradient_map(x_lmn[:, :1], kernels)
    gm_y = gradient_map(y_lmn[:, :1], kernels)

    s_vs = similarity_map(vs_x, vs_y, c1)
    s_gm = similarity_map(gm_x, gm_y, c2)
    s_c = (similarity_map(x_lmn[:, 1:2], y_lmn[:, 1:2], c3)
           * similarity_map(x_lmn[:, 2:], y_lmn[:, 2:], c3))

    mag, ang = s_c.abs(), torch.atan2(torch.zeros_like(s_c), s_c)
    s_c_real_pow = (mag ** beta) * torch.cos(ang * beta)

    s = s_vs * s_gm.pow(alpha) * s_c_real_pow
    vs_max = torch.max(vs_x, vs_y)
    eps = torch.finfo(vs_max.dtype).eps
    out = ((s * vs_max).sum(dim=(-1, -2)) + eps) / (vs_max.sum(dim=(-1, -2)) + eps)
    return _reduce(out.squeeze(-1), reduction)
