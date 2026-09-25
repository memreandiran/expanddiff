"""Spatial attention fusion: route each pixel between model prediction(s) and
the guidance.

Softmax over per-pixel logits guarantees the weights sum to 1, so the output
stays a convex combination of its inputs and cannot drift out of range.

Two arities:
  n_sources=2  [x_hat, L0]              -> on top of a single-head model
  n_sources=3  [x_minus, L0, x_plus]    -> the two-specialist setup

All tensors are expected in [-1, 1], the same convention the U-Net uses.
"""
import torch
import torch.nn as nn


class SpatialAttentionFusion(nn.Module):
    def __init__(self, n_sources=2, width=32, use_soft_mask=True):
        """
        Args:
            n_sources: how many images are being blended (2 or 3).
            width: hidden channels.
            use_soft_mask: append 2 channels derived from the guidance marking
                how close it is to the clip limits, as a soft ramp. This is
                computed from the input only.
        """
        super().__init__()
        if n_sources not in (2, 3):
            raise ValueError("n_sources must be 2 or 3")
        self.n_sources = n_sources
        self.use_soft_mask = use_soft_mask
        in_ch = 3 * n_sources + (2 if use_soft_mask else 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width // 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, n_sources, 3, padding=1),
        )
        # Start near "pass the guidance through": zero the last conv so all
        # logits are equal at init, then bias the guidance slot upward.
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.zero_()
            self.net[-1].bias[self._guidance_slot()] = 2.0

    def _guidance_slot(self):
        # [x_hat, L0] -> 1 ;  [x_minus, L0, x_plus] -> 1
        return 1

    @staticmethod
    def soft_clip_mask(guidance, thr=0.05):
        """Two channels in [0,1] from guidance in [-1,1]: proximity to the
        upper and lower clip limits, ramped over `thr` of the range."""
        g01 = (guidance + 1.0) * 0.5
        m = g01.amax(dim=1, keepdim=True)
        hi = ((m - 1.0 + thr) / thr).clamp(0.0, 1.0)
        lo = ((thr - g01.amin(dim=1, keepdim=True)) / thr).clamp(0.0, 1.0)
        return torch.cat([hi, lo], dim=1)

    def forward(self, sources, guidance):
        """
        Args:
            sources: list of `n_sources` tensors (B,3,H,W) in [-1,1], in the
                order the module was configured for, guidance included.
            guidance: the L0 tensor, used for the soft mask only.
        Returns:
            fused (B,3,H,W), weights (B,n_sources,H,W)
        """
        if len(sources) != self.n_sources:
            raise ValueError(f"expected {self.n_sources} sources, "
                             f"got {len(sources)}")
        x = torch.cat(sources, dim=1)
        if self.use_soft_mask:
            x = torch.cat([x, self.soft_clip_mask(guidance)], dim=1)
        w = torch.softmax(self.net(x), dim=1)
        fused = sum(w[:, i:i + 1] * sources[i] for i in range(self.n_sources))
        return fused, w
