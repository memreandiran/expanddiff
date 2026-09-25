"""Mergers that take (predicted image, guidance) and output one image.

Both mergers below learn what they need from the (prediction, guidance) pair
alone. Nothing external, such as the clip thresholds, is required at inference.

    LearnedAffineRouter   predicts a rescale and blend weights, then blends.
                          Output is a convex combination of two sources, so
                          pixels routed to the guidance keep its exactness.

    FreeFormMerger        a plain CNN that writes the output pixels itself.
                          It can learn the rescale too, but its output is not
                          constrained to a convex combination.

Everything is in [-1, 1], the convention the U-Net uses.
"""
import torch
import torch.nn as nn


def _soft_clip_mask(guidance, thr=0.05):
    """Two channels in [0,1]: how close the guidance is to its upper and lower
    limits, ramped over `thr` of the range. Derived from the input only."""
    g01 = (guidance + 1.0) * 0.5
    hi = ((g01.amax(1, keepdim=True) - 1.0 + thr) / thr).clamp(0, 1)
    lo = ((thr - g01.amin(1, keepdim=True)) / thr).clamp(0, 1)
    return torch.cat([hi, lo], 1)


def _source_softmax(logits, n_sources, per_channel):
    """Blend weights summing to 1 over the source axis.

    per_channel=False -> logits (B, n_sources, H, W): one weight per source per
        pixel, shared across RGB.
    per_channel=True  -> logits (B, n_sources*3, H, W) viewed as
        (B, n_sources, 3, H, W): an independent weight per source per pixel per
        channel, as in LEDiff ("normalized using a softmax ... to ensure the
        merging weights sum to one across each pixel and channel").

    Returns (B, n_sources, 3, H, W), broadcastable against the sources.
    """
    b, _, h, w = logits.shape
    if per_channel:
        return torch.softmax(logits.view(b, n_sources, 3, h, w), dim=1)
    return torch.softmax(logits, dim=1).unsqueeze(2)


def _trunk(in_ch, width, depth, out_ch):
    layers, c = [], in_ch
    for _ in range(depth):
        layers += [nn.Conv2d(c, width, 3, padding=1), nn.ReLU(inplace=True)]
        c = width
    layers += [nn.Conv2d(c, out_ch, 3, padding=1)]
    return nn.Sequential(*layers)


class LearnedAffineRouter(nn.Module):
    """out = W1 * x_hat + W2 * (gain * L0 + bias),  W = softmax, so convex.

    The affine (gain, bias) is estimated globally per image via spatial
    pooling, as 6 numbers (3 gain, 3 bias), so no clip thresholds are needed.

    For stops [-12,-6];[-4,0] the exact mapping on [-1,1] tensors is
    gain = t_hi - t_lo = 0.248 and bias = 2*t_lo + (t_hi-t_lo) - 1 = -0.748.
    In general gain lies in (0, 1] and bias in about (-1, 0], so bias_range
    must exceed 1.

    At initialisation the blend logit for x_hat starts high, so the output
    starts close to x_hat. gain is centred at 1 at init via the inverse-sigmoid
    offset.
    """

    def __init__(self, width=48, depth=4, use_soft_mask=True,
                 gain_range=4.0, bias_range=2.0, per_channel=False):
        super().__init__()
        self.use_soft_mask = use_soft_mask
        self.per_channel = per_channel
        self.gain_range = gain_range
        self.bias_range = bias_range
        in_ch = 6 + (2 if use_soft_mask else 0)
        # shared trunk -> (a) per-pixel blend logits, (b) pooled global affine
        self.trunk = _trunk(in_ch, width, depth, width)
        self.blend_head = nn.Conv2d(width, 2 * (3 if per_channel else 1),
                                    3, padding=1)
        self.affine_head = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(inplace=True),
            nn.Linear(width, 6),
        )
        # centre gain at 1.0 at init: gain_range*sigmoid(x + off) = 1
        self._gain_off = float(torch.logit(torch.tensor(1.0 / gain_range)))
        nn.init.zeros_(self.blend_head.weight)
        nn.init.zeros_(self.affine_head[-1].weight)
        with torch.no_grad():
            self.blend_head.bias.zero_()
            # slot 0 is x_hat; with per_channel the first 3 entries are its
            # three colour channels
            self.blend_head.bias[: (3 if per_channel else 1)] = 3.0
            self.affine_head[-1].bias.zero_()

    def forward(self, x_hat, guidance):
        inp = torch.cat([x_hat, guidance], 1)
        if self.use_soft_mask:
            inp = torch.cat([inp, _soft_clip_mask(guidance)], 1)
        f = self.trunk(inp)
        logits = self.blend_head(f)                       # per pixel
        pooled = f.mean(dim=(2, 3))                       # global statistics
        a = self.affine_head(pooled)                      # (B, 6)
        g_raw = a[:, :3].view(-1, 3, 1, 1)
        b_raw = a[:, 3:].view(-1, 3, 1, 1)
        # gain in (0, gain_range) centred near 1; bias in (-bias_range, +)
        gain = self.gain_range * torch.sigmoid(g_raw + self._gain_off)
        bias = self.bias_range * torch.tanh(b_raw)
        src2 = (gain * guidance + bias).clamp(-1, 1)
        w = _source_softmax(logits, 2, self.per_channel)   # (B,2,3,H,W)
        src = torch.stack([x_hat, src2], dim=1)            # (B,2,3,H,W)
        out = (w * src).sum(dim=1)
        return out, {"weights": w.mean(2), "gain": gain, "bias": bias,
                     "src2": src2}


class FreeFormMerger(nn.Module):
    """A plain CNN that predicts the merged image directly.

    Predicts a residual on top of x_hat, and the last conv is zero-initialised
    so training begins as an exact identity.
    """

    def __init__(self, width=64, depth=6, use_soft_mask=True):
        super().__init__()
        self.use_soft_mask = use_soft_mask
        in_ch = 6 + (2 if use_soft_mask else 0)
        self.net = _trunk(in_ch, width, depth, 3)
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.zero_()

    def forward(self, x_hat, guidance):
        inp = torch.cat([x_hat, guidance], 1)
        if self.use_soft_mask:
            inp = torch.cat([inp, _soft_clip_mask(guidance)], 1)
        out = (x_hat + self.net(inp)).clamp(-1, 1)
        return out, {}


def build_merger(kind, **kw):
    if kind == "affine_router":
        return LearnedAffineRouter(**kw)
    if kind == "freeform":
        return FreeFormMerger(**kw)
    if kind == "specialist_router":
        return SpecialistRouter(**kw)
    raise ValueError(f"unknown merger: {kind}")


class SpecialistRouter(nn.Module):
    """Fuse the two specialists, optionally with the guidance as a third source.

    The analogue of LEDiff's F(C-, C0, C+), which merges a low exposure, the
    original, and a high exposure. Here:

        x_minus  highlight specialist, trained on clip(L, t_lo, 1)
        src_mid  guidance mapped into the target's scale (learned affine)
        x_plus   shadow specialist,    trained on clip(L, 0, t_hi)

    No un-normalisation is needed for the specialists: absolute_scale targets
    mean both already live in L's units. Between them they cover the whole
    range, since every pixel is either above t_lo (where x_minus's target
    equals L) or below t_hi (where x_plus's does), and midtones by both.

    Output is a convex combination of the sources. The affine for the guidance
    is global (6 numbers), as in LearnedAffineRouter.
    """

    def __init__(self, width=96, depth=8, use_soft_mask=True,
                 use_guidance=True, gain_range=4.0, bias_range=2.0,
                 per_channel=False):
        super().__init__()
        self.use_soft_mask = use_soft_mask
        self.per_channel = per_channel
        self.use_guidance = use_guidance
        self.n_sources = 3 if use_guidance else 2
        self.gain_range, self.bias_range = gain_range, bias_range
        in_ch = 9 + (2 if use_soft_mask else 0)   # x_minus, x_plus, guidance
        self.trunk = _trunk(in_ch, width, depth, width)
        self.blend_head = nn.Conv2d(
            width, self.n_sources * (3 if per_channel else 1), 3, padding=1)
        self.affine_head = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(inplace=True), nn.Linear(width, 6))
        self._gain_off = float(torch.logit(torch.tensor(1.0 / gain_range)))
        nn.init.zeros_(self.blend_head.weight)
        nn.init.zeros_(self.affine_head[-1].weight)
        with torch.no_grad():
            self.blend_head.bias.zero_()      # equal weights at init
            self.affine_head[-1].bias.zero_()

    def forward(self, x_minus, x_plus, guidance):
        inp = torch.cat([x_minus, x_plus, guidance], 1)
        if self.use_soft_mask:
            inp = torch.cat([inp, _soft_clip_mask(guidance)], 1)
        f = self.trunk(inp)
        w = _source_softmax(self.blend_head(f), self.n_sources,
                            self.per_channel)              # (B,n,3,H,W)
        a = self.affine_head(f.mean(dim=(2, 3)))
        gain = self.gain_range * torch.sigmoid(
            a[:, :3].view(-1, 3, 1, 1) + self._gain_off)
        bias = self.bias_range * torch.tanh(a[:, 3:].view(-1, 3, 1, 1))
        src_mid = (gain * guidance + bias).clamp(-1, 1)

        srcs = [x_minus, x_plus] + ([src_mid] if self.use_guidance else [])
        out = (w * torch.stack(srcs, dim=1)).sum(dim=1)
        return out, {"weights": w.mean(2), "gain": gain, "bias": bias,
                     "src2": src_mid}
