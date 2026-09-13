import torch
from .base_metric import BaseMetric
from rawdiffusion.utils import linear_to_srgb


class LPIPSMetric(BaseMetric):
    """Learned Perceptual Image Patch Similarity (Zhang et al. 2018).

    Inputs are expected in [0, 1] (matching the rest of the metrics in this
    package). The LPIPS backbone (AlexNet/VGG) was trained on sRGB-encoded
    natural images, so when the rest of the pipeline operates in *linear*
    RGB, this metric internally applies the sRGB EOTF-inverse before calling
    the LPIPS network. Disable via `input_is_linear=False` if your inputs
    are already sRGB-encoded. Lower is better.
    """

    def __init__(self, net="alex", input_is_linear=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        import lpips

        self.lpips = lpips.LPIPS(net=net, verbose=False)
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        self.input_is_linear = input_is_linear
        self.reset()

    def reset(self) -> None:
        self.value = torch.tensor(0.0)
        self.count = 0

    def _to_lpips_input(self, x):
        """[0, 1] (linear or sRGB) -> [-1, 1] sRGB-encoded for LPIPS."""
        if self.input_is_linear:
            x = linear_to_srgb(x)
        return x * 2 - 1

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = self.preprocess(pred, target)

        self.lpips = self.lpips.to(pred.device)
        with torch.no_grad():
            value = self.lpips(self._to_lpips_input(pred), self._to_lpips_input(target))

        self.value = self.value.to(value.device) + value.sum()
        self.count += pred.size(0)

    def compute(self):
        if self.count == 0:
            return torch.tensor(float("nan"))
        return self.value / self.count
