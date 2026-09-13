import torch


class BaseMetric:
    def __init__(self, min_value=0, max_value=1):
        self.min_value = min_value
        self.max_value = max_value

    def check_value(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if (pred < self.min_value).any() or (pred > self.max_value).any():
            print(
                f"Predicted value is not in range [{self.min_value}, {self.max_value}], got {pred.min()} and {pred.max()}"
            )
        if (target < self.min_value).any() or (target > self.max_value).any():
            print(
                f"Target value is not in range [{self.min_value}, {self.max_value}], got {target.min()} and {target.max()}"
            )

    def preprocess(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.check_value(pred, target)
        return pred, target
