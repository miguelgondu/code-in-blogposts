from dataclasses import dataclass

import torch

from batch_bo.utils.constants import LIMITS


@dataclass
class Dataset:
    X: torch.Tensor
    y: torch.Tensor

    @property
    def min_max_scaled_X(self):
        min_, max_ = LIMITS
        return (self.X - min_) / (max_ - min_)

    @property
    def standardized_y(self):
        return (self.y - self.y.mean()) / self.y.std() if self.y.std() > 0 else self.y
