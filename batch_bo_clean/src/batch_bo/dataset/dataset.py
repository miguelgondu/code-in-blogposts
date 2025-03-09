from dataclasses import dataclass

import torch
import numpy as np
import gpjax as gpx

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

    def to_gpx_dataset(self):
        return gpx.Dataset(
            X=self.X.numpy(force=True).astype(np.float64),
            y=self.y.numpy(force=True).astype(np.float64),
        )
