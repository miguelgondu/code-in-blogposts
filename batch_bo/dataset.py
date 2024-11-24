from dataclasses import dataclass

import torch


@dataclass
class Dataset:
    X: torch.Tensor
    y: torch.Tensor
