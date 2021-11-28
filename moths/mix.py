import numpy as np
import torch
from torch import Tensor


def cutmix_batch(x: Tensor, y: Tensor, p: float, a: float):
    if np.random.rand() > p:
        return x, y, y, 1

    l = np.random.beta(a, a)

    batch_size = x.size()[0] // 2
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = l * x + (1 - l) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, l


def mixup_batch(x: Tensor, y: Tensor, p: float, a: float):
    if np.random.rand() > p:
        return x, y, y, 1

    l = np.random.beta(a, a)

    batch_size = x.size()[0] // 2
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = l * x + (1 - l) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, l


def mix_loss(crit, y_hat, y_a, y_b, l) -> Tensor:
    if l == 1:
        return crit(y_hat, y_a)

    return l * crit(y_hat, y_a) + (1 - l) * crit(y_hat, y_b)
