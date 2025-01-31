import torch
from torch import Tensor, Size

from src.improved_model import ModelMode


def scaled_sigmoid(tensor: Tensor) -> Tensor:
    return 2 * torch.sigmoid(tensor) - 1


def binarizing_activation(
    tensor: Tensor, model_mode: ModelMode, scramble_distance: float
) -> Tensor:
    # Scramble the tensor if enabled
    match model_mode:
        case ModelMode.train:
            tensor = tensor + (scramble_distance * random_plus_or_minus(tensor))
            return scaled_sigmoid(tensor)
        case ModelMode.evaluation:
            torch.sign(tensor)


def random_plus_or_minus(tensor: Tensor) -> Tensor:
    size: Size = tensor.size()
    return (torch.randint(0, 2, size).float() * 2 - 1).to(tensor.device)


def random_plus_or_zero(tensor: Tensor) -> Tensor:
    size: Size = tensor.size()
    return (torch.randint(0, 1, size).float()).to(tensor.device)
