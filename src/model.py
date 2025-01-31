import torch
from torch import Tensor, Size


def scaled_sigmoid(tensor: Tensor) -> Tensor:
    return 2 * torch.sigmoid(tensor) - 1


def scramble_activation(
    tensor: Tensor, scramble: bool, scramble_distance: float
) -> Tensor:
    """
    Custom activation function with optional scrambling.
    Args:
        tensor (Tensor): Input tensor.
        scramble (bool): Whether to apply scrambling.
        scramble_distance (float): Magnitude of scrambling.

    Returns:
        Tensor: Transformed tensor.
    """
    # Scramble the tensor if enabled
    if scramble:
        tensor = tensor + (scramble_distance * random_plus_or_minus(tensor))
        return scaled_sigmoid(tensor)
    else:
        return torch.sign(tensor)


def random_plus_or_minus(tensor: Tensor) -> Tensor:
    size: Size = tensor.size()
    return (torch.randint(0, 2, size).float() * 2 - 1).to(tensor.device)


def random_plus_or_zero(tensor: Tensor) -> Tensor:
    size: Size = tensor.size()
    return (torch.randint(0, 1, size).float()).to(tensor.device)
