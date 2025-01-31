import torch
from torch import nn as nn, Tensor, Size


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
        tensor = tensor + (
            scramble_distance * random_plus_or_minus(tensor.size()).to(tensor.device)
        )
        return scaled_sigmoid(tensor)
    else:
        return torch.sign(tensor)


class ScrambleLayer(nn.Linear):
    scramble: bool
    scramble_distance: float

    def __init__(self, scramble_distance: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scramble_distance = scramble_distance
        self.scramble = True

    def train_mode(self):
        self.scramble = True

    def eval_mode(self):
        self.scramble = False

    def scramble_function(self, tensor: Tensor) -> Tensor:
        if self.scramble:
            return tensor + (
                self.scramble_distance
                * random_plus_or_minus(tensor.size()).to(tensor.device)
            )
        return tensor

    def matrix_multiplication(self, *args, **kwargs) -> Tensor:
        return super().forward(*args, **kwargs)

    @staticmethod
    def activation_function(tensor: Tensor) -> Tensor:
        return torch.sigmoid(tensor)

    def forward(self, *args, **kwargs) -> Tensor:
        x = self.matrix_multiplication(*args, **kwargs)
        if self.scramble:
            x = self.scramble_function(x)
        return 2 * self.activation_function(x) - 1


def random_plus_or_minus(size: Size) -> Tensor:
    return torch.randint(0, 2, size).float() * 2 - 1


def random_plus_or_zero(size: Size) -> Tensor:
    return torch.randint(0, 1, size).float()
