import torch
from torch import nn as nn, Tensor, Size


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

    # Apply the activation function (sigmoid scaled to [-1, 1])
    return 2 * torch.sigmoid(tensor) - 1


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


class OneScrambleLayerNN(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, scramble_distance: float
    ) -> None:
        super(OneScrambleLayerNN, self).__init__()
        hidden_size: int = 20
        self.layer1 = ScrambleLayer(scramble_distance, input_size, hidden_size)
        self.layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.fc2 = nn.Linear(hidden_size, output_size)

    def train_mode(self) -> None:
        self.layer1.train_mode()

    def eval_mode(self) -> None:
        self.layer1.eval_mode()

    def forward(self, x):
        x = self.fc2(self.layer2(self.layer1(x)))
        return x


class SimpleNN(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, scramble_distance: float
    ) -> None:
        super(SimpleNN, self).__init__()
        hidden_size: int = 20
        self.layer1 = ScrambleLayer(scramble_distance, input_size, hidden_size)
        self.layer2 = ScrambleLayer(scramble_distance, hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def train_mode(self) -> None:
        self.layer1.train_mode()
        self.layer2.train_mode()

    def eval_mode(self) -> None:
        self.layer1.eval_mode()
        self.layer2.eval_mode()

    def forward(self, x):
        x = self.fc2(self.layer2(self.layer1(x)))
        return x
