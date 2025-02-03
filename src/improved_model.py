from enum import Enum, auto
from functools import wraps

import torch
import torch.nn as nn
from torch import Tensor, Size
from torch.nn.functional import conv2d

OUT_CHANNELS: int = 8


def apply_to_child_networks(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        for child in self._child_networks:
            getattr(child, method.__name__)(*args, **kwargs)
        return result

    return wrapper


class ModelMode(Enum):
    train = auto()
    evaluation = auto()


class BinarizingNetwork:
    model_mode: ModelMode
    scramble_distance: float

    def __init__(self, scramble_distance: float = 0.0):
        self.model_mode = ModelMode.train
        self.scramble_distance = scramble_distance

    @apply_to_child_networks
    def binarize_weights(self):
        if hasattr(self, "weight") and hasattr(self, "bias"):
            self.weight: nn.Parameter = nn.Parameter(
                binary_sign(self.weight.detach()).to(torch.float), requires_grad=False
            )
            self.bias: nn.Parameter = nn.Parameter(
                torch.floor(self.bias.detach()).to(torch.float), requires_grad=False
            )

    @apply_to_child_networks
    def train_mode(self):
        if hasattr(self, "train"):
            self.train()
        self.model_mode = ModelMode.train

    @apply_to_child_networks
    def eval_mode(self):
        if hasattr(self, "eval"):
            self.eval()
        self.model_mode = ModelMode.evaluation

    @apply_to_child_networks
    def set_scramble_distance(self, scramble_distance: float):
        self.scramble_distance = scramble_distance

    @property
    def _child_networks(self) -> list:
        return [
            getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith("_")
            and isinstance(getattr(self, attr, None), BinarizingNetwork)
        ]


class BinarizingLinear(nn.Linear, BinarizingNetwork):
    def __init__(self, in_features, out_features, scramble_distance: float = 0):
        nn.Linear.__init__(self, in_features, out_features)
        BinarizingNetwork.__init__(self, scramble_distance)
        self.scale = nn.Parameter(torch.ones(1))

    @property
    def transformed_weight(self):
        return binarizing_activation(
            self.weight, self.model_mode, self.scramble_distance
        )

    def forward(self, x):
        output = torch.matmul(x, self.transformed_weight.t()) + self.bias
        output = output * self.scale
        return output


class BinarizingConv2d(nn.Conv2d, BinarizingNetwork):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        scramble_distance: float = 0,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        BinarizingNetwork.__init__(self, scramble_distance)
        self.scale = nn.Parameter(torch.ones(1))

    @property
    def transformed_weight(self):
        return binarizing_activation(
            self.weight, self.model_mode, self.scramble_distance
        )

    def forward(self, x):
        output = conv2d(
            x,
            self.transformed_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        output = output * self.scale
        return output


def scaled_sigmoid(tensor: Tensor) -> Tensor:
    return 2 * torch.sigmoid(tensor) - 1


def binary_sign(tensor: Tensor) -> Tensor:
    return torch.where(tensor >= 0, torch.tensor(1), torch.tensor(-1))


def binarizing_activation(
    tensor: Tensor, model_mode: ModelMode, scramble_distance: float
) -> Tensor:
    # Scramble the tensor if enabled
    match model_mode:
        case ModelMode.train:
            return scaled_sigmoid(
                tensor + (scramble_distance * random_plus_or_minus(tensor))
            )
        case ModelMode.evaluation:
            return binary_sign(tensor).to(torch.float)


def random_plus_or_minus(tensor: Tensor) -> Tensor:
    size: Size = tensor.size()
    return (torch.randint(0, 2, size).float() * 2 - 1).to(tensor.device)


def random_plus_or_zero(tensor: Tensor) -> Tensor:
    size: Size = tensor.size()
    return (torch.randint(0, 1, size).float()).to(tensor.device)
