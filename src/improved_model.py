from enum import Enum, auto
from functools import wraps

import torch
import torch.nn as nn
from torch import Tensor, clamp
from torch.nn.functional import conv2d

OUT_CHANNELS: int = 8

epsilon: float = float(1e-6)


def apply_to_child_networks(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        for child in self._child_networks:
            getattr(child, method.__name__)(*args, **kwargs)
        return result

    return wrapper


class ModelMode(Enum):
    scramble = auto()
    clean = auto()
    binarized = auto()


class BinarizingNetwork:
    model_mode: ModelMode
    scramble_distance: float

    # TODO: Post init set_scramble_distance on self
    def __init__(self, scramble_distance: float = 0.0):
        self.model_mode = ModelMode.scramble
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

    @property
    def transformed_weight(self):
        if self.model_mode == ModelMode.binarized:
            return nn.Parameter(
                binary_sign(self.weight.detach()).to(torch.float), requires_grad=False
            )
        return binarizing_activation(
            self.weight, self.model_mode, self.scramble_distance
        )

    @property
    def transformed_bias(self):
        if self.model_mode == ModelMode.binarized:
            return nn.Parameter(
                torch.floor(self.bias.detach()).to(torch.float), requires_grad=False
            )
        return binarizing_activation(
            self.weight, self.model_mode, self.scramble_distance
        )

    @apply_to_child_networks
    def clamp_weights(self):
        if hasattr(self, "weight") and hasattr(self, "bias"):
            self.weight: nn.Parameter = nn.Parameter(  # type: ignore
                clamp(self.weight, -1 + epsilon, 1 - epsilon)
            )

    @apply_to_child_networks
    def scramble_mode(self):
        if hasattr(self, "train"):
            self.train()
        self.model_mode = ModelMode.scramble

    @apply_to_child_networks
    def clean_mode(self):
        if hasattr(self, "eval"):
            self.eval()
        self.model_mode = ModelMode.clean

    @apply_to_child_networks
    def binary_mode(self):
        if hasattr(self, "eval"):
            self.eval()
        self.model_mode = ModelMode.binarized

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
        case ModelMode.scramble:
            return scaled_sigmoid(
                tensor + random_plus_or_minus(tensor) * scramble_distance
            )
        case ModelMode.clean:
            return binarizing_activation(tensor, ModelMode.scramble, 0.0)
        case ModelMode.binarized:
            return binary_sign(tensor).to(torch.float)


def random_plus_or_minus(tensor: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(torch.empty_like(tensor).fill_(0.5)) * 2 - 1


def random_one_or_zero(tensor: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(torch.empty_like(tensor).fill_(0.5))


def random_continuous(tensor: torch.Tensor, scramble_distance: float) -> torch.Tensor:
    return torch.empty_like(tensor).uniform_(-scramble_distance, scramble_distance)
