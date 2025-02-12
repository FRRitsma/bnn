from enum import Enum, auto
from functools import wraps
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import conv2d

MAX_SCRAMBLE_DISTANCE: float = 2.05
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
    # TODO: Settings for binarize_output True/False, binarize_weights True/False
    scramble_distance: float = 0.0
    model_mode: ModelMode = ModelMode.scramble
    binarize_parameters: bool
    binarize_output: bool
    scale: Union[nn.Parameter, float]

    def __init__(self, binarize_parameters: bool, binarize_output: bool):
        if binarize_output:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = 1.0

        self.binarize_parameters = binarize_parameters
        self.binarize_output = binarize_output

    @apply_to_child_networks
    def binarize_weights(self):
        # TODO: Implement the effect of binarize_output, binarize_parameters

        if hasattr(self, "weight") and hasattr(self, "bias"):
            self.weight: nn.Parameter = nn.Parameter(
                binary_sign(self.weight.detach()).to(torch.float), requires_grad=False
            )
            self.bias: nn.Parameter = nn.Parameter(
                torch.floor(self.bias.detach()).to(torch.float), requires_grad=False
            )

    @property
    def transformed_weight(self):
        if not self.binarize_parameters:
            return self.weight

        if self.model_mode == ModelMode.binarized:
            return nn.Parameter(
                binary_sign(self.weight.detach()).to(torch.float), requires_grad=False
            )
        return binarizing_weight_activation(
            self.weight, self.model_mode, self.scramble_distance
        )

    @property
    def transformed_bias(self):
        if not self.binarize_parameters:
            return self.bias

        if self.model_mode == ModelMode.binarized:
            return nn.Parameter(
                torch.floor(self.bias.detach()).to(torch.float), requires_grad=False
            )
        return self.bias

    @apply_to_child_networks
    def clamp_parameters(self):
        if hasattr(self, "weight") and hasattr(self, "bias"):
            self.weight.data.clamp_(min=epsilon - 1, max=1 - epsilon)
            self.scale.data.clamp_(min=1.0)

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

    @property
    def _child_networks(self) -> list:
        return [
            getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith("_")
            and isinstance(getattr(self, attr, None), BinarizingNetwork)
        ]

    def set_scramble_distance(
        self, add_scramble_distance: float, decay_rate: float
    ) -> None:
        assert decay_rate < 1
        for child_network in self._child_networks:
            add_scramble_distance = child_network._inner_set_scramble_distance(
                add_scramble_distance, decay_rate
            )

    def _inner_set_scramble_distance(
        self, add_scramble_distance: float, decay_rate: float
    ) -> float:
        if self.scramble_distance >= MAX_SCRAMBLE_DISTANCE:
            return add_scramble_distance
        else:
            self.scramble_distance = min(
                self.scramble_distance + add_scramble_distance, MAX_SCRAMBLE_DISTANCE
            )
            return add_scramble_distance * decay_rate


class BinarizingLinear(nn.Linear, BinarizingNetwork):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        binarize_parameters: bool,
        binarize_output: bool,
    ):
        nn.Linear.__init__(self, in_features, out_features)
        BinarizingNetwork.__init__(self, binarize_parameters, binarize_output)

    def forward(self, x):
        output = torch.matmul(x, self.transformed_weight.t()) + self.transformed_bias
        output = output * self.scale
        if self.binarize_output:
            output = binarizing_activation(
                output, self.model_mode, self.scramble_distance
            )
        return output


class BinarizingConv2d(nn.Conv2d, BinarizingNetwork):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        binarize_parameters: bool,
        binarize_output: bool,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
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
        BinarizingNetwork.__init__(self, binarize_parameters, binarize_output)

    def forward(self, x):
        output = conv2d(
            x,
            self.transformed_weight,
            self.transformed_bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        output = output * self.scale
        if self.binarize_output:
            output = binarizing_activation(
                output, self.model_mode, self.scramble_distance
            )
        return output


def scaled_sigmoid(tensor: Tensor) -> Tensor:
    return 2 * torch.sigmoid(tensor) - 1


def binary_sign(tensor: Tensor) -> Tensor:
    return torch.where(tensor >= 0, torch.tensor(1), torch.tensor(-1))


def binarizing_activation(
    tensor: Tensor, model_mode: ModelMode, scramble_distance: float
) -> Tensor:
    match model_mode:
        case ModelMode.scramble:
            return scaled_sigmoid(
                (tensor * 4) + random_plus_or_minus(tensor) * scramble_distance
            )
        case ModelMode.clean:
            return scaled_sigmoid(tensor * 4)
        case ModelMode.binarized:
            return binary_sign(tensor).to(torch.float)


def binarizing_weight_activation(
    tensor: Tensor, model_mode: ModelMode, scramble_distance: float
) -> Tensor:
    match model_mode:
        case ModelMode.scramble:
            return torch.clamp(
                tensor + random_plus_or_minus(tensor) * scramble_distance, -1, 1
            )
        case ModelMode.clean:
            return torch.clamp(tensor, -1, 1)
        case ModelMode.binarized:
            return binary_sign(tensor).to(torch.float)


def random_plus_or_minus(tensor: torch.Tensor) -> torch.Tensor:
    return (
        torch.bernoulli(torch.empty_like(tensor, requires_grad=False).fill_(0.5)) * 2
        - 1
    )


def random_one_or_zero(tensor: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(torch.empty_like(tensor).fill_(0.5))


def random_continuous(tensor: torch.Tensor, scramble_distance: float) -> torch.Tensor:
    return torch.empty_like(tensor).uniform_(-scramble_distance, scramble_distance)
