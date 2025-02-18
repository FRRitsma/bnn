from enum import Enum, auto
from functools import wraps

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import conv2d

MAX_SCRAMBLE_DISTANCE: float = 2.1
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


class BinarizingBase:
    # TODO: Settings for binarize_output True/False, binarize_weights True/False
    scramble_distance: float = 0.0
    _finished_training: bool = False
    model_mode: ModelMode = ModelMode.scramble
    binarize_parameters: bool
    binarize_output: bool

    def __init__(self, binarize_parameters: bool, binarize_output: bool):
        self.binarize_parameters = binarize_parameters
        self.binarize_output = binarize_output

    @property
    def is_dead(self) -> bool:
        return self.scramble_distance >= MAX_SCRAMBLE_DISTANCE

    @property
    def _child_networks(self) -> list[nn.Module]:
        return [
            getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith("_")
            and issubclass(getattr(self, attr, int), nn.Module)
        ]

    @apply_to_child_networks
    def eval(self):
        self.eval()

    @apply_to_child_networks
    def train(self):
        self.train()

    def binarize_weights(self):
        # TODO: Implement the effect of binarize_output, binarize_parameters
        if (
            hasattr(self, "weight")
            and hasattr(self, "bias")
            and self.binarize_parameters
        ):
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

    def train_mode(self):
        if hasattr(self, "train"):
            if self._finished_training:
                self.binary_mode()
                return
            else:
                self.train()
        self.model_mode = ModelMode.scramble

    def clean_mode(self):
        # Applies the forward pass without added noise
        if hasattr(self, "eval"):
            self.eval()
            if self._finished_training:
                self.binary_mode()
                return
        self.model_mode = ModelMode.clean

    def binary_mode(self):
        if hasattr(self, "eval"):
            self.eval()
        self.model_mode = ModelMode.binarized

    def inner_set_scramble_distance(
        self, add_scramble_distance: float, decay_rate: float
    ) -> float:
        if self.scramble_distance >= MAX_SCRAMBLE_DISTANCE:
            self.binary_mode()
            self._finished_training = True
            return add_scramble_distance
        else:
            self.scramble_distance = min(
                self.scramble_distance + add_scramble_distance, MAX_SCRAMBLE_DISTANCE
            )
            return add_scramble_distance * decay_rate


class BinarizingConv2dBatchNorm(nn.Conv2d, BinarizingBase):
    batch_norm: nn.BatchNorm2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        binarize_parameters: bool,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
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
            bias=False,
        )
        BinarizingBase.__init__(self, binarize_parameters, binarize_output=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = conv2d(
            x,
            self.transformed_weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        x = self.batch_norm(x)
        return x

    def activation(self, x):
        if self.binarize_output:
            x = binarizing_activation(x, self.model_mode, self.scramble_distance)
        return x

    def eval(self):
        super().eval()
        self.batch_norm.weight.requires_grad = False
        self.batch_norm.bias.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.batch_norm.weight.requires_grad = False
        self.batch_norm.bias.requires_grad = False


class BinarizingConv2d(nn.Conv2d, BinarizingBase):
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
        BinarizingBase.__init__(self, binarize_parameters, binarize_output)

    def forward(self, x):
        x = conv2d(
            x,
            self.transformed_weight,
            self.transformed_bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x

    def activation(self, x):
        if self.binarize_output:
            x = binarizing_activation(x, self.model_mode, self.scramble_distance)
        return x


def scaled_sigmoid(tensor: Tensor) -> Tensor:
    return torch.sigmoid(tensor) * 2 - 1


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
            return scaled_sigmoid(
                (tensor * 4) + random_plus_or_minus(tensor) * scramble_distance
            )
        case ModelMode.clean:
            return scaled_sigmoid(tensor * 4)
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
