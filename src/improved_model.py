from enum import Enum, auto

import torch
import torch.nn as nn
from torch import sigmoid, sign

from src.model import scramble_activation, scaled_sigmoid, random_plus_or_zero
from src.utils import device

OUT_CHANNELS: int = 8


class ModelMode(Enum):
    train = auto()
    evaluation = auto()


class BinarizingNetwork:
    model_mode: ModelMode
    scramble: bool
    scramble_distance: float

    def __init__(self, scramble_distance: float = 0.0):
        self.model_mode = ModelMode.train
        self.scramble = True
        self.scramble_distance = scramble_distance

    def train_mode(self):
        self.model_mode = ModelMode.train
        self.scramble = True
        self._set_child_modes()

    def eval_mode(self):
        self.model_mode = ModelMode.evaluation
        self.scramble = False
        self._set_child_modes()

    def set_scramble_distance(self, scramble_distance: float):
        self.scramble_distance = scramble_distance
        self._set_child_modes()

    def _set_child_modes(self):
        for attribute_name in dir(self):
            if attribute_name.startswith("_"):
                continue
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, BinarizingNetwork):
                match self.model_mode:
                    case ModelMode.train:
                        attribute.train_mode()
                    case ModelMode.evaluation:
                        attribute.eval_mode()
                attribute.set_scramble_distance(self.scramble_distance)


class BinarizingLinear(nn.Linear, BinarizingNetwork):
    def __init__(self, in_features, out_features, scramble_distance: float = 0):
        nn.Linear.__init__(self, in_features, out_features)
        BinarizingNetwork.__init__(self, scramble_distance)
        self.scale = nn.Parameter(torch.ones(1))

    @property
    def transformed_weight(self):
        if self.scramble:
            return scaled_sigmoid(
                self.weight
                - self.scramble_distance
                * sign(self.weight)
                * random_plus_or_zero(self.weight.size()).to(device)
            )
        else:
            return sign(self.weight)

    def forward(self, x):
        output = torch.matmul(x, self.transformed_weight.t()) + self.bias
        output = output * self.scale
        return output


class BinarizingCNN(nn.Module, BinarizingNetwork):
    def __init__(self):
        nn.Module.__init__(self)
        BinarizingNetwork.__init__(self)
        # A single convolutional layer
        self.layer1 = nn.Conv2d(
            in_channels=1, out_channels=OUT_CHANNELS, kernel_size=5, stride=5, padding=1
        )
        # Flatten layer
        self.flatten = nn.Flatten()
        # A fully connected layer
        self.layer2 = BinarizingLinear(OUT_CHANNELS * 36, 77)
        self.layer3 = nn.Linear(77, 10)

    def float_to_binary_layer(self, x):
        y = scramble_activation(self.layer1(x), self.scramble, self.scramble_distance)
        y = self.flatten(y)
        return y

    def second_layer(self, x):
        y = scramble_activation(self.layer2(x), self.scramble, self.scramble_distance)
        return y

    def third_layer(self, x):
        y = sigmoid(self.layer3(x))
        return y

    def forward(self, x):
        y = self.float_to_binary_layer(x)
        y = self.second_layer(y)
        y = self.third_layer(y)
        return y


class SimpleCNN_1(nn.Module):
    scramble: bool = True
    scramble_distance: float

    def train_mode(self):
        self.scramble = True

    def eval_mode(self):
        self.scramble = False

    def float_to_binary_layer(self, x):
        y = scramble_activation(self.layer1(x), self.scramble, self.scramble_distance)
        y = self.flatten(y)
        return y

    def second_layer(self, x):
        y = scramble_activation(self.layer2(x), self.scramble, self.scramble_distance)
        return y

    def third_layer(self, x):
        y = sigmoid(self.layer3(x))
        return y

    def __init__(self, scramble_distance: float = 2.0):
        super(SimpleCNN_1, self).__init__()
        self.scramble_distance = scramble_distance
        # A single convolutional layer
        self.layer1 = nn.Conv2d(
            in_channels=1, out_channels=OUT_CHANNELS, kernel_size=5, stride=5, padding=1
        )
        # Flatten layer
        self.flatten = nn.Flatten()
        # A fully connected layer
        self.layer2 = nn.Linear(OUT_CHANNELS * 36, 54)
        self.layer3 = nn.Linear(54, 10)

    def forward(self, x):
        y = self.float_to_binary_layer(x)
        y = self.second_layer(y)
        y = self.third_layer(y)
        return y
