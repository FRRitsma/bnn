from torch import nn as nn

from src.improved_model import (
    BinarizingNetwork,
    OUT_CHANNELS,
    BinarizingLinear,
    binarizing_activation,
    BinarizingConv2d,
)


class MNIST_CNN(nn.Module, BinarizingNetwork):
    def __init__(self, scramble_distance: float):
        nn.Module.__init__(self)
        BinarizingNetwork.__init__(self, scramble_distance)
        self.layer1 = nn.Conv2d(
            in_channels=1, out_channels=OUT_CHANNELS, kernel_size=4, stride=2, padding=0
        )
        self.layer2 = BinarizingConv2d(
            in_channels=OUT_CHANNELS,
            out_channels=OUT_CHANNELS,
            kernel_size=4,
            stride=2,
            padding=0,
        )
        self.flatten = nn.Flatten()
        self.layer3 = BinarizingLinear(200, 77)
        self.layer4 = BinarizingLinear(77, 10)

    def first_layer(self, x):
        y = binarizing_activation(
            self.layer1(x), self.model_mode, self.scramble_distance
        )
        return y

    def second_layer(self, x):
        y = binarizing_activation(
            self.layer2(x), self.model_mode, self.scramble_distance
        )
        y = self.flatten(y)
        return y

    def third_layer(self, x):
        y = binarizing_activation(
            self.layer3(x), self.model_mode, self.scramble_distance
        )
        return y

    def fourth_layer(self, x):
        y = binarizing_activation(
            self.layer4(x), self.model_mode, self.scramble_distance
        )
        return y

    def forward(self, x):
        y = self.first_layer(x)
        y = self.second_layer(y)
        y = self.third_layer(y)
        y = self.fourth_layer(y)
        return y
