from torch import nn as nn

from src.improved_model import (
    BinarizingNetwork,
    OUT_CHANNELS,
    BinarizingLinear,
    binarizing_activation,
)


class BinarizingCNN(nn.Module, BinarizingNetwork):
    def __init__(self):
        nn.Module.__init__(self)
        BinarizingNetwork.__init__(self)
        self.layer1 = nn.Conv2d(
            in_channels=1, out_channels=OUT_CHANNELS, kernel_size=5, stride=5, padding=1
        )
        self.flatten = nn.Flatten()
        self.layer2 = BinarizingLinear(OUT_CHANNELS * 36, 77)
        self.layer3 = BinarizingLinear(77, 10)

    def float_to_binary_layer(self, x):
        y = binarizing_activation(
            self.layer1(x), self.model_mode, self.scramble_distance
        )
        y = self.flatten(y)
        return y

    def second_layer(self, x):
        y = binarizing_activation(
            self.layer2(x), self.model_mode, self.scramble_distance
        )
        return y

    def third_layer(self, x):
        y = binarizing_activation(
            self.layer3(x), self.model_mode, self.scramble_distance
        )
        return y

    def forward(self, x):
        y = self.float_to_binary_layer(x)
        y = self.second_layer(y)
        y = self.third_layer(y)
        return y
