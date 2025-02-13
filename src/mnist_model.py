from torch import nn as nn

from src.improved_model import (
    BinarizingNetwork,
    BinarizingLinear,
    BinarizingConv2d,
)


class MNIST_CNN(nn.Module, BinarizingNetwork):
    def __init__(self):
        nn.Module.__init__(self)
        self.layer1 = BinarizingConv2d(
            in_channels=1,
            out_channels=OUT_CHANNELS,
            kernel_size=4,
            stride=2,
            padding=0,
            binarize_parameters=False,
            binarize_output=True,
        )
        self.layer2 = BinarizingConv2d(
            in_channels=OUT_CHANNELS,
            out_channels=OUT_CHANNELS,
            kernel_size=4,
            stride=2,
            padding=0,
            binarize_parameters=True,
            binarize_output=True,
        )
        self.flatten = nn.Flatten()
        self.layer3 = BinarizingLinear(
            200, 77, binarize_parameters=True, binarize_output=True
        )
        self.layer4 = BinarizingLinear(
            77, 10, binarize_parameters=True, binarize_output=False
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


OUT_CHANNELS: int = 8
