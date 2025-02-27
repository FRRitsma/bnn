from torch import nn as nn

from src.improved_model import (
    BinarizingConv2dBatchNorm,
)
from src.linear_layer import BinarizingLinearBatchNorm
from src.top_level import BinarizingTopLevel

OUT_CHANNELS: int = 8


class MNIST_CNN(nn.Module, BinarizingTopLevel):
    def __init__(self):
        nn.Module.__init__(self)
        # Convolutional layer
        self.layer1 = BinarizingConv2dBatchNorm(
            in_channels=1,
            out_channels=OUT_CHANNELS,
            kernel_size=4,
            stride=2,
            padding=0,
            binarize_parameters=False,
        )
        self.flatten = nn.Flatten()

        # Dropout before the first linear layer
        self.dropout1 = nn.Dropout(p=0.0)

        # First linear layer
        self.layer3 = BinarizingLinearBatchNorm(
            1352, 50, binarize_parameters=False, binarize_output=True
        )

        # Dropout before the second linear layer
        self.dropout2 = nn.Dropout(p=0.5)

        # Second linear layer
        self.layer4 = BinarizingLinearBatchNorm(
            50, 10, binarize_parameters=False, binarize_output=False
        )

    def forward(self, x):
        # Convolutional layer
        x = self.layer1.activation(self.layer1(x))
        x = self.flatten(x)

        # Dropout before the first linear layer
        x = self.dropout1(x)

        # First linear layer
        x = self.layer3.activation(self.layer3(x))

        # Dropout before the second linear layer
        x = self.dropout2(x)

        # Second linear layer
        x = self.layer4.activation(self.layer4(x))

        return x
