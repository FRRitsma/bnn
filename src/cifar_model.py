import torch
from torch import nn as nn
from src.improved_model import (
    BinarizingConv2dBatchNorm,
)
from src.linear_layer import BinarizingLinear, BinarizingLinearBatchNorm
from src.top_level import BinarizingTopLevel

N_OUTPUT_CLASSES: int = 10


class BinarizingCIFAR(nn.Module, BinarizingTopLevel):
    def __init__(self):
        nn.Module.__init__(self)
        output_channels: int = 32

        # First layer (real-valued conv)
        self.conv1 = BinarizingConv2dBatchNorm(
            3,
            1 * output_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            binarize_parameters=False,
        )

        # Fully binarized convolutional layers
        self.conv2 = BinarizingConv2dBatchNorm(
            1 * output_channels,
            2 * output_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            binarize_parameters=True,
        )

        self.conv3 = BinarizingConv2dBatchNorm(
            2 * output_channels,
            4 * output_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            binarize_parameters=True,
        )

        self.conv4 = BinarizingConv2dBatchNorm(
            4 * output_channels,
            8 * output_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            binarize_parameters=True,
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # Fully connected layers
        self.fc1 = BinarizingLinearBatchNorm(
            8 * output_channels,
            8 * output_channels,
            binarize_parameters=True,
            binarize_output=True,
        )
        self.fc2 = BinarizingLinear(
            8 * output_channels,
            N_OUTPUT_CLASSES,
            binarize_parameters=True,
            binarize_output=False,
        )

    def forward(self, x):
        # First conv layer (not binarized)
        x = self.conv1(x)
        x = self.conv1.activation(x)
        x = self.max_pool(x)

        # Binarized convolutional layers
        x = self.conv2(x)
        x = self.conv2.activation(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.conv3.activation(x)
        x = self.max_pool(x)

        x = self.conv4(x)
        x = self.conv4.activation(x)
        x = self.max_pool(x)

        x = self.global_pool(x)

        # Flatten before fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers (binarized)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
