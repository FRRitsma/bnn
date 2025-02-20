import torch
from torch import nn as nn
from src.improved_model import (
    BinarizingConv2dBatchNorm,
)
from src.linear_layer import BinarizingLinearBatchNorm
from src.top_level import BinarizingTopLevel

N_OUTPUT_CLASSES: int = 10


class BinarizingCIFAR(nn.Module, BinarizingTopLevel):
    def __init__(self):
        super(BinarizingCIFAR, self).__init__()
        output_channels: int = 64

        self.conv1 = BinarizingConv2dBatchNorm(
            3,
            output_channels,
            kernel_size=4,
            stride=2,
            padding=0,
            binarize_parameters=False,
        )  # , binarize_output=True

        self.conv2 = BinarizingConv2dBatchNorm(
            output_channels,
            output_channels * 2,
            kernel_size=4,
            stride=2,
            padding=0,
            binarize_parameters=True,
        )  # , binarize_output=True

        self.conv3 = BinarizingConv2dBatchNorm(
            output_channels * 2,
            output_channels * 3,
            kernel_size=4,
            stride=2,
            padding=0,
            binarize_parameters=True,
        )  # , binarize_output=True

        # Fully connected layers
        self.fc1 = BinarizingLinearBatchNorm(
            12 * output_channels,
            12 * output_channels,
            binarize_parameters=True,
            binarize_output=True,
        )
        self.fc2 = BinarizingLinearBatchNorm(
            12 * output_channels,
            N_OUTPUT_CLASSES,
            binarize_parameters=True,
            binarize_output=False,
        )

    def forward(self, x):
        # First conv layer
        x = self.conv1(x)
        x = self.conv1.activation(x)

        # Second conv layer
        x = self.conv2(x)
        x = self.conv2.activation(x)

        # Third conv layer (no more pooling)
        x = self.conv3(x)
        x = self.conv3.activation(x)

        # Flatten before fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        return x
