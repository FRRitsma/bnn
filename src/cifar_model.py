import torch
from torch import nn as nn
from src.improved_model import (
    BinarizingNetwork,
    BinarizingConv2d,
    BinarizingLinear,
)


class BinarizingCIFAR(nn.Module, BinarizingNetwork):
    def __init__(self):
        nn.Module.__init__(self)
        output_channels: int = 64

        # First layer (real-valued conv)
        self.conv1 = BinarizingConv2d(
            3,
            1 * output_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            binarize_parameters=False,
            binarize_output=True,
        )
        self.bn1 = nn.BatchNorm2d(1 * output_channels)  # BatchNorm after conv1

        # Fully binarized convolutional layers
        self.conv2 = BinarizingConv2d(
            1 * output_channels,
            2 * output_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            binarize_parameters=True,
            binarize_output=True,
        )
        self.bn2 = nn.BatchNorm2d(2 * output_channels)  # BatchNorm after conv2

        self.conv3 = BinarizingConv2d(
            2 * output_channels,
            4 * output_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            binarize_parameters=True,
            binarize_output=True,
        )
        self.bn3 = nn.BatchNorm2d(4 * output_channels)  # BatchNorm after conv3

        self.conv4 = BinarizingConv2d(
            4 * output_channels,
            8 * output_channels,
            kernel_size=3,
            stride=1,
            padding="same",
            binarize_parameters=True,
            binarize_output=True,
        )
        self.bn4 = nn.BatchNorm2d(8 * output_channels)  # BatchNorm after conv4

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Fully connected layers
        self.fc1 = BinarizingLinear(
            8 * output_channels,
            4 * output_channels,
            binarize_parameters=True,
            binarize_output=True,
        )
        self.fc2 = BinarizingLinear(
            4 * output_channels,
            N_OUTPUT_CLASSES,
            binarize_parameters=True,
            binarize_output=False,
        )

    def forward(self, x):
        # First conv layer (not binarized)
        x = self.conv1(x)
        x = self.bn1(x)  # Apply BatchNorm
        x = self.max_pool(x)

        # Binarized convolutional layers
        x = self.conv2(x)
        x = self.bn2(x)  # Apply BatchNorm
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.bn3(x)  # Apply BatchNorm
        x = self.max_pool(x)

        x = self.conv4(x)
        x = self.bn4(x)  # Apply BatchNorm
        x = self.max_pool(x)

        x = self.average_pool(x)

        # Flatten before fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers (binarized)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


N_OUTPUT_CLASSES: int = 10
