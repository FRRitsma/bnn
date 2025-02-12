import torch
from torch import nn as nn
from src.improved_model import (
    BinarizingNetwork,
    BinarizingConv2d,
    BinarizingLinear,
    binarizing_activation,
)


class BinarizingCIFAR(nn.Module, BinarizingNetwork):
    def __init__(self):
        nn.Module.__init__(self)
        BinarizingNetwork.__init__(self)

        # First layer (real-valued conv)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding="same")

        # Fully binarized convolutional layers
        self.conv2 = BinarizingConv2d(64, 128, kernel_size=3, stride=1, padding="same")
        self.conv3 = BinarizingConv2d(128, 256, kernel_size=3, stride=1, padding="same")
        self.conv4 = BinarizingConv2d(256, 512, kernel_size=3, stride=1, padding="same")

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Fully connected layers
        self.fc1 = BinarizingLinear(512, 512)  # Flattened output from conv layers
        self.fc3 = BinarizingLinear(512, N_OUTPUT_CLASSES)

    def forward(self, x):
        # First conv layer (not binarized)
        x = self.max_pool(
            binarizing_activation(
                self.conv1(x), self.model_mode, self.scramble_distance
            )
        )
        # Binarized convolutional layers
        x = self.max_pool(
            binarizing_activation(
                self.conv2(x), self.model_mode, self.scramble_distance
            )
        )
        x = self.max_pool(
            binarizing_activation(
                self.conv3(x), self.model_mode, self.scramble_distance
            )
        )
        x = self.max_pool(
            binarizing_activation(
                self.conv4(x), self.model_mode, self.scramble_distance
            )
        )
        x = self.average_pool(x)
        # Flatten before fully connected layers
        x = torch.flatten(x, start_dim=1)
        # Fully connected layers (binarized)
        x = binarizing_activation(self.fc1(x), self.model_mode, self.scramble_distance)
        x = self.fc3(x)
        return x


N_OUTPUT_CLASSES: int = 10
