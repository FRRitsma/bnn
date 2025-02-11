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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)

        # Fully binarized convolutional layers
        self.conv2 = BinarizingConv2d(32, 32, kernel_size=3, stride=2, padding=0)
        self.conv3 = BinarizingConv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv4 = BinarizingConv2d(32, 32, kernel_size=3, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = BinarizingLinear(800, 800)  # Flattened output from conv layers
        self.fc2 = BinarizingLinear(800, 800)  # Flattened output from conv layers
        self.fc3 = BinarizingLinear(800, N_OUTPUT_CLASSES)

    def forward(self, x):
        # First conv layer (not binarized)
        x = binarizing_activation(
            self.conv1(x), self.model_mode, self.scramble_distance
        )
        # Binarized convolutional layers
        x = binarizing_activation(
            self.conv2(x), self.model_mode, self.scramble_distance
        )
        x = binarizing_activation(
            self.conv3(x), self.model_mode, self.scramble_distance
        )
        x = binarizing_activation(
            self.conv4(x), self.model_mode, self.scramble_distance
        )

        # Flatten before fully connected layers
        x = torch.flatten(x, start_dim=1)
        # Fully connected layers (binarized)
        x = binarizing_activation(self.fc1(x), self.model_mode, self.scramble_distance)
        x = binarizing_activation(self.fc2(x), self.model_mode, self.scramble_distance)
        x = self.fc3(x)
        return x


N_OUTPUT_CLASSES: int = 10
