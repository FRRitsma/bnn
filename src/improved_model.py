import torch.nn as nn
from torch import sigmoid

from src.model import scramble_activation

OUT_CHANNELS: int = 8


class SimpleCNN(nn.Module):
    scramble: bool = True
    scramble_distance: float

    def train_mode(self):
        self.scramble = True

    def eval_mode(self):
        self.scramble = False

    def first_layer(self, x):
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
        super(SimpleCNN, self).__init__()
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
        y = self.first_layer(x)
        y = self.second_layer(y)
        y = self.third_layer(y)
        return y
