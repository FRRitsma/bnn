import torch

from torch import nn as nn

from settings import settings
from src.improved_model import (
    BinarizingNetwork,
    binarizing_activation,
    BinarizingConv2d,
    BinarizingLinear,
)
from src.training import get_accuracy, OneHotEncode, num_classes

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from src.utils import device

# Define transformations for data augmentation and normalization
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Download and load CIFAR-10 dataset
batch_size = 64  # Define batch size

# Load the training dataset
train_dataset = torchvision.datasets.CIFAR10(
    root=settings.data_path / "CIFAR",
    train=True,
    download=True,
    transform=transform,
    target_transform=OneHotEncode(num_classes),
)

# Split into training and validation sets (e.g., 80% train, 20% validation)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Load the test dataset
test_dataset = torchvision.datasets.CIFAR10(
    root=settings.data_path / "CIFAR",
    train=False,
    download=True,
    transform=transform,
    target_transform=OneHotEncode(num_classes),
)

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class BinarizingCIFAR(nn.Module, BinarizingNetwork):
    def __init__(self, scramble_distance: float):
        nn.Module.__init__(self)
        BinarizingNetwork.__init__(self, scramble_distance)

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1
        )  # Input: 3 channels, Output: 32 channels
        self.conv2 = BinarizingConv2d(
            32, 12, kernel_size=6, stride=3, padding=1
        )  # Input: 32 channels, Output: 64 channels

        # Flatten layer
        # No explicit flatten layer in PyTorch; we flatten in the forward pass.

        # Fully connected layers
        self.fc1 = BinarizingLinear(1200, 400)
        self.fc2 = BinarizingLinear(400, 10)  # Output: 10 classes for CIFAR-10

    def forward(self, x):
        # Convolutional layers
        x = binarizing_activation(
            self.conv1(x), self.model_mode, self.scramble_distance
        )  # Apply ReLU after first conv layer
        x = binarizing_activation(
            self.conv2(x), self.model_mode, self.scramble_distance
        )  # Apply ReLU after second conv layer

        # Flatten the output for fully connected layers
        x = x.view(-1, 1200)  # Flatten to (batch_size, 64 * 32 * 32)

        # Fully connected layers
        x = binarizing_activation(
            self.fc1(x), self.model_mode, self.scramble_distance
        )  # Apply ReLU after first fully connected layer
        x = self.fc2(
            x
        )  # Output layer (no activation, as CrossEntropyLoss will handle it)

        return x


if __name__ == "__main__":
    model = BinarizingCIFAR(scramble_distance=0.01)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(1e-2))

    num_epochs: int = 200
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        accuracy: float = get_accuracy(model, val_loader)
        print(accuracy)
