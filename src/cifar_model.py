import torch

from torch import nn as nn

from settings import settings
from src.improved_model import (
    BinarizingNetwork,
    binarizing_activation,
    BinarizingConv2d,
    BinarizingLinear,
)
from src.load_and_save import save_model
from src.training import get_accuracy

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

n_output_classes: int = 10

# Download and load CIFAR-10 dataset
batch_size = 32  # Define batch size

# Load the training dataset
train_dataset = torchvision.datasets.CIFAR10(
    root=settings.data_path / "CIFAR",
    train=True,
    download=True,
    transform=transform,
    # target_transform=OneHotEncode(n_output_classes),
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
    # target_transform=OneHotEncode(n_output_classes),
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = BinarizingConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = BinarizingConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # Pooling layer to reduce dimensionality
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = BinarizingLinear(128 * 4 * 4, 512)
        self.fc2 = BinarizingLinear(512, 256)
        self.fc3 = BinarizingLinear(256, n_output_classes)

    def forward(self, x):
        # Convolutional layers with activation and pooling
        x = self.pool(
            binarizing_activation(
                self.conv1(x), self.model_mode, self.scramble_distance
            )
        )
        x = self.pool(
            binarizing_activation(
                self.conv2(x), self.model_mode, self.scramble_distance
            )
        )
        x = self.pool(
            binarizing_activation(
                self.conv3(x), self.model_mode, self.scramble_distance
            )
        )
        x = torch.flatten(x, start_dim=1)
        # Fully connected layers
        x = binarizing_activation(self.fc1(x), self.model_mode, self.scramble_distance)
        x = binarizing_activation(self.fc2(x), self.model_mode, self.scramble_distance)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = BinarizingCIFAR(scramble_distance=0.0)
    model.to(device)

    criterion = nn.MultiMarginLoss(margin=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(1e-3))
    target_accuracy: float = 0.75
    num_epochs: int = 1000
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
        if accuracy > target_accuracy:
            scramble_distance: float = max(
                min(5.0, model.scramble_distance * 1.05), float(1e-3)
            )
            model.set_scramble_distance(scramble_distance)
        print(
            f"accuracy: {accuracy:.4f}, scramble_distance: {model.scramble_distance:.5f}, epoch: {epoch+1}"
        )
        if epoch % 10 == 0:
            save_model(model, "cifar_model_v3")
