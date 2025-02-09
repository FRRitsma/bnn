import torch

from torch import nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

from src.utils import device

train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ]
)

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
cifar_train = torchvision.datasets.CIFAR10(
    root=settings.data_path / "CIFAR",
    train=True,
    download=True,
    transform=train_transforms,
)
# Move dataset to the device
train_data = torch.stack([cifar_train[i][0] for i in range(len(cifar_train))]).to(
    device
)
train_labels = torch.tensor(
    [cifar_train[i][1] for i in range(len(cifar_train))], device=device
)

# Wrap in a TensorDataset for DataLoader compatibility
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

# Load the test dataset
cifar_test = torchvision.datasets.CIFAR10(
    root=settings.data_path / "CIFAR",
    train=False,
    download=True,
    transform=transform,
)
test_data = torch.stack([cifar_test[i][0] for i in range(len(cifar_test))]).to(device)
test_labels = torch.tensor(
    [cifar_test[i][1] for i in range(len(cifar_test))], device=device
)

# Wrap in a TensorDataset for DataLoader compatibility
test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)


class BinarizingCIFAR(nn.Module, BinarizingNetwork):
    def __init__(self, scramble_distance: float, n_output_classes=10):
        nn.Module.__init__(self)
        BinarizingNetwork.__init__(self, scramble_distance)

        # First layer (real-valued conv)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)

        # Fully binarized convolutional layers
        self.conv2 = BinarizingConv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = BinarizingConv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv4 = BinarizingConv2d(16, 16, kernel_size=3, stride=2, padding=1)

        # Fully connected layers
        self.fc1 = BinarizingLinear(64, 64)  # Flattened output from conv layers
        # self.fc2 = BinarizingLinear(256, 256)
        self.fc2 = BinarizingLinear(64, n_output_classes)

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
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    starting_scramble_distance: float = 0.05
    # scramble_distance_growth_rate: float = 10 ** (1 / 20)

    model = BinarizingCIFAR(scramble_distance=0.0)
    model.set_scramble_distance(starting_scramble_distance)
    # model.load_state_dict(torch.load(settings.models_path / "cifar_model_v6.pth"))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(1e-3))
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, min_lr=float(1e-3)
    )

    target_accuracy: float = 0.7
    num_epochs: int = 1000

    for epoch in range(num_epochs):
        model.train()
        total_loss: float = 0.0

        indices = torch.randperm(len(train_data), device=device)

        for i in range(0, len(train_data), batch_size):
            batch_indices = indices[i : i + batch_size]  # Select batch indices
            inputs, labels = train_data[batch_indices], train_labels[batch_indices]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        # model.clamp_weights()

        val_accuracy: float = get_accuracy(model, test_loader)
        train_accuracy: float = get_accuracy(model, train_loader)

        scheduler.step(total_loss)
        if epoch > 0:
            if epoch % 10 == 0:
                # Move dataset to the device
                train_data = torch.stack(
                    [cifar_train[i][0] for i in range(len(cifar_train))]
                ).to(device)
                train_labels = torch.tensor(
                    [cifar_train[i][1] for i in range(len(cifar_train))], device=device
                )
            if epoch % 10 == 0:
                save_model(model, "cifar_model_v5")
        update_string: str = ""
        update_string += f"val acc: {val_accuracy:.3f}"
        update_string += f", train acc: {train_accuracy:.3f}, scramble: {model.scramble_distance:.1e}, epoch: {epoch + 1}, lr: {optimizer.param_groups[0]['lr']:.1e}"
        update_string += f" loss: {total_loss:.1f}"
        model.binary_mode()
        binary_accuracy: float = get_accuracy(model, test_loader)
        model.scramble_mode()
        update_string += f" bin acc: {binary_accuracy:.3f}"
        print(update_string)
        if val_accuracy > target_accuracy:
            scramble_distance: float = max(
                min(5.0, model.scramble_distance + 0.05),
                starting_scramble_distance,
            )
            model.set_scramble_distance(scramble_distance)
