import torch

from torch import nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import v2

from settings import settings
from src.cifar_model import BinarizingCIFAR
from src.load_and_save import save_model
from src.training import get_accuracy

import torchvision
import torchvision.transforms as transforms

from src.utils import device

# augmentation_transforms = v2.Compose(
#     [
#         v2.RandomHorizontalFlip(p=0.5),
#         # v2.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), antialias=True),
#         # v2.RandomRotation(degrees=15),
#         # v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#         # v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
#         # v2.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
#     ]
# )
augmentation_transforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        # v2.RandomCrop(size=(32, 32), padding=2, padding_mode="reflect"),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        v2.RandomAffine(
            degrees=1, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=None
        ),
    ]
)


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Download and load CIFAR-10 dataset
batch_size: int = 512

# Load the training dataset
cifar_train = torchvision.datasets.CIFAR10(
    root=settings.data_path / "CIFAR",
    train=True,
    download=True,
    transform=transform,
)
# Move dataset to the device
train_data = torch.stack([cifar_train[i][0] for i in range(len(cifar_train))]).to(
    device
)
train_labels = torch.tensor(
    [cifar_train[i][1] for i in range(len(cifar_train))], device=device
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


if __name__ == "__main__":
    decay_rate = 0.5

    model = BinarizingCIFAR()
    model.to(device)
    model.set_scramble_distance(0.25, decay_rate)
    starting_lr: float = float(1e-2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=starting_lr)
    patience: int = 10
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=patience, min_lr=float(1e-4)
    )

    target_accuracy: float = 0.8
    num_epochs: int = 1000

    for epoch in range(num_epochs):
        model.train()
        total_loss: float = 0.0

        indices = torch.randperm(len(train_data), device=device)
        augmented_train_data = augmentation_transforms(train_data)
        # augmented_train_data = train_data
        for i in range(0, len(augmented_train_data), batch_size):
            batch_indices = indices[i : i + batch_size]
            inputs, labels = (
                augmented_train_data[batch_indices],
                train_labels[batch_indices],
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())

        # Test data performance:
        model.eval()
        val_accuracy: float = get_accuracy(model, test_data, test_labels)
        train_accuracy: float = get_accuracy(model, train_data, train_labels)
        model.binary_mode()
        binary_accuracy: float = get_accuracy(model, test_data, test_labels)
        model.train_mode()

        scheduler.step(total_loss)
        if epoch > 0 and epoch % 10 == 0:
            save_model(model, "cifar_model_v7")

        update_string: str = ""
        update_string += (
            f"val acc: {val_accuracy:.3f}, train acc: {train_accuracy:.3f},"
        )
        update_string += (
            f" epoch: {epoch + 1}, lr: {optimizer.param_groups[0]['lr']:.1e}"
        )
        update_string += f" loss: {total_loss:.1f}"

        update_string += f" bin acc: {binary_accuracy:.3f}"
        print(update_string)
        if train_accuracy > target_accuracy:
            # Update scheduler:
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=patience, min_lr=float(1e-5)
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = starting_lr / 10
            # Update scramble:
            step_size_scramble: float = 0.25
            model.set_scramble_distance(step_size_scramble, decay_rate)
            print(
                f"{model.conv1.scramble_distance:.2f}, "
                f"{model.conv2.scramble_distance:.2f}, "
                f"{model.conv3.scramble_distance:.2f}, "
                f"{model.conv4.scramble_distance:.2f}, "
                f"{model.fc1.scramble_distance:.2f}, "
                f"{model.fc2.scramble_distance:.2f}, "
            )
