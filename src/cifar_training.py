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

# Define rich data augmentation
cifar10_transforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),  # Flip 50% of the time
        v2.RandomResizedCrop(
            size=(32, 32), scale=(0.8, 1.0), antialias=True
        ),  # Random crop and resize
        v2.RandomRotation(degrees=15),  # Rotate between -15° and +15°
        v2.RandomAffine(
            degrees=0, translate=(0.1, 0.1)
        ),  # Small random translation (max 10% shift)
        v2.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        ),  # Strong color jitter
        v2.RandomErasing(
            p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)
        ),  # Randomly erase patches
        # v2.ToDtype(torch.float32, scale=True),  # Convert to float and scale to [0,1]
        # v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),  # CIFAR-10 mean/std
    ]
)


# Define transformations for data augmentation and normalization
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Download and load CIFAR-10 dataset
batch_size = 64  # Define batch size

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

if __name__ == "__main__":
    starting_scramble_distance: float = 0.01
    # scramble_distance_growth_rate: float = 10 ** (1 / 20)

    model = BinarizingCIFAR()
    model.set_scramble_distance(starting_scramble_distance)
    # model.load_state_dict(torch.load(settings.models_path / "cifar_model_v7.pth"))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(1e-3))
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, min_lr=float(1e-4)
    )

    target_accuracy: float = 0.8
    num_epochs: int = 1000

    for epoch in range(num_epochs):
        model.train()
        total_loss: float = 0.0

        indices = torch.randperm(len(train_data), device=device)
        augmented_train_data = cifar10_transforms(train_data)
        for i in range(0, len(augmented_train_data), batch_size):
            batch_indices = indices[i : i + batch_size]  # Select batch indices
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
        # model.clamp_weights()

        val_accuracy: float = get_accuracy(model, test_data, test_labels)
        train_accuracy: float = get_accuracy(model, train_data, train_labels)

        scheduler.step(total_loss)
        if epoch > 0:
            if epoch % 10 == 0:
                save_model(model, "cifar_model_v7")
        update_string: str = ""
        update_string += f"val acc: {val_accuracy:.3f}"
        update_string += (
            f", train acc: {train_accuracy:.3f}, scramble: {model.scramble_distance:.1e}, "
            f"epoch: {epoch + 1}, lr: {optimizer.param_groups[0]['lr']:.1e}"
        )
        update_string += f" loss: {total_loss:.1f}"
        model.binary_mode()
        binary_accuracy: float = get_accuracy(model, test_data, test_labels)
        model.scramble_mode()
        update_string += f" bin acc: {binary_accuracy:.3f}"
        print(update_string)
        if train_accuracy > target_accuracy:
            scramble_distance: float = max(
                min(5.0, model.scramble_distance + 0.01),
                starting_scramble_distance,
            )
            model.set_scramble_distance(scramble_distance)
