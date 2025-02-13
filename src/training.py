import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from settings import settings
from src.improved_model import BinarizingNetwork
from src.utils import device
from torchvision import transforms


def get_accuracy_one_hot_encoded(
    model: BinarizingNetwork, dataloader: DataLoader
) -> float:  # Vulture: ignore
    scramble_distance: float = model.scramble_distance
    model.scramble_distance = 0.0
    with torch.no_grad():  # Disable gradient computation
        all_correct: int = 0
        for inputs, labels in dataloader:
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            outputs: torch.Tensor = model(inputs)  # type: ignore
            comparison: torch.Tensor = torch.argmax(outputs, axis=1) == torch.argmax(
                labels, axis=1
            )
            all_correct += sum(comparison)
        accuracy: float = all_correct / len(dataloader.dataset)
    model.scramble_distance = scramble_distance
    return accuracy


def get_accuracy(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 512,
) -> float:
    total_correct = 0
    total_samples = labels.size(0)  # Get total number of samples

    with torch.no_grad():  # Disable gradient computation
        for i in range(0, total_samples, batch_size):
            batch_inputs = inputs[i : i + batch_size]  # Slice batch
            batch_labels = labels[i : i + batch_size]

            outputs = model(batch_inputs)  # Get model predictions
            predictions = torch.argmax(outputs, dim=1)  # Get predicted class indices

            total_correct += (
                (predictions == batch_labels).sum().item()
            )  # Count correct predictions

    accuracy = total_correct / total_samples  # Compute accuracy
    return accuracy


def get_quantile_of_separation(
    intermediate_output: list[float], quantile: float
) -> float:
    return np.quantile(abs(np.array(intermediate_output)), quantile)


def get_average_separation(test_data, model) -> float:
    intermediate_layer = model.layer1(test_data)
    all_quantiles = []
    for i in range(intermediate_layer.shape[1]):
        intermediate_output = intermediate_layer[:, i].tolist()
        all_quantiles.append(get_quantile_of_separation(intermediate_output, 0.05))
    return np.mean(all_quantiles)


class OneHotEncode:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, label):
        return torch.eye(self.num_classes)[label] * 2 - 1


num_classes: int = 10

# Define the transformation to apply to the images
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ]
)

if __name__ == "__main__":
    full_train_dataset = datasets.MNIST(
        root=settings.data_path,
        train=True,
        download=True,
        transform=transform,
        target_transform=OneHotEncode(num_classes),
    )
    train_size: int = int(0.8 * len(full_train_dataset))  # 80% for training
    val_size: int = len(full_train_dataset) - train_size  # 20% for validation
    test_dataset = datasets.MNIST(
        root=settings.data_path,
        train=False,
        download=True,
        transform=transform,
        target_transform=OneHotEncode(num_classes),
    )
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )
