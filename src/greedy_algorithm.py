from typing import Optional

from src.mnist_model import MNIST_CNN
from src.training import get_accuracy
from src.utils import device
from settings import settings
import torch
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize_scalar
from mnist_training import test_data, test_labels, train_data


model = MNIST_CNN()
model.to(device)
model.load_state_dict(torch.load(settings.models_path / "mnist_model.pth"))

model.binary_mode()
print(get_accuracy(model, test_data, test_labels))

model.binary_mode()
layer1 = model.flatten(model.layer1.activation(model.layer1(train_data))).detach()
layer2 = model.layer3.activation(model.layer3(layer1)).detach()
layer3 = model.layer4.activation(model.layer4(layer2)).detach()


layer1 = layer1.cpu()
layer2 = layer2.cpu()
layer3 = layer3.cpu()


def get_accuracy_and_threshold(
    z_negative: np.ndarray,
    z_positive: np.ndarray,
    starting_threshold: Optional[int] = None,
) -> tuple[float, int]:
    def outer_accuracy(threshold):
        threshold = int(threshold)
        return inner_accuracy(threshold)

    @lru_cache(None)
    def inner_accuracy(threshold):
        correct_positive = sum(z_positive >= threshold)
        correct_negative = sum(z_negative < threshold)
        return -(correct_negative + correct_positive)

    if starting_threshold is None:
        min_t, max_t = (
            np.min([np.min(z_negative), np.min(z_positive)]),
            np.max([np.max(z_negative), np.max(z_positive)]),
        )
        bracket = (min_t - 2, max_t + 2)
    else:
        bracket = (starting_threshold - 10, starting_threshold + 10)

    result = minimize_scalar(outer_accuracy, bracket=bracket, method="golden")
    best_threshold = int(result.x)
    best_accuracy = -result.fun / (len(z_negative) + len(z_positive))

    return best_accuracy, best_threshold


def compute_best_matrix_for_row(
    x_negative: np.ndarray, x_positive: np.ndarray
) -> tuple[np.ndarray, int]:
    A = np.random.rand(x.shape[1]) > 0.5
    z_negative = np.sum(~(np.bitwise_xor(x_negative, A)), axis=1)
    z_positive = np.sum(~(np.bitwise_xor(x_positive, A)), axis=1)
    best_accuracy, best_threshold = get_accuracy_and_threshold(z_negative, z_positive)

    for epoch in range(5):
        print(f"Epoch {epoch}")
        best_before_epoch = best_accuracy
        for i in np.random.permutation(len(A)):
            # Subtract the old contribution
            z_negative -= x_negative[:, i] == A[i]
            z_positive -= x_positive[:, i] == A[i]

            # Flip the bit in A
            A[i] = ~A[i]

            # Add the new contribution
            z_negative += x_negative[:, i] == A[i]
            z_positive += x_positive[:, i] == A[i]

            # Compute accuracy and threshold
            accuracy, threshold = get_accuracy_and_threshold(
                z_negative, z_positive, best_threshold
            )

            # Update best accuracy and threshold
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
            else:
                # Revert the flip if accuracy doesn't improve
                z_negative -= x_negative[:, i] == A[i]
                z_positive -= x_positive[:, i] == A[i]
                A[i] = ~A[i]
                z_negative += x_negative[:, i] == A[i]
                z_positive += x_positive[:, i] == A[i]

        if best_accuracy == best_before_epoch:
            break
    return A, best_threshold


x = layer1.numpy() == 1
y = layer2.numpy() == 1


def process_row(index: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    print(f"Computing row {index}")
    y_row = y[:, index]
    x_negative = x[~y_row, :]
    x_positive = x[y_row, :]
    return compute_best_matrix_for_row(x_negative, x_positive)


if __name__ == "__main__":
    rows: list[np.ndarray] = []
    thresholds: list[int] = []

    for index in range(y.shape[1]):
        row, threshold = process_row(index, x, y)
        rows.append(row)
        thresholds.append(threshold)
