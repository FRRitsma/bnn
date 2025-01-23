import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model import SimpleNN
from src.utils import device


def get_accuracy(model: SimpleNN, dataloader: DataLoader) -> float:
    model.eval_mode()
    with torch.no_grad():  # Disable gradient computation
        all_correct: int = 0
        for inputs, labels in dataloader:
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            outputs: torch.Tensor = model(inputs)
            comparison: torch.Tensor = torch.argmax(outputs, axis=1) == torch.argmax(
                labels, axis=1
            )
            all_correct += sum(comparison)
        accuracy: float = all_correct / len(dataloader.dataset)
    model.train_mode()
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
