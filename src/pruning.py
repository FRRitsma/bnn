import numpy as np
import torch
from torch.utils.data import DataLoader

from src.improved_model import CNN_2
from src.utils import device

# tolerance: float = float(1e-4)
# percentage: float = float(0.999)


def get_intermediate_outputs_as_numpy(
    model: CNN_2, data_loader: DataLoader
) -> np.ndarray:
    model.eval_mode()
    all_outputs: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model.layer1(inputs)
            all_outputs.append(outputs.cpu().numpy())
    model.train_mode()
    return np.concatenate(all_outputs, axis=0)


def get_two_dimensional_histogram(data: np.ndarray) -> np.ndarray:
    bin_edges: np.ndarray = np.linspace(0, 1, 50)
    all_histograms: np.ndarray = np.concatenate(
        [np.histogram(row, bin_edges)[0].reshape([1, -1]) for row in data.T], axis=0
    )
    return all_histograms


separation_threshold: float = 0.95
separation_proportion: float = 0.95


def which_layers_are_separated(data: np.ndarray) -> np.ndarray:
    data = data.copy()
    data = abs(data - 0.5)
    proportion = np.mean(data > (separation_threshold - 0.5), axis=0)
    separated = proportion > separation_proportion
    return separated


def which_layers_dead_at_zero(data: np.ndarray) -> np.ndarray:
    data = data.copy()
    proportion = np.mean(data < (1 - separation_threshold), axis=0)
    dead = proportion > separation_proportion
    return dead


def which_layers_dead_at_one(data: np.ndarray) -> np.ndarray:
    data = data.copy()
    proportion = np.mean(data > separation_threshold, axis=0)
    dead = proportion > separation_proportion
    return dead


def is_any_layer_dead(data: np.ndarray) -> bool:
    return any(which_layers_dead_at_zero(data)) or any(which_layers_dead_at_one(data))


def is_all_layers_separated(data: np.ndarray) -> bool:
    return all(which_layers_are_separated(data))
