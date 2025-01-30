from pathlib import Path

import torch

from settings import settings
from src.improved_model import SimpleCNN


def save_model(model: SimpleCNN, name: str) -> None:
    path: Path = settings.models_path / f"{name}.pth"
    torch.save(model.state_dict(), path)


def load_model(name: str) -> None:
    path: Path = settings.models_path / f"{name}.pth"
    model = SimpleCNN(0.0)
    model.load_state_dict(torch.load(path))
