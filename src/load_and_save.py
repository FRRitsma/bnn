from pathlib import Path

import torch

from settings import settings
from src.model import SimpleNN


def save_model(model: SimpleNN, name: str) -> None:
    path: Path = settings.models_path / f"{name}.pth"
    torch.save(model.state_dict(), path)


def load_model(name: str) -> None:
    path: Path = settings.models_path / f"{name}.pth"
    model = SimpleNN(784, 10, 2.0)
    model.load_state_dict(torch.load(path))
