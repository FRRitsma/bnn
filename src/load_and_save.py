from pathlib import Path

import torch
from torch import nn

from settings import settings
from src.improved_model import BinarizingBase


def save_model(model: nn.Module, name: str) -> None:
    path: Path = settings.models_path / f"{name}.pth"
    torch.save(model.state_dict(), path)


def load_model(name: str) -> None:
    path: Path = settings.models_path / f"{name}.pth"
    model = BinarizingBase(True, True)
    model.load_state_dict(torch.load(path))  # type: ignore
