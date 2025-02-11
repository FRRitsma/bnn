import torch
from torch.utils.data import DataLoader

from settings import settings
from src.improved_model import BinarizingNetwork, ModelMode
from src.mnist_model import MNIST_CNN
from src.pruning import prune_model_layers
from src.training import train_dataset
from src.utils import device


def test_child_mode_transfer_model_mode():
    # Create a parent and a child network
    parent = BinarizingNetwork()
    child = BinarizingNetwork()
    # Assign the child to the parent
    parent.child = child  # type: ignore
    # Initially, both should be in train mode
    assert parent.model_mode == ModelMode.scramble
    assert child.model_mode == ModelMode.scramble
    # Switch parent to evaluation mode
    parent.clean_mode()
    # Check if both parent and child are in evaluation mode
    assert parent.model_mode == ModelMode.clean
    assert child.model_mode == ModelMode.clean
    # Switch parent back to train mode
    parent.scramble_mode()
    assert parent.model_mode == ModelMode.scramble
    assert child.model_mode == ModelMode.scramble


def test_child_transfer_scramble_distance():
    # Create a parent and a child network
    parent = BinarizingNetwork()
    child = BinarizingNetwork()
    # Assign the child to the parent
    parent.child = child  # type: ignore
    assert parent.scramble_distance == 0.0
    assert child.scramble_distance == 0.0
    parent.set_scramble_distance(1.0)
    assert parent.scramble_distance == 1.0
    assert child.scramble_distance == 1.0


def test_binarizing_keeps_output_similarity():
    model = MNIST_CNN(0.1)
    model.to(device)
    model.load_state_dict(torch.load(settings.models_path / "convnet_v2.pth"))
    model.clean_mode()
    train_dataloader = DataLoader(train_dataset, batch_size=int(1e3), shuffle=True)
    train_data, _ = next(iter(train_dataloader))
    train_data = train_data.to(device)
    output = model(train_data)
    model.binarize_weights()
    binarized_output = model(train_data)
    assert torch.all(output == binarized_output)


def test_pruning_network_keeps_output_similar():
    model = MNIST_CNN(0.1)
    model.to(device)
    model.load_state_dict(torch.load(settings.models_path / "convnet_v2.pth"))
    model.clean_mode()
    model.binarize_weights()
    train_dataloader = DataLoader(train_dataset, batch_size=int(5e4), shuffle=True)
    train_data, _ = next(iter(train_dataloader))
    train_data = train_data.to(device)
    y_before_pruning = model(train_data)
    columns_layer2_before_pruning = model.layer2.weight.shape[0]
    rows_layer3_before_pruning = model.layer3.weight.shape[1]
    x = model.second_layer(model.first_layer(train_data))
    prune_model_layers(model.layer2, model.layer3, x)
    assert model.layer2.weight.shape[0] < columns_layer2_before_pruning
    assert model.layer3.weight.shape[1] < rows_layer3_before_pruning
    y_after_pruning = model(train_data)
    assert torch.all(y_before_pruning == y_after_pruning)


# def test_init_scramble_distance_transfers_to_child_networks():
#     scramble_distance: float = 1.0
#     model = MNIST_CNN(scramble_distance)
#     assert model.layer2.scramble_distance == scramble_distance
