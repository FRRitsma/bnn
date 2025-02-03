import torch
from torch import Tensor, nn

from src.improved_model import BinarizingLinear


def prune_model_layers(  # Vulture: ignore
    first_layer: BinarizingLinear,
    second_layer: BinarizingLinear,
    output_first_layer: Tensor,
) -> None:
    # Check if all values in each column are the same
    same_values_per_column = (output_first_layer == output_first_layer[0]).all(dim=0)
    # Get the indices of columns where all values are the same
    columns_to_remove = torch.nonzero(same_values_per_column, as_tuple=True)[0]
    columns_to_keep = [
        i for i in range(output_first_layer.size(1)) if i not in columns_to_remove
    ]
    new_bias = (
        output_first_layer[0, columns_to_remove]
        @ second_layer.weight[:, columns_to_remove].t()
    )
    first_layer.weight = nn.Parameter(first_layer.weight[columns_to_keep, :])
    first_layer.bias = nn.Parameter(first_layer.bias[columns_to_keep])
    second_layer.weight = nn.Parameter(second_layer.weight[:, columns_to_keep])
    second_layer.bias = nn.Parameter(new_bias + second_layer.bias)
