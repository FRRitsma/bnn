import torch
from torch import nn as nn

from src.improved_model import BinarizingBase, binarizing_activation


class BinarizingLinear(nn.Linear, BinarizingBase):
    scale: nn.Parameter

    def __init__(
        self,
        in_features: int,
        out_features: int,
        binarize_parameters: bool,
        binarize_output: bool,
    ):
        nn.Linear.__init__(self, in_features, out_features)
        BinarizingBase.__init__(self, binarize_parameters, binarize_output)
        self.scale = nn.Parameter(torch.ones(1))

    def activation(self, x):
        x = x * self.scale
        if self.binarize_output:
            x = binarizing_activation(x, self.model_mode, self.scramble_distance)
        return x

    def forward(self, x):
        return torch.matmul(x, self.transformed_weight.t()) + self.transformed_bias


class BinarizingLinearBatchNorm(nn.Linear, BinarizingBase):
    batch_norm: nn.BatchNorm1d

    def __init__(
        self,
        in_features: int,
        out_features: int,
        binarize_parameters: bool,
        binarize_output: bool,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=False)
        BinarizingBase.__init__(self, binarize_parameters, binarize_output)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def eval(self):
        super().eval()
        self.batch_norm.weight.requires_grad = False
        self.batch_norm.bias.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.batch_norm.weight.requires_grad = False
        self.batch_norm.bias.requires_grad = False

    def activation(self, x):
        if self.binarize_output:
            x = binarizing_activation(x, self.model_mode, self.scramble_distance)
        return x

    def forward(self, x):
        x = torch.matmul(x, self.transformed_weight.t())
        x = self.batch_norm(x)
        return x
