import torch.nn as nn

from .mlp import MLP, DropoutMLP, QuantizedMLP
from .cnn import CNN


def build_model(name: str) -> nn.Module:
    if name == 'mlp':
        return MLP()
    elif name == 'cnn':
        return CNN()
    elif name == 'dmlp':
        return DropoutMLP()
    elif name == 'qmlp':
        return QuantizedMLP()
    else:
        raise ValueError(f'Unknown model name: {name}')
