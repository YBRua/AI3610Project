import torch.nn as nn

from .mlp import MLP, DropoutMLP, MultiMLP, QuantizedMLP
from .cnn import CNN, XzrCNN


def build_model(name: str) -> nn.Module:
    if name == 'mlp':
        return MLP()
    elif name == 'cnn':
        return CNN()
    elif name == 'dmlp':
        return DropoutMLP()
    elif name == 'qmlp':
        return QuantizedMLP()
    elif name == 'xzr':
        return XzrCNN()
    elif name == 'mmlp':
        return MultiMLP()
    else:
        raise ValueError(f'Unknown model name: {name}')
