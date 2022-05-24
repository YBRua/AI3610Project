import torch.nn as nn

from .mlp import MLP
from .cnn import CNN


def build_model(name: str) -> nn.Module:
    if name == 'mlp':
        return MLP()
    elif name == 'cnn':
        return CNN()
    else:
        raise ValueError(f'Unknown model name: {name}')
