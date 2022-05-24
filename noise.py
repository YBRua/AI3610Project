import torch
import torch.nn as nn
import logging


def add_noise_to_weights(
        model: nn.Module,
        device: torch.device,
        mean: float = 0.0,
        std: float = 0.1):
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * 0.1)
    """
    if std < 1e-5:
        return
    gassian_kernel = torch.distributions.Normal(mean, std)
    with torch.no_grad():
        for name, param in model.named_parameters():
            noises = torch.exp(gassian_kernel.sample(param.size())).to(device)
            param.mul_(noises)
