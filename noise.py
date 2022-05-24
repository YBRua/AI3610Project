import torch
import torch.nn as nn


def add_noise_to_weights(
        model: nn.Module,
        device: torch.device,
        mean: float = 0.0,
        std: float = 0.1,):
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * 0.1)
    """
    gassian_kernel = torch.distributions.Normal(mean, std)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(
                torch.exp(gassian_kernel.sample(param.size())).to(device))
