import torch
import torch.nn as nn
import torch.autograd as autograd


class Quantization(autograd.Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return torch.where(
            x.abs() <= 1,
            grad_output,
            torch.zeros_like(grad_output))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        # self.drop1 = nn.Dropout(0.5)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        # self.drop2 = nn.Dropout(0.5)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x = x.reshape(B, -1)
        x = self.fc1(x)
        # x = self.drop1(x)
        # x = Quantization.apply(x)
        x = self.act1(x)

        x = self.fc2(x)
        # x = self.drop2(x)
        # x = Quantization.apply(x)
        x = self.act2(x)

        x = self.out(x)
        # x = Quantization.apply(x)
        return x
