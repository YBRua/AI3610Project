import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(4 * 4 * 16, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.conv1(x))  # 28 -> 24
        x = self.pool1(x)  # 24 -> 12
        x = torch.relu(self.conv2(x))  # 12 -> 8
        x = self.pool2(x)  # 8 -> 4
        x = x.view(-1, 4 * 4 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
