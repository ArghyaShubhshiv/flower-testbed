"""
Model definition shared by the server and every client.
All parties MUST use exactly this architecture so that state_dicts
(and therefore weight deltas) line up shape-for-shape.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR100CNN(nn.Module):
    """Small CNN for CIFAR-100 (100 classes, 32x32x3 inputs)."""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # -> 32 x 16 x 16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # -> 64 x 8 x 8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # -> 128 x 4 x 4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def trainable_mask(net):
    """One bool per state_dict entry: True for trainable params, False for buffers (BN stats)."""
    params = {n for n, _ in net.named_parameters()}
    return [k in params for k in net.state_dict().keys()]


def get_parameters(net):
    """state_dict -> list of numpy arrays (the format Flower ships around)."""
    return [val.cpu().numpy() for val in net.state_dict().values()]


def set_parameters(net, parameters):
    """list of numpy arrays -> load into `net` (shapes must match exactly)."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
