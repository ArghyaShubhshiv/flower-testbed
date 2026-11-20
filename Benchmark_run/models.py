"""CIFAR-10 models.

``small`` (default): 3 conv + 2 FC, no normalisation layers, ~0.2M params.
  No BatchNorm on purpose: BN running stats travel in state_dict, get averaged,
  and give ALIE a second (non-parameter) channel -- a confound we don't want
  when the question is "what does the *hardware* see".
  One local epoch over ~8.3k images is ~5-15 s on a single Zen 2 core.

``medium``: the repo's VGG-ish CIFAR100CNN with GroupNorm and 10 classes,
  ~1.1M params, ~20x the FLOPs. Use if you want longer fit windows.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MediumCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                    nn.GroupNorm(min(32, cout), cout), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            *block(3, 64), *block(64, 64), nn.MaxPool2d(2),
            *block(64, 128), *block(128, 128), nn.MaxPool2d(2),
            *block(128, 256), *block(256, 256), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


def build_model(name: str = "small") -> nn.Module:
    return {"small": SmallCNN, "medium": MediumCNN}[name]()
