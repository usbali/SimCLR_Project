"""
phase1/models/encoder.py
ResNet-18 backbone adapted for CIFAR-10 (32x32 images).
Replaces the 7x7 stem conv with 3x3 and removes the first max-pool,
following the SimCLR CIFAR adaptation in Chen et al. 2020.
Output: 512-D average-pooled feature vector h.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 with CIFAR-10 modifications:
      - 7x7 conv (stride 2) -> 3x3 conv (stride 1)
      - MaxPool removed
      - Final FC layer removed
    Output dimension: 512
    """

    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)

        # CIFAR adaptation: smaller stem
        base.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        base.bn1   = nn.BatchNorm2d(64)
        base.relu  = nn.ReLU(inplace=True)
        base.maxpool = nn.Identity()   # remove max-pool

        # Keep all residual blocks, drop the classification head
        self.encoder = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )
        self.feature_dim = 512

    def forward(self, x):
        """x: (B, 3, 32, 32) -> h: (B, 512)"""
        h = self.encoder(x)
        return torch.flatten(h, start_dim=1)
