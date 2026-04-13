"""
phase1/models/simclr_model.py
Full SimCLR model = ResNet-18 encoder + 2-layer MLP projection head.
The projection head is used only during pre-training.
Downstream tasks use h (encoder output), NOT z (projection output).
"""

import torch.nn as nn
from phase1.models.encoder import ResNet18Encoder


class ProjectionHead(nn.Module):
    """
    2-layer MLP: 512 -> 512 (ReLU + BN) -> 128
    As described in SimCLR paper Section 3.
    """

    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, h):
        return self.net(h)


class SimCLR(nn.Module):
    """
    Combines encoder f(.) and projection head g(.).
    Forward pass returns both h and z for flexibility.
    """

    def __init__(self, projection_dim=128, hidden_dim=512):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.projector = ProjectionHead(
            input_dim=self.encoder.feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
        )

    def forward(self, x):
        """
        x  : (B, 3, 32, 32)
        h  : (B, 512)  -- used for downstream evaluation
        z  : (B, 128)  -- used for NT-Xent loss during pre-training
        """
        h = self.encoder(x)
        z = self.projector(h)
        return h, z
