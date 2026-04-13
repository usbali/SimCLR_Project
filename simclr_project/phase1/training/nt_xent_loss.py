"""
phase1/training/nt_xent_loss.py
NT-Xent (Normalised Temperature-scaled Cross-Entropy) loss.
Reference: Chen et al. 2020, Equation 1.

For a batch of N images producing 2N augmented views:
  - Positive pair: (z_i, z_j) = two views of the same image
  - Negatives    : all other 2(N-1) views in the batch
Loss is averaged over all 2N positive pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j : (N, D) — L2-normalised projection vectors
                            for the two views of the same batch.
        Returns   : scalar loss averaged over 2N pairs.
        """
        N = z_i.size(0)
        device = z_i.device

        # L2 normalise
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate: (2N, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Pairwise cosine similarity matrix: (2N, 2N)
        sim = torch.matmul(z, z.T) / self.temperature

        # Mask out self-similarities on the diagonal
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float('-inf'))

        # Positive pair indices:
        #   for i in [0..N-1]  : positive is i+N
        #   for i in [N..2N-1] : positive is i-N
        pos_idx = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device),
        ])

        loss = F.cross_entropy(sim, pos_idx)
        return loss
