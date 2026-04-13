"""
phase1/utils/metrics.py
Simple accuracy and loss tracking utilities for Phase 1.
"""

import torch


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy given raw logits and integer labels."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


class AverageMeter:
    """Tracks a running average of a scalar (loss, accuracy, etc.)."""

    def __init__(self, name=''):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.sum   = 0.0
        self.count = 0
        self.avg   = 0.0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f} (n={self.count})"
