"""
phase1/training/lr_scheduler.py
Learning rate schedule: linear warmup then cosine annealing to zero.
Warmup: epochs 1..warmup_epochs  (lr_start -> lr_peak)
Cosine: epochs warmup_epochs+1..total_epochs  (lr_peak -> ~0)
"""

import math


def get_lr(epoch, total_epochs, lr_peak, lr_start=2e-4, warmup_epochs=9):
    """
    Returns the learning rate for the given epoch (1-indexed).
    epoch          : current epoch (1 to total_epochs)
    total_epochs   : total training epochs
    lr_peak        : peak learning rate reached after warmup
    lr_start       : initial lr at epoch 1
    warmup_epochs  : number of linear warmup epochs
    """
    if epoch <= warmup_epochs:
        # Linear warmup
        return lr_start + (lr_peak - lr_start) * (epoch - 1) / (warmup_epochs - 1)
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return lr_peak * 0.5 * (1 + math.cos(math.pi * progress))


def update_lr(optimizer, epoch, total_epochs, lr_peak,
              lr_start=2e-4, warmup_epochs=9):
    """Update all param groups in optimizer to the scheduled LR."""
    lr = get_lr(epoch, total_epochs, lr_peak, lr_start, warmup_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
