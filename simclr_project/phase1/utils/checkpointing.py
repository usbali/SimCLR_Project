"""
phase1/utils/checkpointing.py
Save and load SimCLR checkpoints.
load_encoder() returns only the frozen encoder (no projection head)
ready for downstream evaluation.
"""

import torch
from phase1.models.simclr_model import SimCLR
from phase1.models.encoder import ResNet18Encoder


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':                 loss,
    }, path)


def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device)


def load_encoder(checkpoint_path, device='cpu'):
    """
    Loads the full SimCLR checkpoint and returns only the
    ResNet-18 encoder (projection head discarded).
    Encoder is returned in eval mode with gradients disabled.
    """
    ckpt  = load_checkpoint(checkpoint_path, device)
    model = SimCLR()
    model.load_state_dict(ckpt['model_state_dict'])

    encoder = model.encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    missing = [k for k in ckpt['model_state_dict']
               if k.startswith('projector')]
    print(f"Encoder loaded and frozen")
    print(f"  Missing keys (projection head — expected): {len(missing)}")
    print(f"  Trainable params: "
          f"{sum(p.numel() for p in encoder.parameters() if p.requires_grad)}"
          f"  <- must be 0")
    return encoder
