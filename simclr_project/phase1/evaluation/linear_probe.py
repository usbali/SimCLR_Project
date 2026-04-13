"""
phase1/evaluation/linear_probe.py
Linear probe evaluation on CIFAR-10.
Loads a saved encoder checkpoint, freezes it, trains a single
Linear(512->10) layer on CIFAR-10 train features, reports test accuracy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from tqdm import tqdm

from phase1.models.simclr_model import SimCLR
from phase1.data.augmentations import EvalTransform
from phase1.utils.checkpointing import load_encoder


def extract_features(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device)
            h = encoder(imgs)
            feats.append(h.cpu())
            labels.append(lbls)
    return torch.cat(feats), torch.cat(labels)


def run_linear_probe(checkpoint_path, epochs=100, lr=0.01, batch_size=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load frozen encoder
    encoder = load_encoder(checkpoint_path, device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    transform = EvalTransform(image_size=32)

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_set  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=4)

    print("Extracting CIFAR-10 features...")
    train_h, train_y = extract_features(encoder, train_loader, device)
    test_h,  test_y  = extract_features(encoder, test_loader,  device)
    print(f"Train: {train_h.shape} | Test: {test_h.shape}")

    # Train linear probe
    probe = nn.Linear(512, 10).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(train_h, train_y)
    probe_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        probe.train()
        for h_batch, y_batch in probe_loader:
            logits = probe(h_batch.to(device))
            loss   = criterion(logits, y_batch.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            probe.eval()
            with torch.no_grad():
                preds = probe(test_h.to(device)).argmax(1).cpu()
            acc = (preds == test_y).float().mean().item()
            print(f"  Epoch {epoch:3d}/{epochs} | Test Acc {acc:.4f} ({acc*100:.2f}%)")

    probe.eval()
    with torch.no_grad():
        preds = probe(test_h.to(device)).argmax(1).cpu()
    final_acc = (preds == test_y).float().mean().item()
    print(f"\nFinal Linear Probe Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    return final_acc
