"""
phase2/evaluation/linear_probe_mnist.py
Metric 2: Linear probe accuracy on MNIST using frozen CIFAR-10 features.
Trains Linear(512->10) on 60,000 MNIST train features for 30 epochs.
Evaluates on 10,000 MNIST test features.
Encoder parameters are NOT updated.
Target: > 80% accuracy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from phase1.utils.checkpointing import load_encoder

NORMALIZE = T.Compose([
    T.Resize(32),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]),
])


def extract_mnist_features(encoder, device, batch_size=512):
    """Extract features from full MNIST train + test splits."""
    train_set = torchvision.datasets.MNIST(
        root='./data', train=True,  download=True, transform=NORMALIZE)
    test_set  = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=NORMALIZE)

    def _extract(dataset, desc):
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
        feats, labels = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc=desc):
                feats.append(encoder(imgs.to(device)).cpu())
                labels.append(lbls)
        return torch.cat(feats), torch.cat(labels)

    print("Extracting MNIST features...")
    train_h, train_y = _extract(train_set, "Train")
    test_h,  test_y  = _extract(test_set,  "Test")
    print(f"Train features: {train_h.shape}")
    print(f"Test  features: {test_h.shape}")
    return train_h, train_y, test_h, test_y


def run_linear_probe_mnist(checkpoint_path,
                            epochs=30, lr=0.01, batch_size=512,
                            target=0.80):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = load_encoder(checkpoint_path, device)

    print("\n" + "=" * 55)
    print("METRIC 2 — LINEAR PROBE ON MNIST")
    print("=" * 55)

    train_h, train_y, test_h, test_y = extract_mnist_features(
        encoder, device, batch_size)

    probe     = nn.Linear(512, 10).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds     = TensorDataset(train_h, train_y)
    probe_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    print(f"\nTraining Linear(512→10) probe...")
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        probe.train()
        total_loss = 0.0
        for h_b, y_b in probe_loader:
            logits = probe(h_b.to(device))
            loss   = criterion(logits, y_b.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            probe.eval()
            with torch.no_grad():
                preds = probe(test_h.to(device)).argmax(1).cpu()
            acc = (preds == test_y).float().mean().item()
            best_acc = max(best_acc, acc)
            avg_loss = total_loss / len(probe_loader)
            print(f"  Epoch {epoch:02d}/{epochs} | "
                  f"Loss {avg_loss:.4f} | Test Acc {acc:.4f}")

    probe.eval()
    with torch.no_grad():
        preds = probe(test_h.to(device)).argmax(1).cpu()
    final_acc = (preds == test_y).float().mean().item()

    print(f"\nLinear Probe Accuracy : {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"Random baseline       : 10.00%")
    print(f"Target                : > {target*100:.0f}%")
    print(f"Result: {'PASS' if final_acc > target else 'FAIL'}")

    return final_acc
