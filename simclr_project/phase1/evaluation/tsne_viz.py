"""
phase1/evaluation/tsne_viz.py
t-SNE visualisation of SimCLR encoder representations on CIFAR-10.
Plots 1000 test samples coloured by class.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torchvision
from sklearn.manifold import TSNE

from phase1.data.augmentations import EvalTransform
from phase1.utils.checkpointing import load_encoder

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']


def plot_tsne_cifar(checkpoint_path, n_samples=1000, save_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = load_encoder(checkpoint_path, device)
    encoder.eval()

    transform = EvalTransform(32)
    test_set  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    subset = Subset(test_set, range(n_samples))
    loader = DataLoader(subset, batch_size=256, shuffle=False)

    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            feats.append(encoder(imgs.to(device)).cpu())
            labels.append(lbls)
    feats  = torch.cat(feats).numpy()
    labels = torch.cat(labels).numpy()

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=40)
    emb  = tsne.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for c in range(10):
        mask = labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=[colors[c]], label=CIFAR10_CLASSES[c],
                   s=12, alpha=0.7)
    ax.legend(markerscale=2, fontsize=9, loc='upper right')
    ax.set_title(
        f"SimCLR Embeddings — t-SNE\n"
        f"(trained with NO labels, 200 epochs)", fontsize=12
    )
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved -> {save_path}")
    plt.show()
