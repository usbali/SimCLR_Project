"""
phase2/evaluation/tsne_viz_mnist.py
t-SNE visualisation of SimCLR features on Logged MNIST.
Left panel  : digits at 0° only (coloured by class)
Right panel : all 7 rotation angles (tests rotation invariance visually)
Tight clusters = encoder is rotation-invariant.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


DIGIT_COLORS = [
    '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
    '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
]


def plot_tsne_mnist(feats, labels, rots,
                   n_per_class=100, save_path=None,
                   invariance_score=None, probe_acc=None):
    """
    feats  : (N, 512) tensor
    labels : (N,) tensor
    rots   : (N,) tensor
    """
    # --- Panel 1: 0° only, n_per_class samples per digit ---
    mask_0 = rots == 0
    idx_0  = []
    for c in range(10):
        class_idx = ((labels == c) & mask_0).nonzero(as_tuple=True)[0]
        idx_0.extend(class_idx[:n_per_class].tolist())
    idx_0  = torch.tensor(idx_0)
    feats_0  = feats[idx_0].numpy()
    labels_0 = labels[idx_0].numpy()

    # --- Panel 2: all rotations, n_per_class per (class, angle) ---
    idx_all = []
    for c in range(10):
        for rot in [-45, -30, -15, 0, 15, 30, 45]:
            class_rot_idx = ((labels == c) & (rots == rot)).nonzero(
                as_tuple=True)[0]
            idx_all.extend(class_rot_idx[:n_per_class].tolist())
    idx_all  = torch.tensor(idx_all)
    feats_all  = feats[idx_all].numpy()
    labels_all = labels[idx_all].numpy()

    print("Running t-SNE (panel 1)...")
    emb_0   = TSNE(n_components=2, random_state=42, perplexity=40
                   ).fit_transform(feats_0)
    print("Running t-SNE (panel 2)...")
    emb_all = TSNE(n_components=2, random_state=42, perplexity=50
                   ).fit_transform(feats_all)

    title = "SimCLR Features on Logged MNIST"
    if invariance_score is not None:
        title += f"  |  Invariance: {invariance_score:.3f}"
    if probe_acc is not None:
        title += f"  |  Probe: {probe_acc*100:.1f}%"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(title, fontsize=13)

    for ax, emb, labs, subtitle in [
        (axes[0], emb_0,   labels_0,   "t-SNE: Digits at 0° only\n(colored by digit class)"),
        (axes[1], emb_all, labels_all, "t-SNE: All rotations (−45° to +45°)\nTight clusters = rotation invariant"),
    ]:
        for c in range(10):
            mask = labs == c
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=DIGIT_COLORS[c], label=str(c),
                       s=8, alpha=0.6)
        ax.legend(title='Digit', markerscale=2, fontsize=8,
                  loc='upper right')
        ax.set_title(subtitle, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved -> {save_path}")
    plt.show()
