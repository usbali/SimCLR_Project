"""
phase2/evaluation/tsne_viz_colour.py
t-SNE visualisation of SimCLR features on Logged Colour MNIST.

Left panel  : digits in red only (one colour, coloured by digit class)
Right panel : all 7 colours overlaid (tests colour invariance visually)

Tight clusters in the right panel = encoder is colour-invariant.
If the encoder were colour-sensitive, each colour would form a sub-cluster
splitting the digit cluster into 7 colour-coded blobs.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from phase2.evaluation.colour_invariance import COLOUR_NAMES

DIGIT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def plot_tsne_colour(feats, labels, cols,
                     n_per_class=100, save_path=None,
                     colour_inv_score=None):
    """
    feats  : (N, 512) tensor
    labels : (N,)     tensor — digit class 0..9
    cols   : (N,)     tensor — colour index
    """
    ref_idx = COLOUR_NAMES.index('red')

    # ── Panel 1: red only ────────────────────────────────────────────────────
    idx_red = []
    for c in range(10):
        mask = (labels == c) & (cols == ref_idx)
        idx_red.extend(
            mask.nonzero(as_tuple=True)[0][:n_per_class].tolist())
    idx_red  = torch.tensor(idx_red)
    f_red    = feats[idx_red].numpy()
    l_red    = labels[idx_red].numpy()

    # ── Panel 2: all colours ─────────────────────────────────────────────────
    idx_all = []
    for c in range(10):
        for ci in range(len(COLOUR_NAMES)):
            mask = (labels == c) & (cols == ci)
            idx_all.extend(
                mask.nonzero(as_tuple=True)[0][:30].tolist())
    idx_all  = torch.tensor(idx_all)
    f_all    = feats[idx_all].numpy()
    l_all    = labels[idx_all].numpy()

    print("Running t-SNE (panel 1 — red only)...")
    emb_red = TSNE(n_components=2, random_state=42,
                   perplexity=40).fit_transform(f_red)

    print("Running t-SNE (panel 2 — all colours)...")
    emb_all = TSNE(n_components=2, random_state=42,
                   perplexity=50).fit_transform(f_all)

    title = "SimCLR Features on Logged Colour MNIST"
    if colour_inv_score is not None:
        title += f"  |  Colour invariance: {colour_inv_score:.4f}"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(title, fontsize=12)

    for ax, emb, labs, subtitle in [
        (axes[0], emb_red, l_red,
         't-SNE: Digits in red only\n(coloured by digit class)'),
        (axes[1], emb_all, l_all,
         't-SNE: All 7 colours overlaid\nTight clusters = colour invariant'),
    ]:
        for c in range(10):
            mask = labs == c
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=DIGIT_COLORS[c], label=str(c),
                       s=10, alpha=0.65)
        ax.legend(title='Digit', markerscale=2,
                  fontsize=8, loc='upper right')
        ax.set_title(subtitle, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()
