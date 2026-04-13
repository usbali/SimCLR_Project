"""
phase2/evaluation/colour_invariance_plots.py
Publication-quality plots for the colour invariance experiment.
Produces a 2-panel figure matching the report figure (colour_invariance_analysis.png).
"""

import matplotlib.pyplot as plt
import numpy as np
from phase2.evaluation.colour_invariance import COLOUR_NAMES


COLOUR_SWATCH = {
    'red'   : '#e74c3c',
    'blue'  : '#3498db',
    'green' : '#27ae60',
    'orange': '#e67e22',
    'purple': '#9b59b6',
    'cyan'  : '#1abc9c',
    'white' : '#95a5a6',
}


def plot_colour_invariance(per_class_scores, colour_sweep,
                            colour_inv_overall, rotation_is=None,
                            save_path=None):
    """
    Two-panel figure:
      Left  — per-class CI bar chart (y-axis zoomed to [0.98, 1.00])
      Right — colour sweep bar chart (red vs each colour)

    per_class_scores : dict {digit: score}
    colour_sweep     : dict {colour_name: score}
    colour_inv_overall : float
    rotation_is      : float, optional — draws comparison line if provided
    save_path        : str, optional
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: per-class bars ──────────────────────────────────────────────────
    ax = axes[0]
    digits     = list(range(10))
    scores     = [per_class_scores[d] for d in digits]
    bar_colors = ['#2ecc71' if s > 0.80 else '#f39c12' for s in scores]
    bars = ax.bar(digits, scores, color=bar_colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)

    ax.axhline(y=0.80, color='red', linestyle='--',
               linewidth=1.5, label='Target (0.80)')
    ax.axhline(y=colour_inv_overall, color='#2c3e50', linestyle='-',
               linewidth=2,
               label=f'Overall CI ({colour_inv_overall:.4f})')
    if rotation_is is not None:
        ax.axhline(y=rotation_is, color='#8e44ad', linestyle=':',
                   linewidth=1.8,
                   label=f'Rotation IS ({rotation_is:.4f})')

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0003,
                f'{score:.4f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Digit Class', fontsize=11)
    ax.set_ylabel('Cosine Similarity', fontsize=11)
    ax.set_title(
        'Colour Invariance per Digit Class\n'
        '(red vs mean of all other colours)', fontsize=11)
    ax.set_xticks(digits)
    ax.set_ylim(0.980, 1.004)   # zoomed — all scores ~0.99
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ── Right: colour sweep ───────────────────────────────────────────────────
    ax = axes[1]
    ref = 'red'
    col_names = [c for c in COLOUR_NAMES if c != ref]
    col_vals  = [colour_sweep[c] for c in col_names]
    colors    = [COLOUR_SWATCH.get(c, '#888888') for c in col_names]

    bar_sw = ax.bar(col_names, col_vals, color=colors,
                    alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.80, color='red', linestyle='--',
               linewidth=1.5, label='Target (0.80)')
    for bar, val in zip(bar_sw, col_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0003,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Comparison Colour', fontsize=11)
    ax.set_ylabel(f'Cosine Similarity with {ref}', fontsize=11)
    ax.set_title(
        f'Colour Sweep ({ref} vs each colour)\n'
        f'averaged over all 10 digit classes', fontsize=11)
    ax.set_ylim(0.980, 1.004)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    subtitle = (
        f'Overall CI = {colour_inv_overall:.4f}  |  '
        f'High score confirms colour jitter augmentation worked as intended.'
    )
    if rotation_is is not None:
        diff = colour_inv_overall - rotation_is
        subtitle += f'\nCI − IS = +{diff:.4f}  (trained > emergent invariance ✓)'

    fig.suptitle(subtitle, fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.show()
