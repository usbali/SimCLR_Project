"""
phase2/evaluation/invariance_score.py
Computes the rotation invariance score (Metric 1).

For each digit class c:
    IS(c) = cosine_sim( mean_feat(c, 0°), mean_feat(c, 45°) )

Overall IS = mean of IS(c) across all 10 classes.
Also computes the full rotation sweep (0° vs each angle).
Target: overall IS > 0.80
"""

import torch
import torch.nn.functional as F
import numpy as np


ROTATIONS = [-45, -30, -15, 0, 15, 30, 45]


def cosine_sim(a, b):
    """Cosine similarity between two 1-D vectors."""
    return F.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()


def compute_invariance_score(feats, labels, rots,
                              ref_angle=0, cmp_angle=45,
                              target=0.80):
    """
    feats  : (N, 512) tensor
    labels : (N,)     tensor — digit class 0..9
    rots   : (N,)     tensor — rotation angle in degrees
    """
    print("=" * 55)
    print("METRIC 1 — ROTATION INVARIANCE SCORE")
    print("=" * 55)

    per_class_scores = {}
    for c in range(10):
        mask_0   = (labels == c) & (rots == ref_angle)
        mask_45  = (labels == c) & (rots == cmp_angle)
        mean_0   = feats[mask_0].mean(dim=0)
        mean_45  = feats[mask_45].mean(dim=0)
        score    = cosine_sim(mean_0, mean_45)
        per_class_scores[c] = score

    overall_mean = np.mean(list(per_class_scores.values()))
    overall_std  = np.std(list(per_class_scores.values()))

    print(f"\nOverall Invariance Score ({ref_angle}° vs {cmp_angle}°)")
    print(f"  Mean : {overall_mean:.4f}")
    print(f"  Std  : {overall_std:.4f}")
    print(f"  Target: > {target}")
    print(f"  Result: {'PASS' if overall_mean > target else 'FAIL'}")

    print(f"\nPer-class invariance score ({ref_angle}° vs {cmp_angle}°):")
    for c, s in per_class_scores.items():
        bar = '█' * int(s * 22)
        chk = '✓' if s > target else '✗'
        print(f"  Digit {c}: {s:.4f}  {bar} {chk}")

    # Rotation sweep
    print(f"\nRotation sweep — cosine similarity with {ref_angle}° reference:")
    sweep = {}
    for angle in ROTATIONS:
        sims = []
        for c in range(10):
            m0 = feats[(labels == c) & (rots == ref_angle)].mean(dim=0)
            ma = feats[(labels == c) & (rots == angle)].mean(dim=0)
            sims.append(cosine_sim(m0, ma))
        sweep[angle] = np.mean(sims)
        chk = '✓' if sweep[angle] > target else '✗'
        label = f"{ref_angle}° vs {angle:+d}°" if angle != ref_angle \
                else f"  {ref_angle}° vs   {angle}°"
        print(f"  {label}: {sweep[angle]:.4f}  {chk}")

    return {
        'per_class': per_class_scores,
        'overall_mean': overall_mean,
        'overall_std':  overall_std,
        'sweep':        sweep,
        'pass':         overall_mean > target,
    }
