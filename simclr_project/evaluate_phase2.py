"""
evaluate_phase2.py
Entry point for Phase 2 — Cross-domain evaluation on Logged MNIST
plus the colour invariance extension (professor's experiment).

Usage:
    python evaluate_phase2.py
    python evaluate_phase2.py --config configs/phase2_config.yaml \
                               --checkpoint simclr_checkpoints/checkpoint_epoch_200.pth
    python evaluate_phase2.py --skip-build       # reuse existing rotation images
    python evaluate_phase2.py --skip-colour      # skip colour invariance
    python evaluate_phase2.py --colour-only      # run colour experiment only

Steps:
    1.  Build Logged MNIST (rotation experiment)
    2.  Extract rotation features  [70000, 512]
    3.  METRIC 1 — Rotation invariance score
    4.  METRIC 2 — MNIST linear probe accuracy
    5.  t-SNE visualisation (rotation)
    6.  Build Logged Colour MNIST & extract features
    7.  METRIC 3 — Colour invariance score
    8.  Colour invariance plots
    9.  t-SNE visualisation (colour)
    10. Save all results to JSON
"""

import argparse
import yaml
import os
import torch

from phase2.dataset.logged_mnist import build_logged_mnist
from phase2.evaluation.feature_extractor import extract_all_features
from phase2.evaluation.invariance_score import compute_invariance_score
from phase2.evaluation.linear_probe_mnist import run_linear_probe_mnist
from phase2.evaluation.tsne_viz_mnist import plot_tsne_mnist
from phase2.evaluation.colour_invariance import (
    extract_colour_features, compute_colour_invariance
)
from phase2.evaluation.colour_invariance_plots import plot_colour_invariance
from phase2.evaluation.tsne_viz_colour import plot_tsne_colour
from phase2.utils.results_io import save_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='SimCLR Phase 2 Evaluation')
    parser.add_argument('--config', type=str,
                        default='configs/phase2_config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--skip-build',  action='store_true')
    parser.add_argument('--skip-colour', action='store_true')
    parser.add_argument('--colour-only', action='store_true')
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args   = parse_args()
    config = load_config(args.config)

    checkpoint = args.checkpoint or config['encoder']['checkpoint']
    out_dir    = config['results']['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("SimCLR Phase 2 — Cross-Domain Evaluation")
    print("=" * 60)
    print(f"Checkpoint : {checkpoint}")
    print(f"Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")

    results     = {}
    inv_results = None
    probe_acc   = None
    col_results = None

    # ── ROTATION EXPERIMENT ───────────────────────────────────────────────────
    if not args.colour_only:
        csv_path = config['logged_mnist']['csv_path']
        if not args.skip_build:
            print("\nStep 1: Building Logged MNIST...")
            csv_path = build_logged_mnist(
                output_dir=config['logged_mnist']['output_dir'],
                out_size=config['logged_mnist']['image_size'],
            )
        else:
            print(f"\nStep 1: Reusing {csv_path}")

        print("\nStep 2: Extracting rotation features...")
        feats, labels, rots = extract_all_features(
            checkpoint_path=checkpoint,
            csv_path=csv_path,
            batch_size=config['evaluation']['batch_size'],
            num_workers=config['evaluation']['num_workers'],
        )

        print()
        inv_results = compute_invariance_score(
            feats, labels, rots,
            ref_angle=config['invariance']['reference_angle'],
            cmp_angle=config['invariance']['comparison_angle'],
            target=config['invariance']['target_score'],
        )

        print()
        probe_acc = run_linear_probe_mnist(
            checkpoint_path=checkpoint,
            epochs=config['linear_probe']['epochs'],
            lr=config['linear_probe']['learning_rate'],
            batch_size=config['evaluation']['batch_size'],
        )

        print("\nGenerating t-SNE (rotation)...")
        plot_tsne_mnist(
            feats, labels, rots, n_per_class=100,
            save_path=os.path.join(out_dir, 'tsne_logged_mnist.png'),
            invariance_score=inv_results['overall_mean'],
            probe_acc=probe_acc,
        )

        results['metric1_rotation_invariance'] = {
            'overall_mean': inv_results['overall_mean'],
            'overall_std' : inv_results['overall_std'],
            'per_class'   : inv_results['per_class'],
            'sweep'       : inv_results['sweep'],
            'pass'        : inv_results['pass'],
        }
        results['metric2_linear_probe'] = {
            'accuracy': probe_acc,
            'pass'    : probe_acc > 0.80,
        }

    # ── COLOUR INVARIANCE EXPERIMENT ──────────────────────────────────────────
    if not args.skip_colour:
        cfg_col  = config.get('logged_colour_mnist', {})
        save_dir = cfg_col.get('save_dir') if cfg_col.get('save_images') \
                   else None

        print("\n" + "=" * 60)
        print("METRIC 3 — COLOUR INVARIANCE EXPERIMENT")
        print("=" * 60)

        print("\nBuilding Logged Colour MNIST & extracting features...")
        c_feats, c_labels, c_cols = extract_colour_features(
            checkpoint_path=checkpoint,
            batch_size=config['evaluation']['batch_size'],
            num_workers=config['evaluation']['num_workers'],
            save_dir=save_dir,
        )

        rotation_is = inv_results['overall_mean'] if inv_results else None
        print()
        col_results = compute_colour_invariance(
            c_feats, c_labels, c_cols,
            ref_colour=cfg_col.get('reference_colour', 'red'),
            target=cfg_col.get('target_score', 0.80),
            rotation_is=rotation_is,
        )

        print("\nGenerating colour invariance plots...")
        plot_colour_invariance(
            per_class_scores=col_results['per_class'],
            colour_sweep=col_results['sweep'],
            colour_inv_overall=col_results['overall_mean'],
            rotation_is=rotation_is,
            save_path=os.path.join(out_dir, 'colour_invariance_analysis.png'),
        )

        print("\nGenerating t-SNE (colour)...")
        plot_tsne_colour(
            c_feats, c_labels, c_cols, n_per_class=100,
            save_path=os.path.join(out_dir, 'tsne_logged_colour_mnist.png'),
            colour_inv_score=col_results['overall_mean'],
        )

        results['metric3_colour_invariance'] = {
            'overall_mean': col_results['overall_mean'],
            'overall_std' : col_results['overall_std'],
            'per_class'   : col_results['per_class'],
            'sweep'       : col_results['sweep'],
            'pass'        : col_results['pass'],
        }

    # ── Save + summary ────────────────────────────────────────────────────────
    results['context'] = {
        'checkpoint'             : checkpoint,
        'encoder_trained_on'     : 'CIFAR-10 (no labels)',
        'evaluated_on'           : 'Logged MNIST + Logged Colour MNIST',
        'encoder_params_updated' : 0,
    }
    save_results(results, config['results']['json_file'])

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE — RESULTS SUMMARY")
    print("=" * 60)
    if inv_results:
        print(f"  METRIC 1 — Rotation IS : {inv_results['overall_mean']:.4f}  "
              f"({'PASS' if inv_results['pass'] else 'FAIL'})")
    if probe_acc is not None:
        print(f"  METRIC 2 — MNIST probe  : {probe_acc*100:.2f}%  "
              f"({'PASS' if probe_acc > 0.80 else 'FAIL'})")
    if col_results:
        print(f"  METRIC 3 — Colour CI    : {col_results['overall_mean']:.4f}  "
              f"({'PASS' if col_results['pass'] else 'FAIL'})")
        if inv_results:
            diff = col_results['overall_mean'] - inv_results['overall_mean']
            print(f"  CI - IS (trained vs emergent) : +{diff:.4f}  ✓")
    print(f"\n  Results → {config['results']['json_file']}")


if __name__ == '__main__':
    main()
