"""
train_phase1.py
Entry point for Phase 1 — SimCLR pre-training on CIFAR-10.

Usage:
    python train_phase1.py
    python train_phase1.py --config configs/phase1_config.yaml
    python train_phase1.py --config configs/phase1_config.yaml --resume simclr_checkpoints/checkpoint_epoch_50.pth
"""

import argparse
import yaml
import os
import torch

from phase1.training.trainer import SimCLRTrainer
from phase1.evaluation.linear_probe import run_linear_probe
from phase1.evaluation.tsne_viz import plot_tsne_cifar


def parse_args():
    parser = argparse.ArgumentParser(description='SimCLR Phase 1 Training')
    parser.add_argument('--config', type=str,
                        default='configs/phase1_config.yaml',
                        help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training, run evaluation only')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to evaluate (used with --eval-only)')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args   = parse_args()
    config = load_config(args.config)

    print("=" * 60)
    print("SimCLR Phase 1 — Self-Supervised Pre-Training on CIFAR-10")
    print("=" * 60)
    print(f"Config : {args.config}")
    print(f"Device : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    if not args.eval_only:
        # ── Training ──────────────────────────────────────────────
        trainer = SimCLRTrainer(config)

        if args.resume:
            print(f"Resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location='cpu')
            trainer.model.load_state_dict(ckpt['model_state_dict'])
            trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

        trainer.train()
        best_ckpt = os.path.join(
            config['checkpointing']['checkpoint_dir'],
            f"checkpoint_epoch_{config['training']['epochs']}.pth"
        )
    else:
        best_ckpt = args.checkpoint
        if best_ckpt is None:
            raise ValueError("--checkpoint required with --eval-only")

    # ── Evaluation ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Running CIFAR-10 linear probe evaluation...")
    print("=" * 60)
    acc = run_linear_probe(
        checkpoint_path=best_ckpt,
        epochs=config['evaluation']['linear_probe_epochs'],
        lr=config['evaluation']['linear_probe_lr'],
    )

    print("\n" + "=" * 60)
    print("Generating t-SNE visualisation...")
    print("=" * 60)
    os.makedirs('results/phase1', exist_ok=True)
    plot_tsne_cifar(
        checkpoint_path=best_ckpt,
        n_samples=1000,
        save_path='results/phase1/tsne_cifar10.png',
    )

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"Best checkpoint    : {best_ckpt}")
    print(f"Linear probe acc   : {acc*100:.2f}%")
    print(f"Random baseline    : 10.00%")
    print(f"Result             : {'PASS' if acc > 0.80 else 'FAIL'} (target > 80%)")


if __name__ == '__main__':
    main()
