# SimCLR: Self-Supervised Representation Learning

**Course:** Lab Project on Self-Supervised Learning  
**Institution:** CentraleSupélec, Université Paris-Saclay  
**Authors:** Utkarsh Singh · Arvind Siddhart Reddy  
**Supervisor:** Pierre Marza  
**Professor:** Maria Vakalopoulou  
**Date:** April 2026

---

## Overview

This repository implements and evaluates **SimCLR** (Chen et al., 2020) across two phases:

| Phase | Objective | Key result |
|-------|-----------|------------|
| Phase 1 | Contrastive pre-training on CIFAR-10 (no labels) | Linear probe **85.96%** |
| Phase 2 | Cross-domain rotation invariance on Logged MNIST | IS = **0.878**, probe = **95.66%** |
| Phase 2 ext. | Colour invariance on Logged Colour MNIST (professor's experiment) | CI = **0.9953** |

**Key finding:** Colour invariance (CI = 0.9953) is substantially higher than rotation invariance
(IS = 0.878) because colour jitter was an *explicit* SimCLR augmentation, while rotation
invariance is *emergent*. This directly validates the augmentation-invariance principle.

---

## Repository structure

```
simclr_project/
│
├── configs/
│   ├── phase1_config.yaml          # all Phase 1 hyperparameters
│   └── phase2_config.yaml          # Phase 2 + colour invariance settings
│
├── phase1/                         # ── Utkarsh Singh ──────────────────────
│   ├── data/
│   │   └── augmentations.py        # SimCLR two-view augmentation pipeline
│   ├── models/
│   │   ├── encoder.py              # ResNet-18 CIFAR-adapted, outputs 512-D h
│   │   └── simclr_model.py         # encoder + projection head
│   ├── training/
│   │   ├── trainer.py              # main training loop
│   │   ├── nt_xent_loss.py         # NT-Xent contrastive loss
│   │   └── lr_scheduler.py         # warmup + cosine annealing
│   ├── evaluation/
│   │   ├── linear_probe.py         # CIFAR-10 frozen linear probe
│   │   └── tsne_viz.py             # t-SNE of CIFAR-10 features
│   └── utils/
│       ├── checkpointing.py        # save/load, load_encoder() helper
│       └── metrics.py              # AverageMeter, top1_accuracy
│
├── phase2/                         # ── Siddhart Reddy ──────────────────────
│   ├── dataset/
│   │   └── logged_mnist.py         # build 70,000 rotated digit images
│   ├── evaluation/
│   │   ├── feature_extractor.py    # frozen encoder → [70000, 512]
│   │   ├── invariance_score.py     # METRIC 1: rotation invariance score
│   │   ├── linear_probe_mnist.py   # METRIC 2: MNIST linear probe
│   │   ├── tsne_viz_mnist.py       # t-SNE: rotation experiment
│   │   ├── colour_invariance.py    # METRIC 3: colour invariance score
│   │   ├── colour_invariance_plots.py  # publication-quality CI plots
│   │   └── tsne_viz_colour.py      # t-SNE: colour experiment
│   └── utils/
│       └── results_io.py           # save/load phase2_results.json
│
├── notebooks/
│   ├── Phase1_Training.ipynb       # Phase 1 Colab notebook
│   └── SimCLR_Phase2_Colab.ipynb  # Phase 2 + colour invariance Colab notebook
│
├── results/
│   ├── phase1/                     # training loss, t-SNE plots
│   └── phase2/                     # invariance JSON, all figures
│
├── assets/                         # figures used in the report
│
├── train_phase1.py                 # entry point — Phase 1 training
├── evaluate_phase2.py              # entry point — Phase 2 evaluation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Phase 1 — pre-train SimCLR on CIFAR-10
```bash
python train_phase1.py --config configs/phase1_config.yaml
```
Best checkpoint saved to `simclr_checkpoints/checkpoint_epoch_200.pth`.

### 3. Phase 2 — full evaluation (rotation + colour)
```bash
python evaluate_phase2.py --config configs/phase2_config.yaml \
                           --checkpoint simclr_checkpoints/checkpoint_epoch_200.pth
```

### 4. Phase 2 — rotation only (skip colour experiment)
```bash
python evaluate_phase2.py --skip-colour
```

### 5. Phase 2 — colour invariance only
```bash
python evaluate_phase2.py --colour-only
```

### 6. Colab notebooks
- Phase 1: `notebooks/Phase1_Training.ipynb`
- Phase 2: `notebooks/SimCLR_Phase2_Colab.ipynb` (includes colour invariance cells)

---

## Results

### Phase 1 — CIFAR-10

| Method | Architecture | Accuracy |
|--------|-------------|----------|
| Random baseline | — | 10.00% |
| SimCLR (ours, 200 ep) | ResNet-18 | **85.96%** |
| SimCLR published range | ResNet-18 | 80–88% |
| Fully supervised | ResNet-18 | ~93% |

### Phase 2 — Rotation invariance (Logged MNIST)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Rotation IS (overall) | 0.878 | > 0.80 | PASS ✓ |
| IS std deviation | 0.059 | — | — |
| MNIST linear probe | 95.66% | > 80% | PASS ✓ |
| Trainable params | 0 | 0 | ✓ |

### Phase 2 extension — Colour invariance (Logged Colour MNIST)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Colour CI (overall) | 0.9953 | > 0.80 | PASS ✓ |
| CI std deviation | 0.0004 | — | — |
| CI − IS (trained vs emergent) | +0.117 | CI > IS | ✓ |

**Interpretation:** Colour invariance (0.9953) >> Rotation invariance (0.878)
because colour jitter was an explicit augmentation during training, while
rotation invariance is emergent. This validates the augmentation-invariance
principle of contrastive learning.

---

## Work distribution

| Task | Owner | Hours |
|------|-------|-------|
| Literature review | US & SR | 8 |
| SimCLR implementation (encoder, head, loss) | Utkarsh Singh | 6 |
| Training loop, LR schedule, checkpointing | Utkarsh Singh | 4 |
| GPU training run (200 epochs) | Utkarsh Singh | 10 |
| Phase 1 evaluation (linear probe, t-SNE) | Siddhart Reddy | 5 |
| Logged MNIST construction | Siddhart Reddy | 5 |
| Phase 2 evaluation (rotation invariance, probe) | Siddhart Reddy | 9 |
| Colour invariance experiment (extension) | Siddhart Reddy | 4 |
| Visualisations & figures | Siddhart Reddy | 5 |
| Presentation slides | Siddhart Reddy | 4 |
| Report writing | US & SR | 6 |
| **Total** | | **~70 h** |

---

## Reference

Chen, T., Kornblith, S., Norouzi, M., Hinton, G. (2020).  
*A Simple Framework for Contrastive Learning of Visual Representations.*  
ICML 2020. https://arxiv.org/abs/2002.05709
