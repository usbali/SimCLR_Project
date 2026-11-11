Author - Utkarsh Singh ·
Supervisor - Pierre Marza and Professor Maria Vakalopoulou 
Institution-CentraleSupélec, Université Paris-SaclayDateApril 2026
</div>

What this project does
We implement SimCLR (Chen et al., ICML 2020) from scratch in PyTorch and run three experiments on a frozen encoder:
CIFAR-10 (no labels)
      │
      │  SimCLR pre-training
      │  NT-Xent loss · colour jitter · ResNet-18
      │  200 epochs · batch 512 · τ = 0.5
      ▼
  Encoder  ──────────────────────────────────────────────────────────
      │                    │                          │
      ▼                    ▼                          ▼
 Linear probe        Rotation invariance       Colour invariance
 CIFAR-10            Logged MNIST              Logged Colour MNIST
 85.96% ✓            IS = 0.878 ✓              CI = 0.9953 ✓
Key scientific finding: Colour invariance (CI = 0.9953) is substantially higher than rotation invariance (IS = 0.878). Colour jitter was an explicit augmentation — the encoder was directly trained to ignore colour. Rotation invariance is emergent — it arose as a side-effect of shape learning. The gap of +0.117 directly quantifies the difference between trained and emergent invariance, validating the augmentation-invariance principle of contrastive learning.

Results at a glance
Phase 1 — CIFAR-10 linear probe
Method Architecture Accuracy 
Random baseline—10.00%SimCLR (ours, 200 ep),ResNet-1885.96%SimCLR published rangeResNet-18 is 80–88%
Fully supervised (upper bound)ResNet-18~93%
Phase 2 — Rotation invariance (Logged MNIST)
MetricValue Target Rotation invariance score (IS)0.878> 0.80 PASS 
(IS) std deviation-0.059
MNIST linear probe accuracy-95.66%> 80% PASS
Encoder params  (frozen)

The MNIST probe (95.66%) exceeds the fully supervised CIFAR-10 baseline (~93%), despite the encoder never seeing a single MNIST image during training.

Phase 2 extension — Colour invariance (Logged Colour MNIST)
Metric Value Target Colour invariance score-> (CI)0.9953 > 0.80 
CI std deviation-0.0004——CI − IS (trained vs emergent gap)+ 0.117 CI > IS

Repository structure
simclr_project/
│
├── configs/
│   ├── phase1_config.yaml          # all Phase 1 hyperparameters
│   └── phase2_config.yaml          # Phase 2 + colour invariance settings
│
├── phase1/                         
│   ├── data/
│   │   └── augmentations.py        # SimCLR two-view augmentation pipeline
│   ├── models/
│   │   ├── encoder.py              # ResNet-18 CIFAR-adapted, outputs 512-D h
│   │   └── simclr_model.py         # encoder + 2-layer MLP projection head
│   ├── training/
│   │   ├── trainer.py              # main training loop (tqdm, checkpointing)
│   │   ├── nt_xent_loss.py         # NT-Xent contrastive loss
│   │   └── lr_scheduler.py         # linear warmup + cosine annealing
│   ├── evaluation/
│   │   ├── linear_probe.py         # CIFAR-10 frozen linear probe
│   │   └── tsne_viz.py             # t-SNE of CIFAR-10 features
│   └── utils/
│       ├── checkpointing.py        # save/load, load_encoder() strips proj head
│       └── metrics.py              # AverageMeter, top1_accuracy
│
├── phase2/                         
│   ├── dataset/
│   │   └── logged_mnist.py         # builds 70,000 rotated digit images
│   ├── evaluation/
│   │   ├── feature_extractor.py    # frozen encoder → (70000, 512) features
│   │   ├── invariance_score.py     # METRIC 1: rotation invariance score
│   │   ├── linear_probe_mnist.py   # METRIC 2: MNIST linear probe
│   │   ├── tsne_viz_mnist.py       # t-SNE: 0° only vs all rotations
│   │   ├── colour_invariance.py    # METRIC 3: colour invariance score
│   │   ├── colour_invariance_plots.py  # 2-panel CI figure for report
│   │   └── tsne_viz_colour.py      # t-SNE: one colour vs all 7 colours
│   └── utils/
│       └── results_io.py           # save/load phase2_results.json
│
├── notebooks/
│   ├── Phase1_Training.ipynb       # Phase 1 full walkthrough (Colab-ready)
│   └── SimCLR_Phase2_Colab.ipynb  # Phase 2 evaluation + colour invariance
│
├── results/
│   ├── phase1/                     # training loss curve, t-SNE plots
│   └── phase2/                     # invariance scores JSON, all figures
│
├── assets/                         # figures used in the report
├── train_phase1.py                 # entry point: Phase 1 training
├── evaluate_phase2.py              # entry point: Phase 2 evaluation
├── requirements.txt
├── .gitignore
└── README.md

Quickstart
Prerequisites
bashpip install -r requirements.txt
<details>
<summary>requirements.txt contents</summary>
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
Pillow>=9.5.0
tqdm>=4.65.0
pyyaml>=6.0
pandas>=2.0.0
seaborn>=0.12.0
</details>
Phase 1 — Pre-train SimCLR on CIFAR-10
bashpython train_phase1.py --config configs/phase1_config.yaml
CIFAR-10 downloads automatically. Checkpoints are saved to simclr_checkpoints/ every 5 epochs. Training takes ~10 hours on a single T4 GPU (200 epochs, batch 512).
bash# Resume from a checkpoint
python train_phase1.py --config configs/phase1_config.yaml \
                       --resume simclr_checkpoints/checkpoint_epoch_100.pth

# Evaluation only (skip training)
python train_phase1.py --eval-only \
                       --checkpoint simclr_checkpoints/checkpoint_epoch_200.pth
Phase 2 — Cross-domain evaluation
bash# Full run: rotation invariance + linear probe + colour invariance
python evaluate_phase2.py \
    --checkpoint simclr_checkpoints/checkpoint_epoch_200.pth

# Rotation experiment only
python evaluate_phase2.py --skip-colour

# Colour invariance only (professor's extension)
python evaluate_phase2.py --colour-only

# Reuse existing Logged MNIST images
python evaluate_phase2.py --skip-build
Results saved to results/phase2/phase2_results.json.
Google Colab
Open either notebook directly in Colab — all paths and dependencies are pre-configured:
NotebookWhat it coversPhase1_Training.ipynbFull Phase 1: data, training, linear probe, t-SNESimCLR_Phase2_Colab.ipynbPhase 2: rotation invariance, MNIST probe, colour invariance

Method
SimCLR framework
For each image xx
x in a batch of NN
N images, two random augmentations produce 2N2N
2N views. The encoder f(⋅)f(\cdot)
f(⋅) maps each view to h∈R512\mathbf{h} \in \mathbb{R}^{512}
h∈R512, and a 2-layer MLP projection head g(⋅)g(\cdot)
g(⋅) maps to z∈R128\mathbf{z} \in \mathbb{R}^{128}
z∈R128.

The NT-Xent loss maximises agreement between the two views of the same image while separating all other pairs:
ℓi,j=−log⁡exp⁡(sim(zi,zj)/τ)∑k≠iexp⁡(sim(zi,zk)/τ)\ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k \neq i} \exp(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}ℓi,j​=−log∑k=i​exp(sim(zi​,zk​)/τ)exp(sim(zi​,zj​)/τ)​
Augmentation pipeline
TransformSettingRandom resized cropscale ∈ [0.2, 1.0]Random horizontal flipp = 0.5Colour jitterbrightness, contrast, saturation, hue · p = 0.8Random grayscalep = 0.2Gaussian blurσ ∈ [0.1, 2.0] · p = 0.5
Colour jitter is explicitly included → the encoder is trained to be colour-invariant.
Large-angle rotation is not included → rotation invariance is only emergent.
Encoder
ResNet-18 modified for CIFAR-10 (32×32 input):

7×7 stem conv → 3×3 conv (stride 1)
First max-pool → Identity
Final FC → Identity — outputs 512-D feature vector h

Training hyperparameters
ParameterValueEpochs200Batch size512Temperature τ0.5OptimiserAdamLearning rate (peak)1×10⁻³LR warmup9 epochs (linear)LR decayCosine annealing to 0Weight decay1×10⁻⁴

The invariance experiments
Logged MNIST — rotation invariance
10,000 MNIST test digits × 7 rotation angles (−45° to +45° in 15° steps) = 70,000 images.
The encoder is frozen; 0 parameters are updated.
Rotation invariance score for digit class cc
c:

IS(c)=cos⁡ ⁣(hˉc, 0°,  hˉc, 45°)\mathrm{IS}(c) = \cos\!\left(\bar{\mathbf{h}}_{c,\,0°},\;\bar{\mathbf{h}}_{c,\,45°}\right)IS(c)=cos(hˉc,0°​,hˉc,45°​)
All 10 classes exceed 0.80. Digit 8 scores highest (0.903 — nearly rotation-symmetric). Digit 7 scores lowest (0.844 — a rotated 7 resembles an L).
Logged Colour MNIST — colour invariance
10,000 MNIST test digits × 7 colours (red, blue, green, orange, purple, cyan, white) = 70,000 images, all at 0° rotation.
Colour invariance score for digit class cc
c:

CI(c)=16∑k≠redcos⁡ ⁣(hˉc, red,  hˉc, k)\mathrm{CI}(c) = \frac{1}{6}\sum_{k \neq \mathrm{red}} \cos\!\left(\bar{\mathbf{h}}_{c,\,\mathrm{red}},\;\bar{\mathbf{h}}_{c,\,k}\right)CI(c)=61​k=red∑​cos(hˉc,red​,hˉc,k​)
Why CI >> IS
Rotation IS = 0.878Colour CI = 0.9953In augmentation pipeline?❌ No✅ YesType of invarianceEmergentTrainedGradient signal during trainingNoneMillions of stepsVector angle in 512-D space~28°~6°
The +0.117 gap is a direct empirical measurement of the difference between a trained invariance and an emergent one. This validates the core principle of contrastive learning: the augmentation strategy directly controls the invariance structure of the learned representation.
