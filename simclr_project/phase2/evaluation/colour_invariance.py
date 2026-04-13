"""
phase2/evaluation/colour_invariance.py
Colour Invariance Score — Metric 3 (professor's extension).

Builds Logged Colour MNIST: 10,000 MNIST test digits × 7 distinct colours
= 70,000 images, all at 0° rotation (only colour varies).

CI(c) = mean over non-ref colours of:
        cosine_sim( mean_feat(c, red), mean_feat(c, colour_i) )

Overall CI = mean of CI(c) across all 10 digit classes.

Key hypothesis: CI > rotation invariance score (0.878) because
colour jitter was an EXPLICIT augmentation during SimCLR pre-training,
while rotation invariance is only EMERGENT.

Expected result: CI ~ 0.99+  (trained invariance, near-perfect)
vs IS ~ 0.878               (emergent invariance, partial)
"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from phase1.utils.checkpointing import load_encoder

# 7 maximally distinct colours spanning the full hue wheel
COLOUR_PALETTE = {
    'red'   : (220,  50,  50),
    'blue'  : ( 50,  80, 220),
    'green' : ( 50, 200,  80),
    'orange': (220, 150,  50),
    'purple': (160,  50, 220),
    'cyan'  : ( 50, 200, 200),
    'white' : (220, 220, 220),
}
COLOUR_NAMES = list(COLOUR_PALETTE.keys())
COLOUR_REF   = 'red'    # reference colour for sweep

NORMALIZE = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]),
])


def render_digit_colour(digit_img_np, colour_rgb, out_size=32):
    """
    Render a grayscale MNIST digit in a given RGB colour on a black canvas.
    Returns a PIL Image of size (out_size, out_size).
    """
    img_f  = digit_img_np.astype(np.float32) / 255.0
    canvas = np.zeros((28, 28, 3), dtype=np.uint8)
    for ch, val in enumerate(colour_rgb):
        canvas[:, :, ch] = (img_f * val).astype(np.uint8)
    img = Image.fromarray(canvas, 'RGB')
    return img.resize((out_size, out_size), Image.BILINEAR)


class LoggedColourMNIST(Dataset):
    """
    MNIST test set rendered at 0° rotation in each of N colours.
    Returns (image_tensor, digit_label, colour_index, original_mnist_index).

    The dataset is built in-memory (no images saved to disk by default)
    to keep things fast. Pass save_dir to also save PNG files.
    """

    def __init__(self, transform=None, save_dir=None, out_size=32):
        self.mnist     = torchvision.datasets.MNIST(
            root='./data', train=False, download=True)
        self.transform = transform or NORMALIZE
        self.save_dir  = save_dir
        self.out_size  = out_size
        self.samples   = []   # (digit_np, label, colour_name, colour_idx, orig_idx)
        self._build()

    def _build(self):
        print(f"Building Logged Colour MNIST "
              f"({len(self.mnist)} × {len(COLOUR_PALETTE)} = "
              f"{len(self.mnist) * len(COLOUR_PALETTE):,} images)...")
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        for idx in tqdm(range(len(self.mnist)), desc="Rendering colours"):
            img_pil, label = self.mnist[idx]
            img_np = np.array(img_pil)
            for ci, (col_name, col_rgb) in enumerate(COLOUR_PALETTE.items()):
                self.samples.append((img_np, label, col_name, ci, idx))

                if self.save_dir:
                    rendered = render_digit_colour(img_np, col_rgb,
                                                   self.out_size)
                    fname = f"{idx:05d}_d{label}_{col_name}.png"
                    rendered.save(os.path.join(self.save_dir, fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_np, label, col_name, col_idx, orig_idx = self.samples[idx]
        col_rgb  = COLOUR_PALETTE[col_name]
        rendered = render_digit_colour(img_np, col_rgb, self.out_size)
        if self.transform:
            rendered = self.transform(rendered)
        return rendered, label, col_idx, orig_idx


def extract_colour_features(checkpoint_path, batch_size=512, num_workers=2,
                              save_dir=None):
    """
    Build Logged Colour MNIST and extract 512-D features from the frozen encoder.
    Returns (feats, labels, col_idx) tensors of shape (70000, 512), (70000,), (70000,).
    """
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = load_encoder(checkpoint_path, device)

    dataset = LoggedColourMNIST(save_dir=save_dir)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers)

    print(f"\nExtracting colour features from {len(dataset):,} images...")
    t0 = time.time()

    all_feats, all_labels, all_cols = [], [], []
    with torch.no_grad():
        for imgs, labels, col_idxs, _ in tqdm(loader, desc="Extracting"):
            feats = encoder(imgs.to(device)).cpu()
            all_feats.append(feats)
            all_labels.append(labels)
            all_cols.append(col_idxs)

    feats  = torch.cat(all_feats)
    labels = torch.cat(all_labels)
    cols   = torch.cat(all_cols)

    print(f"Done in {int(time.time()-t0)}s  |  "
          f"features: {feats.shape}  dtype={feats.dtype}")
    return feats, labels, cols


def compute_colour_invariance(feats, labels, cols,
                               ref_colour=COLOUR_REF,
                               target=0.80,
                               rotation_is=None):
    """
    Compute the colour invariance score from pre-extracted features.

    feats  : (N, 512) tensor
    labels : (N,)     tensor — digit class 0..9
    cols   : (N,)     tensor — colour index (0=red, 1=blue, ...)

    rotation_is : float, optional — if provided, prints head-to-head comparison.
    """
    ref_idx = COLOUR_NAMES.index(ref_colour)

    print("=" * 55)
    print("METRIC 3 — COLOUR INVARIANCE SCORE")
    print("=" * 55)

    # ── Per-class colour invariance ───────────────────────────────────────────
    print(f"\nPer-class colour invariance "
          f"({ref_colour} vs all other colours):")
    per_class = {}
    for c in range(10):
        mask_ref = (labels == c) & (cols == ref_idx)
        mean_ref = feats[mask_ref].mean(dim=0)
        sims = []
        for ci, col_name in enumerate(COLOUR_NAMES):
            if ci == ref_idx:
                continue
            mask_i = (labels == c) & (cols == ci)
            mean_i = feats[mask_i].mean(dim=0)
            sims.append(F.cosine_similarity(
                mean_ref.unsqueeze(0), mean_i.unsqueeze(0)).item())
        score = float(np.mean(sims))
        per_class[c] = score
        bar    = '█' * int(score * 25)
        status = '✓' if score > target else '~'
        print(f"  Digit {c}: {score:.4f}  {bar} {status}")

    overall_mean = float(np.mean(list(per_class.values())))
    overall_std  = float(np.std(list(per_class.values())))

    print(f"\nOverall Colour Invariance Score")
    print(f"  Mean   : {overall_mean:.4f}")
    print(f"  Std    : {overall_std:.4f}")
    print(f"  Target : > {target}")
    print(f"  Result : {'✓ PASS' if overall_mean > target else '✗ FAIL'}")

    # ── Colour sweep ─────────────────────────────────────────────────────────
    print(f"\nColour sweep "
          f"({ref_colour} vs each colour, averaged over all digits):")
    sweep = {}
    for ci, col_name in enumerate(COLOUR_NAMES):
        if ci == ref_idx:
            sweep[col_name] = 1.000
            print(f"  {ref_colour:8s} vs {col_name:8s}: 1.0000  (reference)")
            continue
        sims = []
        for c in range(10):
            m_r = feats[(labels == c) & (cols == ref_idx)].mean(dim=0)
            m_i = feats[(labels == c) & (cols == ci)].mean(dim=0)
            sims.append(F.cosine_similarity(
                m_r.unsqueeze(0), m_i.unsqueeze(0)).item())
        sweep[col_name] = float(np.mean(sims))
        status = '✓' if sweep[col_name] > target else '~'
        print(f"  {ref_colour:8s} vs {col_name:8s}: "
              f"{sweep[col_name]:.4f}  {status}")

    # ── Head-to-head comparison ───────────────────────────────────────────────
    if rotation_is is not None:
        print(f"\n{'─'*55}")
        print(f"HEAD-TO-HEAD: ROTATION vs COLOUR INVARIANCE")
        print(f"{'─'*55}")
        print(f"  Rotation IS (emergent — not in augmentation) : "
              f"{rotation_is:.4f}")
        print(f"  Colour   CI (trained  — explicit colour jitter): "
              f"{overall_mean:.4f}")
        diff = overall_mean - rotation_is
        if diff > 0:
            print(f"  Colour IS higher by {diff:.4f} — hypothesis CONFIRMED ✓")
            print(f"  Explicit augmentation → stronger invariance than emergent.")
        else:
            print(f"  Colour is NOT higher (diff={diff:.4f}) — unexpected.")
        print(f"{'─'*55}")

    return {
        'per_class'    : per_class,
        'overall_mean' : overall_mean,
        'overall_std'  : overall_std,
        'sweep'        : sweep,
        'pass'         : overall_mean > target,
    }
