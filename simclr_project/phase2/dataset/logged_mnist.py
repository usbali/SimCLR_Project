"""
phase2/dataset/logged_mnist.py
Builds the Logged MNIST dataset:
  - Takes all 10,000 MNIST test images
  - Renders each digit at 7 rotation angles on a black RGB canvas
  - Resizes to 32x32 to match the CIFAR-10 encoder input
  - Saves metadata CSV (path, digit, rotation)
Total: 10,000 x 7 = 70,000 images
"""

import os
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torchvision
import torchvision.transforms as T


ROTATIONS = [-45, -30, -15, 0, 15, 30, 45]
COLORS     = [
    (220, 50,  50),   # 0  red
    (50,  80, 220),   # 1  blue
    (50, 200,  80),   # 2  green
    (220, 150,  50),  # 3  orange
    (160,  50, 220),  # 4  purple
    (50,  200, 200),  # 5  cyan
    (220, 220,  50),  # 6  yellow
    (180, 100,  50),  # 7  brown
    (220,  80, 160),  # 8  pink
    (100, 180,  50),  # 9  lime
]


def render_digit(digit_img_np, angle, color, out_size=32):
    """
    Render a grayscale MNIST digit on a black RGB canvas,
    coloured and rotated by `angle` degrees, resized to out_size.
    """
    # Create coloured digit on black background
    canvas = Image.new('RGB', (28, 28), (0, 0, 0))
    digit  = Image.fromarray(digit_img_np, mode='L')
    digit  = digit.rotate(angle, expand=False, fillcolor=0)
    digit_np = np.array(digit)

    canvas_np = np.zeros((28, 28, 3), dtype=np.uint8)
    mask = digit_np > 30
    for ch, c in enumerate(color):
        canvas_np[:, :, ch][mask] = c

    img = Image.fromarray(canvas_np, 'RGB')
    img = img.resize((out_size, out_size), Image.BILINEAR)
    return img


def build_logged_mnist(output_dir='phase2_outputs', out_size=32):
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    mnist = torchvision.datasets.MNIST(
        root='./data', train=False, download=True)
    print(f"MNIST test set: {len(mnist)} images")
    print(f"\nBuilding Logged MNIST ({len(mnist)} x {len(ROTATIONS)} = "
          f"{len(mnist)*len(ROTATIONS):,} images)...")

    rows = []
    for idx in tqdm(range(len(mnist)), desc="Creating images"):
        img_np, label = mnist[idx]
        img_np = np.array(img_np)
        color  = COLORS[label]
        for angle in ROTATIONS:
            fname = f"digit{label}_idx{idx:05d}_rot{angle:+d}.png"
            fpath = os.path.join(img_dir, fname)
            rendered = render_digit(img_np, angle, color, out_size)
            rendered.save(fpath)
            rows.append({'path': fpath, 'label': label, 'rotation': angle})

    csv_path = os.path.join(output_dir, 'logged_mnist_metadata.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'label', 'rotation'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nLogged MNIST ready:")
    print(f"  Total images : {len(rows):,}")
    print(f"  Classes      : {list(range(10))}")
    print(f"  Rotations    : {ROTATIONS}")
    print(f"  Saved CSV    : {csv_path}")
    return csv_path
