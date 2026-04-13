"""
phase2/evaluation/feature_extractor.py
Passes all 70,000 Logged MNIST images through the frozen encoder
to produce a (70000, 512) feature matrix.
Encoder parameters are NOT updated — zero gradients computed.
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import time

from phase1.utils.checkpointing import load_encoder


NORMALIZE = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]),
])


class LoggedMNISTDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.transform = transform or NORMALIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row['path']).convert('RGB')
        label = int(row['label'])
        rot   = int(row['rotation'])
        if self.transform:
            img = self.transform(img)
        return img, label, rot


def extract_all_features(checkpoint_path, csv_path,
                         batch_size=512, num_workers=4):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = load_encoder(checkpoint_path, device)

    dataset = LoggedMNISTDataset(csv_path)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers)

    print(f"Extracting features from {len(dataset):,} images...")
    print("Encoder frozen — zero gradients computed")

    all_feats, all_labels, all_rots = [], [], []
    t0 = time.time()

    with torch.no_grad():
        for imgs, labels, rots in tqdm(loader, desc="Extracting"):
            feats = encoder(imgs.to(device)).cpu()
            all_feats.append(feats)
            all_labels.append(labels)
            all_rots.append(rots)

    feats  = torch.cat(all_feats)
    labels = torch.cat(all_labels)
    rots   = torch.cat(all_rots)

    elapsed = int(time.time() - t0)
    print(f"Done in {elapsed}s")
    print(f"features  : {feats.shape}  dtype={feats.dtype}")
    print(f"labels    : {labels.shape}")
    print(f"rotations : {rots.shape}")
    print(f"\nFeature stats:")
    print(f"  mean : {feats.mean():.4f}")
    print(f"  std  : {feats.std():.4f}")
    print(f"  min  : {feats.min():.4f}")
    print(f"  max  : {feats.max():.4f}")

    return feats, labels, rots
