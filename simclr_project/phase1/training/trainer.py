"""
phase1/training/trainer.py
Main SimCLR training loop for Phase 1.
Trains the encoder + projection head on CIFAR-10 without labels.
Logs loss and LR per epoch. Saves checkpoints every N epochs.
"""

import os
import time
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase1.models.simclr_model import SimCLR
from phase1.training.nt_xent_loss import NTXentLoss
from phase1.training.lr_scheduler import update_lr
from phase1.data.augmentations import SimCLRAugmentation
from phase1.utils.checkpointing import save_checkpoint


class SimCLRTrainer:

    def __init__(self, config):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._build_model()
        self._build_dataloader()
        self._build_optimizer()
        self.criterion = NTXentLoss(temperature=config['training']['temperature'])
        self.best_loss = float('inf')

    def _build_model(self):
        cfg = self.cfg['model']
        self.model = SimCLR(
            projection_dim=cfg['projection_dim'],
            hidden_dim=cfg['hidden_dim'],
        ).to(self.device)
        print(f"Model on {self.device}")
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total:,}")

    def _build_dataloader(self):
        cfg = self.cfg
        aug = SimCLRAugmentation(
            image_size=32,
            color_jitter_strength=cfg['augmentation']['color_jitter_strength'],
            gaussian_blur=cfg['augmentation']['gaussian_blur'],
            grayscale_prob=cfg['augmentation']['grayscale_prob'],
        )
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=aug
        )
        self.loader = DataLoader(
            dataset,
            batch_size=cfg['training']['batch_size'],
            shuffle=True,
            num_workers=cfg['data']['num_workers'],
            pin_memory=True,
            drop_last=True,
        )
        print(f"Dataset: {len(dataset)} images | "
              f"Batches per epoch: {len(self.loader)}")

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg['training']['learning_rate'],
            weight_decay=self.cfg['training']['weight_decay'],
        )

    def train(self):
        cfg = self.cfg['training']
        total_epochs = cfg['epochs']
        save_every   = self.cfg['checkpointing']['save_every']
        ckpt_dir     = self.cfg['checkpointing']['checkpoint_dir']
        os.makedirs(ckpt_dir, exist_ok=True)

        print(f"\nStarting training: {total_epochs} epochs, "
              f"batch={cfg['batch_size']}, tau={cfg['temperature']}")
        print(f"Color jitter : {'enabled' if self.cfg['augmentation']['color_jitter'] else 'disabled'}")
        print("=" * 60)

        for epoch in range(1, total_epochs + 1):
            lr = update_lr(
                self.optimizer, epoch, total_epochs,
                lr_peak=cfg['learning_rate'],
                lr_start=2e-4, warmup_epochs=9,
            )
            epoch_loss = self._run_epoch(epoch, total_epochs)

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                print(f"  New best loss: {epoch_loss:.4f}")

            if epoch % save_every == 0 or epoch == total_epochs:
                path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pth')
                save_checkpoint(self.model, self.optimizer, epoch,
                                epoch_loss, path)
                print(f"  Checkpoint saved -> {path}")

        print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")

    def _run_epoch(self, epoch, total_epochs):
        self.model.train()
        total_loss = 0.0
        t0 = time.time()

        pbar = tqdm(self.loader, desc=f"Epoch {epoch:03d}")
        for (view1, view2), _ in pbar:
            view1 = view1.to(self.device, non_blocking=True)
            view2 = view2.to(self.device, non_blocking=True)

            _, z1 = self.model(view1)
            _, z2 = self.model(view2)

            loss = self.criterion(z1, z2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.loader)
        elapsed  = int(time.time() - t0)
        lr_now   = self.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{total_epochs} | "
              f"Loss {avg_loss:.4f} | LR {lr_now:.6f} | Time {elapsed}s")
        return avg_loss
