"""
phase1/data/augmentations.py
SimCLR augmentation pipeline for CIFAR-10.
Produces two randomly augmented views of each image (positive pair).
Reference: Chen et al. 2020, Appendix A.
"""

import torchvision.transforms as T


class SimCLRAugmentation:
    """
    Applies two independent random augmentations to the same image,
    returning a positive pair (view1, view2) for contrastive training.
    """

    def __init__(self, image_size=32, color_jitter_strength=0.5,
                 gaussian_blur=True, grayscale_prob=0.2):
        self.transform = self._build_transform(
            image_size, color_jitter_strength, gaussian_blur, grayscale_prob
        )

    def _build_transform(self, size, s, blur, gs_prob):
        color_jitter = T.ColorJitter(
            brightness=0.8 * s,
            contrast=0.8 * s,
            saturation=0.8 * s,
            hue=0.2 * s,
        )
        transforms = [
            T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=gs_prob),
        ]
        if blur:
            transforms.append(
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=int(0.1 * size) | 1,
                                    sigma=(0.1, 2.0))],
                    p=0.5,
                )
            )
        transforms += [
            T.ToTensor(),
            T.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ]
        return T.Compose(transforms)

    def __call__(self, x):
        """Return two independently augmented views of x."""
        return self.transform(x), self.transform(x)


class EvalTransform:
    """Standard (non-augmented) transform for linear probe evaluation."""

    def __init__(self, image_size=32):
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ])

    def __call__(self, x):
        return self.transform(x)
