import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


class RandomGaussianBlur:
    """Apply Gaussian blur with random radius"""

    def __init__(self, radius_range=(0.1, 2.0), p=0.5):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class RandomNoise:
    """Add random noise to image"""

    def __init__(self, noise_factor=0.1, p=0.3):
        self.noise_factor = noise_factor
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img_tensor = F.to_tensor(img)
            noise = torch.randn_like(img_tensor) * self.noise_factor
            img_tensor = torch.clamp(img_tensor + noise, 0, 1)
            return F.to_pil_image(img_tensor)
        return img


class RandomJPEGCompression:
    """Simulate JPEG compression artifacts"""

    def __init__(self, quality_range=(60, 95), p=0.3):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)
            # Convert to bytes and back to simulate compression
            import io

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        return img


class RandomPerspective:
    """Apply random perspective transformation"""

    def __init__(self, distortion_scale=0.2, p=0.3):
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return F.perspective(img, distortion_scale=self.distortion_scale)
        return img


def get_advanced_transforms(config, split="train"):
    """Get advanced transforms with custom augmentations"""
    data_config = config.get("data", {})
    input_size = data_config.get("input_size", 224)
    augment = data_config.get("augment", True)

    if split == "train" and augment:
        # Advanced training transforms
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                ),
                RandomGaussianBlur(radius_range=(0.1, 1.0), p=0.3),
                RandomNoise(noise_factor=0.05, p=0.2),
                RandomJPEGCompression(quality_range=(70, 95), p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return transform


def get_lightweight_transforms(config, split="train"):
    """Get lightweight transforms for faster training"""
    data_config = config.get("data", {})
    input_size = data_config.get("input_size", 224)
    augment = data_config.get("augment", True)

    if split == "train" and augment:
        # Lightweight training transforms
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        # Validation/test transforms
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return transform


def get_test_time_augmentation():
    """Get transforms for test time augmentation (TTA)"""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def apply_tta(model, image, tta_transforms, num_augments=5):
    """Apply test time augmentation"""
    predictions = []

    # Original image
    with torch.no_grad():
        pred = model({"images": image.unsqueeze(0), "labels": None})[1]
        predictions.append(pred)

    # Augmented versions
    for _ in range(num_augments - 1):
        # Apply random augmentation
        aug_image = tta_transforms(image)
        with torch.no_grad():
            pred = model({"images": aug_image.unsqueeze(0), "labels": None})[1]
            predictions.append(pred)

    # Average predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred


class MixupAugmentation:
    """Mixup augmentation for training"""

    def __init__(self, alpha=0.2, p=0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, batch):
        if random.random() > self.p:
            return batch

        images, labels = batch["images"], batch["labels"]
        batch_size = images.size(0)

        # Generate random mixing weights
        lam = np.random.beta(self.alpha, self.alpha)

        # Shuffle indices
        indices = torch.randperm(batch_size)

        # Mix images and labels
        mixed_images = lam * images + (1 - lam) * images[indices]
        mixed_labels = labels  # Keep original labels for loss calculation

        return {
            "images": mixed_images,
            "labels": mixed_labels,
            "mixup_lam": lam,
            "mixup_indices": indices,
        }


class CutMixAugmentation:
    """CutMix augmentation for training"""

    def __init__(self, alpha=1.0, p=0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, batch):
        if random.random() > self.p:
            return batch

        images, labels = batch["images"], batch["labels"]
        batch_size, channels, height, width = images.size()

        # Generate random mixing weights
        lam = np.random.beta(self.alpha, self.alpha)

        # Shuffle indices
        indices = torch.randperm(batch_size)

        # Generate random box
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_rat)
        cut_h = int(height * cut_rat)

        cx = np.random.randint(width)
        cy = np.random.randint(height)

        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)

        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[
            indices, :, bby1:bby2, bbx1:bbx2
        ]

        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (width * height))

        return {
            "images": mixed_images,
            "labels": labels,
            "cutmix_lam": lam,
            "cutmix_indices": indices,
        }
