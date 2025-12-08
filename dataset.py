import numpy as np
import torch
import os
import cv2
from pathlib import Path
import logging
import torchvision.transforms as transforms


class ImageForgeryDataset(torch.utils.data.Dataset):
    def __init__(self, config, split, transform=None):
        self.config = config
        self.split = split
        self.transform = transform

        self.root = config.get("root", "./data")
        self.input_size = config.get("input_size", 224)
        self.class_names = config.get("class_names", ["authentic", "ai", "splicing"])
        self.num_classes = config.get("num_classes", 3)

        self.data_dir = Path(self.root) / split
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.label2id = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.id2label = {i: class_name for class_name, i in self.label2id.items()}

        self.samples = self._load_samples()

        logging.info(f"Loaded {len(self.samples)} samples for {split} split")
        logging.info(f"Class distribution: {self._get_class_distribution()}")

    def _load_samples(self):
        samples = []

        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logging.warning(f"Class directory not found: {class_dir}")
                continue

            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".TIF"]
            for ext in image_extensions:
                image_files = list(class_dir.glob(f"*{ext}"))
                image_files.extend(list(class_dir.glob(f"*{ext.upper()}")))

                for image_file in image_files:
                    samples.append(
                        {
                            "image_path": str(image_file),
                            "label": self.label2id[class_name],
                            "class_name": class_name,
                        }
                    )

        return samples

    def _get_class_distribution(self):
        distribution = {}
        for sample in self.samples:
            class_name = sample["class_name"]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution

    def _load_and_preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.input_size, self.input_size))

        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor / 255.0

        return image_tensor

    def _apply_advanced_augmentation(self, image_tensor):
        """Apply advanced augmentation techniques"""
        # Random Cutout
        if torch.rand(1).item() < 0.3:
            height, width = image_tensor.shape[1], image_tensor.shape[2]
            cutout_size = int(min(height, width) * 0.2)
            x = torch.randint(0, width - cutout_size, (1,)).item()
            y = torch.randint(0, height - cutout_size, (1,)).item()
            image_tensor[:, y : y + cutout_size, x : x + cutout_size] = 0

        # Random Gaussian noise
        if torch.rand(1).item() < 0.2:
            noise = torch.randn_like(image_tensor) * 0.1
            image_tensor = torch.clamp(image_tensor + noise, 0, 1)

        # Random brightness/contrast adjustment
        if torch.rand(1).item() < 0.3:
            factor = torch.empty(1).uniform_(0.7, 1.3).item()
            image_tensor = torch.clamp(image_tensor * factor, 0, 1)

        return image_tensor

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            image = self._load_and_preprocess_image(sample["image_path"])

            if self.transform:
                image = self.transform(image)

            # Apply advanced augmentation if enabled
            if isinstance(self.config, dict) and self.config.get("data", {}).get(
                "advanced_augment", False
            ):
                image = self._apply_advanced_augmentation(image)

            return {
                "images": image,
                "labels": torch.tensor(sample["label"], dtype=torch.long),
                "image_path": sample["image_path"],
                "class_name": sample["class_name"],
            }

        except Exception as e:
            logging.error(f"Error loading sample {idx}: {e}")
            dummy_image = torch.randn(3, self.input_size, self.input_size)
            return {
                "images": dummy_image,
                "labels": torch.tensor(0, dtype=torch.long),
                "image_path": sample["image_path"],
                "class_name": "Authentic",
            }

    def __len__(self):
        return len(self.samples)

    def data_collator(self, batch):
        images = torch.stack([item["images"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        return {"images": images, "labels": labels}


def get_transforms(config, split="train"):
    """Get transforms for data augmentation"""
    config = config.get("data", {})
    input_size = config.get("input_size", 224)
    augment = config.get("augment", True)

    if split == "train" and augment:
        # Training transforms with augmentation
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
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


class Datasets(torch.utils.data.Dataset):
    def __init__(self, root, split, shuffle=True, augment=False, transform=None):
        self.root = root
        self.split = split
        self.augment = augment
        self.transform = transform

        config = {
            "data": {
                "root": root,
                "input_size": 224,
                "class_names": ["Authentic", "AI", "Splicing"],
                "num_classes": 3,
                "augment": augment,
            }
        }

        self.dataset = ImageForgeryDataset(config, split, transform)

        if shuffle:
            indices = list(range(len(self.dataset)))
            np.random.shuffle(indices)
            self.indices = indices
        else:
            self.indices = list(range(len(self.dataset)))

    def class_from_dir(self, dir_path):
        return {k: i for i, k in enumerate(os.listdir(dir_path))}

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]

    def apply_augment(self, image):
        if self.augment and np.random.uniform(0, 1) < 0.4:
            if self.transform:
                image = self.transform(image)
        return image

    def __len__(self):
        return len(self.dataset)

    def data_collator(self, batch):
        return self.dataset.data_collator(batch)


def get_training_set(root, config):
    """Get training dataset with augmentation"""
    transform = get_transforms(config, split="train")
    return ImageForgeryDataset(config, "train", transform=transform)


def get_test_set(root, config):
    """Get test dataset without augmentation"""
    transform = get_transforms(config, split="test")
    return ImageForgeryDataset(config, "test", transform=transform)


def create_dataloaders(config, generator=None):
    from torch.utils.data import DataLoader

    train_config = config.get("training", {})
    batch_size = train_config.get("batch_size", 4)

    train_dataset = get_training_set(config["data"]["root"], config)
    val_dataset = get_test_set(config["data"]["root"], config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=train_dataset.data_collator,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.data_collator,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    config = {
        "data": {
            "root": "./data",
            "input_size": 224,
            "class_names": ["Authentic", "AI", "Splicing"],
            "num_classes": 3,
            "augment": True,
        }
    }

    try:
        dataset = ImageForgeryDataset(config, "train")
        print(f"Dataset loaded with {len(dataset)} samples")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Image shape: {sample['images'].shape}")
            print(f"Label: {sample['labels']}")
    except Exception as e:
        print(f"Error testing dataset: {e}")
