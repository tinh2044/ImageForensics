import torch
from torch import nn
import torch.utils.checkpoint
import torchvision.models as models
import cv2
import numpy as np
import os


class ImageForgeryDetection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        model_config = config.get("model", {})

        backbone_name = model_config.get("backbone", "resnet18")
        pretrained = model_config.get("pretrained", True)

        if backbone_name == "resnet18":
            self.resnet = models.resnet18(pretrained=pretrained)
        elif backbone_name == "resnet34":
            self.resnet = models.resnet34(pretrained=pretrained)
        elif backbone_name == "resnet50":
            self.resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        input_channels = model_config.get("input_channels", 4)
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        fusion_config = model_config.get("fusion", {})
        self.cnn_features = fusion_config.get("cnn_features", 1000)
        self.fft_features = fusion_config.get("fft_features", 65536)
        self.hidden_dim = fusion_config.get("hidden_dim", 512)
        self.output_dim = fusion_config.get("output_dim", 3)

        self.fc_input_size = self.cnn_features + self.fft_features
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_config.get("dropout", 0.2)),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.ela_config = model_config.get("ela", {})
        self.ela_enabled = self.ela_config.get("enabled", True)
        self.ela_quality = self.ela_config.get("quality", 75)

        self.fft_config = model_config.get("fft", {})
        self.fft_enabled = self.fft_config.get("enabled", True)
        self.fft_epsilon = float(self.fft_config.get("epsilon", 1e-10))

    def generate_ela_map(self, image, quality=None):
        if not self.ela_enabled:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if quality is None:
            quality = self.ela_quality

        temp_path = "temp_ela.jpg"
        cv2.imwrite(temp_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        ela_image = cv2.imread(temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        if ela_image is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        ela_map = cv2.absdiff(image, ela_image)
        ela_map = cv2.cvtColor(ela_map, cv2.COLOR_BGR2GRAY)

        return ela_map

    def fft_feature_extraction(self, image):
        if not self.fft_enabled:
            return np.zeros(self.fft_features, dtype=np.float32)

        image = image.astype(np.float32)

        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        magnitude_spectrum = 20 * np.log(
            np.abs(fshift).astype(np.float32) + self.fft_epsilon
        )

        features = magnitude_spectrum.flatten().astype(np.float32)

        if len(features) > self.fft_features:
            features = features[: self.fft_features]
        elif len(features) < self.fft_features:
            features = np.pad(
                features, (0, self.fft_features - len(features)), "constant"
            )

        return features

    def forward(self, src_inputs):
        images = src_inputs.get("images", None)
        labels = src_inputs.get("labels", None)

        if images is None:
            raise ValueError("images is None")

        loss = None
        logits = None
        batch_fused = []

        for image in images:
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)

            ela_map = self.generate_ela_map(image_np)

            stacked_input = torch.cat(
                [image, torch.from_numpy(ela_map).unsqueeze(0).float()], dim=0
            )

            cnn_features = self.resnet(stacked_input.unsqueeze(0))

            gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            fft_features = self.fft_feature_extraction(gray_image)

            fused_features = torch.cat(
                [
                    cnn_features,
                    torch.from_numpy(fft_features)
                    .unsqueeze(0)
                    .float()
                    .to(cnn_features.device),
                ],
                dim=1,
            )
            batch_fused.append(fused_features)

        if batch_fused:
            batch_fused = torch.cat(batch_fused, dim=0)

            logits = self.fc(batch_fused)

        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.compute_loss(logits, labels)

        return (loss, logits)

    def compute_loss(self, logits, labels):
        return self.loss_fn(logits, labels)


def create_model(config):
    model_name = config.get("model", {}).get("name", "ImageForgeryDetection")

    if model_name == "ImageForgeryDetection":
        return ImageForgeryDetection(config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    pass
