import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from pathlib import Path
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from model import create_model
import utils


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Register hooks on the target layer
        target_module = self.get_target_module()
        if target_module is not None:
            self.hooks.append(target_module.register_forward_hook(forward_hook))
            self.hooks.append(target_module.register_backward_hook(backward_hook))

    def get_target_module(self):
        """Get the target layer module for GradCAM"""
        # For the ImageForgeryDetection model, we'll target the last convolutional layer
        # which is typically the most informative for visualization
        if hasattr(self.model, "backbone"):
            # Target the last conv layer in backbone
            if hasattr(self.model.backbone, "conv_proj"):
                return self.model.backbone.conv_proj
            elif hasattr(self.model.backbone, "ds3"):
                return self.model.backbone.ds3
        return None

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_cam(self, input_image, target_class=None):
        """Generate GradCAM for the input image"""
        self.model.eval()

        # Forward pass
        model_output = self.model(input_image)
        if isinstance(model_output, tuple):
            logits = model_output[1]
        else:
            logits = model_output

        # If no target class specified, use the predicted class
        if target_class is None:
            target_class = logits.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        logits[:, target_class].backward()

        # Get gradients and activations
        if self.gradients is None or self.activations is None:
            raise ValueError(
                "Gradients or activations not captured. Check if hooks are working."
            )

        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the channels by corresponding gradients
        cam = torch.zeros(
            self.activations.shape[2:],
            dtype=torch.float32,
            device=self.activations.device,
        )
        for i, w in enumerate(pooled_gradients):
            cam += w * self.activations[0, i, :, :]

        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()

    def overlay_cam(self, original_image, cam, alpha=0.6):
        """Overlay GradCAM on the original image"""
        # Resize CAM to match original image size
        cam_resized = cv2.resize(
            cam, (original_image.shape[1], original_image.shape[0])
        )

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Normalize original image to 0-1 range
        if original_image.max() > 1.0:
            original_image = original_image / 255.0

        # Convert to uint8 for overlay
        original_uint8 = (original_image * 255).astype(np.uint8)

        # Overlay
        output = heatmap * alpha + original_uint8 * (1 - alpha)
        output = output.astype(np.uint8)

        return output


def preprocess_image(image_path, input_size=224):
    """Preprocess image for model input"""
    # Load image
    if image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.open(image_path)

    # Resize
    image = image.resize((input_size, input_size))

    # Convert to tensor
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

    return image_tensor, np.array(image)


def generate_ela_image(image_tensor, quality=75):
    """Generate ELA (Error Level Analysis) image"""
    import cv2

    # Convert tensor to numpy array
    img_np = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_uint8 = (img_np * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    # Encode with JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode(".jpg", img_bgr, encode_param)

    # Decode
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)

    # Calculate difference
    diff = cv2.absdiff(img_uint8, decoded_rgb)
    ela_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    # Normalize
    ela_normalized = ela_gray.astype(np.float32) / 255.0

    return torch.from_numpy(ela_normalized).unsqueeze(0).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate GradCAM for Image Forgery Detection"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--config", type=str, default="configs/aiot.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="gradcam_output.png",
        help="Path to save output image",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=None,
        help="Target class (0: authentic, 1: diffusion, 2: spliced)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.6, help="Transparency for overlay"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Preprocess image
    print(f"Processing image: {args.image_path}")
    image_tensor, original_image = preprocess_image(
        args.image_path, config["data"]["input_size"]
    )

    # Generate ELA if enabled in config
    if config["model"]["ela"]["enabled"]:
        print("Generating ELA image...")
        ela_tensor = generate_ela_image(image_tensor, config["model"]["ela"]["quality"])
        # Combine RGB + ELA (4 channels)
        input_tensor = torch.cat([image_tensor, ela_tensor], dim=1)
    else:
        # If ELA disabled, duplicate RGB channels to make 4 channels
        input_tensor = torch.cat([image_tensor, image_tensor[:, :1, :, :]], dim=1)

    input_tensor = input_tensor.to(device)

    # Get model prediction
    with torch.no_grad():
        model_output = model({"images": input_tensor})
        if isinstance(model_output, tuple):
            logits = model_output[1]
        else:
            logits = model_output

        probabilities = F.softmax(logits, dim=1)
        predicted_class = logits.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    class_names = config["data"]["class_names"]
    print(
        f"Predicted class: {class_names[predicted_class]} (confidence: {confidence:.3f})"
    )

    # Generate GradCAM
    print("Generating GradCAM...")
    gradcam = GradCAM(model, target_layer=None)

    try:
        cam = gradcam.generate_cam(
            {"images": input_tensor}, target_class=args.target_class or predicted_class
        )

        # Overlay on original image
        output_image = gradcam.overlay_cam(original_image, cam, alpha=args.alpha)

        # Save result
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis("off")

        # GradCAM heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap="jet")
        plt.title("GradCAM Heatmap")
        plt.axis("off")

        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(output_image)
        plt.title(f"GradCAM Overlay\nPredicted: {class_names[predicted_class]}")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(args.output_path, dpi=300, bbox_inches="tight")
        print(f"GradCAM saved to: {args.output_path}")

        # Also save just the overlay image
        overlay_path = args.output_path.replace(".png", "_overlay.png")
        plt.imsave(overlay_path, output_image)
        print(f"Overlay image saved to: {overlay_path}")

    except Exception as e:
        print(f"Error generating GradCAM: {e}")
        print(
            "This might be due to model architecture changes. Trying alternative approach..."
        )

        # Alternative: save individual components
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(original_image)
        plt.title(
            f"Predicted: {class_names[predicted_class]} (confidence: {confidence:.3f})"
        )
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(args.output_path, dpi=300, bbox_inches="tight")
        print(f"Basic visualization saved to: {args.output_path}")

    finally:
        gradcam.remove_hooks()

    print("Done!")


if __name__ == "__main__":
    main()
