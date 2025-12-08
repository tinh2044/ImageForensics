#!/usr/bin/env python3
"""
Batch GradCAM script for processing multiple images
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import yaml
import os
from tqdm import tqdm

from model import create_model


def simple_gradcam(model, input_tensor, target_class=None):
    """Simple GradCAM implementation"""
    model.eval()

    # Forward pass
    input_tensor.requires_grad_(True)
    model_output = model({"images": input_tensor})

    if isinstance(model_output, tuple):
        logits = model_output[1]
    else:
        logits = model_output

    # If no target class specified, use the predicted class
    if target_class is None:
        target_class = logits.argmax(dim=1)

    # Backward pass
    model.zero_grad()
    logits[:, target_class].backward()

    # Get gradients w.r.t input
    gradients = input_tensor.grad

    # Global average pooling of gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Weight the input channels by corresponding gradients
    cam = torch.zeros(
        input_tensor.shape[2:], dtype=torch.float32, device=input_tensor.device
    )
    for i, w in enumerate(pooled_gradients):
        cam += w * input_tensor[0, i, :, :]

    # Apply ReLU to focus on positive contributions
    cam = F.relu(cam)

    # Normalize
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam.detach().cpu().numpy()


def preprocess_image(image_path, input_size=224):
    """Preprocess image for model input"""
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Resize
        image = image.resize((input_size, input_size))

        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

        return image_tensor, np.array(image)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


def generate_ela_image(image_tensor, quality=75):
    """Generate ELA (Error Level Analysis) image"""
    try:
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
    except Exception as e:
        print(f"Error generating ELA: {e}")
        # Return zero tensor if ELA fails
        return torch.zeros(1, 1, image_tensor.shape[2], image_tensor.shape[3])


def overlay_cam(original_image, cam, alpha=0.6):
    """Overlay GradCAM on the original image"""
    try:
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
    except Exception as e:
        print(f"Error creating overlay: {e}")
        return original_image


def process_single_image(
    model, image_path, output_dir, config, device, target_class=None
):
    """Process a single image and generate GradCAM"""
    try:
        # Preprocess image
        image_tensor, original_image = preprocess_image(
            image_path, config["data"]["input_size"]
        )
        if image_tensor is None:
            return False

        # Generate ELA
        ela_tensor = generate_ela_image(
            image_tensor, quality=config["model"]["ela"]["quality"]
        )

        # Combine RGB + ELA (4 channels)
        input_tensor = torch.cat([image_tensor, ela_tensor], dim=1)
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

        # Generate GradCAM
        cam = simple_gradcam(model, input_tensor, target_class or predicted_class)

        # Overlay on original image
        output_image = overlay_cam(original_image, cam, alpha=0.6)

        # Save results
        image_name = Path(image_path).stem
        overlay_path = output_dir / f"{image_name}_overlay.png"
        plt.imsave(overlay_path, output_image)

        print(f"✓ {image_name}: {class_names[predicted_class]} ({confidence:.3f})")
        return True

    except Exception as e:
        print(f"✗ Error processing {image_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch GradCAM for Image Forgery Detection"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gradcam_outputs",
        help="Output directory for GradCAM images",
    )
    parser.add_argument(
        "--config", type=str, default="configs/aiot.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
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
        "--extensions",
        type=str,
        default="jpg,jpeg,png,bmp,tif",
        help="Image file extensions to process",
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

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find image files
    input_dir = Path(args.input_dir)
    extensions = args.extensions.split(",")
    image_files = []

    for ext in extensions:
        image_files.extend(input_dir.glob(f"*.{ext}"))
        image_files.extend(input_dir.glob(f"*.{ext.upper()}"))

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Process images
    success_count = 0
    for image_path in tqdm(image_files, desc="Processing images"):
        if process_single_image(
            model, image_path, output_dir, config, device, args.target_class
        ):
            success_count += 1

    print(
        f"\nProcessing complete! {success_count}/{len(image_files)} images processed successfully."
    )
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
