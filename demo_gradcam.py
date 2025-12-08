#!/usr/bin/env python3
"""
Demo script for GradCAM visualization of Image Forgery Detection model
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

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
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize
    image = image.resize((input_size, input_size))

    # Convert to tensor
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

    return image_tensor, np.array(image)


def generate_ela_image(image_tensor, quality=75):
    """Generate ELA (Error Level Analysis) image"""
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


def overlay_cam(original_image, cam, alpha=0.6):
    """Overlay GradCAM on the original image"""
    # Resize CAM to match original image size
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

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


def main():
    # Configuration
    image_path = "images/input.jpg"  # Change this to your image path
    checkpoint_path = (
        "outputs/aiot/best_checkpoint.pth"  # Change this to your checkpoint path
    )
    output_path = "gradcam_demo.png"

    # Check if files exist
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Please place an image in the images/ folder or change the image_path")
        return

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first or change the checkpoint_path")
        return

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    print("Creating model...")
    config = {
        "model": {
            "cnn_features": 256,
            "fft_K": 16,
            "hidden_dim": 128,
            "output_dim": 3,
            "dropout": 0.2,
            "ela": {"enabled": True, "quality": 75},
        },
        "data": {
            "input_size": 224,
            "class_names": ["authentic", "diffusion", "spliced"],
        },
    }

    model = create_model(config)
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Preprocess image
    print(f"Processing image: {image_path}")
    image_tensor, original_image = preprocess_image(image_path, 224)

    # Generate ELA
    print("Generating ELA image...")
    ela_tensor = generate_ela_image(image_tensor, quality=75)

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
    print(
        f"Predicted class: {class_names[predicted_class]} (confidence: {confidence:.3f})"
    )

    # Generate GradCAM
    print("Generating GradCAM...")
    cam = simple_gradcam(model, input_tensor, target_class=predicted_class)

    # Overlay on original image
    output_image = overlay_cam(original_image, cam, alpha=0.6)

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
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"GradCAM saved to: {output_path}")

    # Also save just the overlay image
    overlay_path = output_path.replace(".png", "_overlay.png")
    plt.imsave(overlay_path, output_image)
    print(f"Overlay image saved to: {overlay_path}")

    print("Done!")


if __name__ == "__main__":
    main()
