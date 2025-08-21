import torch
import torch.nn.functional as F
import os
from utils import save_metrics, calculate_metrics
from pathlib import Path


def train_one_epoch(model, data_loader, optimizer, device, epoch, args, config):
    """Train for one epoch with detailed logging"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_top5_acc = 0.0

    for iteration, batch in enumerate(data_loader, 1):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        loss, logits = model({"images": images, "labels": labels})
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics
        acc = calculate_accuracy(logits, labels)
        top5_acc = calculate_top5_accuracy(logits, labels)

        total_loss += loss.item()
        total_acc += acc
        total_top5_acc += top5_acc

        if iteration % args.print_freq == 0:
            print(
                f"Epoch [{epoch:3d}]({iteration:4d}/{len(data_loader):4d}): "
                f"Loss: {loss.item():.6f}, "
                f"Acc: {acc:.4f}, "
                f"Top5 Acc: {top5_acc:.4f}"
            )

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_acc / len(data_loader)
    avg_top5_acc = total_top5_acc / len(data_loader)

    return {
        "loss": avg_loss, 
        "accuracy": avg_acc, 
        "top5_accuracy": avg_top5_acc
    }


def evaluate_model(model, data_loader, device, epoch, args, config, save_images=False):
    """Evaluate model and optionally save result images"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_top5_acc = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for test_iter, batch in enumerate(data_loader, 1):
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            loss, logits = model({"images": images, "labels": labels})

            # Calculate metrics
            acc = calculate_accuracy(logits, labels)
            top5_acc = calculate_top5_accuracy(logits, labels)

            total_loss += loss.item()
            total_acc += acc
            total_top5_acc += top5_acc

            all_logits.append(logits)
            all_labels.append(labels)

            if test_iter % args.print_freq == 0:
                print(
                    f"Eval [{epoch:3d}]({test_iter:4d}/{len(data_loader):4d}): "
                    f"Loss: {loss.item():.6f}, "
                    f"Acc: {acc:.4f}, "
                    f"Top5 Acc: {top5_acc:.4f}"
                )

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_acc / len(data_loader)
    avg_top5_acc = total_top5_acc / len(data_loader)

    # Calculate detailed metrics
    if all_logits:
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        detailed_metrics = calculate_metrics(all_logits, all_labels)
    else:
        detailed_metrics = {}

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "top5_accuracy": avg_top5_acc,
        "detailed_metrics": detailed_metrics
    }


def calculate_accuracy(logits, labels):
    """Calculate top-1 accuracy"""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0


def calculate_top5_accuracy(logits, labels, k=5):
    """Calculate top-k accuracy"""
    top_k_preds = torch.topk(logits, k, dim=1).indices
    correct = (top_k_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0


def save_training_state(model, optimizer, scheduler, epoch, metrics, args, config, is_best=False):
    """Save training state with better organization"""
    checkpoint_dir = Path(config["training"]["model_dir"]) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if is_best:
        checkpoint_path = checkpoint_dir / "best_checkpoint.pth"
    else:
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Clean up old checkpoints (keep only last 3)
    if not is_best and epoch > 3:
        old_checkpoint = checkpoint_dir / f"checkpoint_epoch_{epoch - 3}.pth"
        if old_checkpoint.exists():
            old_checkpoint.unlink()

    return checkpoint_path


def load_training_state(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load training state from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})

    print(f"Loaded checkpoint from epoch {epoch}")
    if metrics:
        print(f"Best metrics: {metrics}")

    return epoch, metrics


def print_training_summary(epoch, train_stats, eval_stats, lr, elapsed_time):
    """Print formatted training summary"""
    print("\n" + "=" * 80)
    print(f"EPOCH {epoch:3d} SUMMARY")
    print("=" * 80)
    print(f"Training Loss:   {train_stats['loss']:.6f}")
    print(f"Training Acc:    {train_stats['accuracy']:.4f}")
    print(f"Training Top5:   {train_stats['top5_accuracy']:.4f}")
    print(f"Learning Rate:   {lr:.8f}")
    print(f"Epoch Time:      {elapsed_time:.2f}s")
    
    if eval_stats:
        print(f"Validation Loss: {eval_stats['loss']:.6f}")
        print(f"Validation Acc:  {eval_stats['accuracy']:.4f}")
        print(f"Validation Top5: {eval_stats['top5_accuracy']:.4f}")
        
        if 'detailed_metrics' in eval_stats:
            detailed = eval_stats['detailed_metrics']
            print(f"Precision:       {detailed.get('precision', 0):.4f}")
            print(f"Recall:          {detailed.get('recall', 0):.4f}")
            print(f"F1-Score:        {detailed.get('f1', 0):.4f}")
    
    print("=" * 80 + "\n")


def save_sample_images(images, predictions, labels, batch_idx, epoch, output_dir, class_names):
    """Save sample images with predictions for visualization"""
    output_dir = Path(output_dir) / "sample_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert predictions to class names
    pred_classes = torch.argmax(predictions, dim=1)
    true_classes = labels
    
    # Save first image from batch
    if images.size(0) > 0:
        import matplotlib.pyplot as plt
        import torchvision.transforms as transforms
        
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denormalize = transforms.Normalize(-mean/std, 1/std)
        
        img = denormalize(images[0]).cpu()
        img = torch.clamp(img, 0, 1)
        
        pred_class = class_names[pred_classes[0].item()]
        true_class = class_names[true_classes[0].item()]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Pred: {pred_class}, True: {true_class}")
        plt.axis('off')
        
        filename = f"{output_dir}/epoch_{epoch}_batch_{batch_idx}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
