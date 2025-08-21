import torch

from utils import calculate_metrics
from train_opt import save_sample_images
from logger import MetricLogger, SmoothedValue


def train_one_epoch(
    args, model, data_loader, optimizer, epoch, print_freq=10, log_dir="logs", config=None
):
    """Train for one epoch"""
    model.train()

    metric_logger = MetricLogger(delimiter="  ", log_dir=log_dir)
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Train epoch: [{epoch}]"

    # Update learning rate
    for param_group in optimizer.param_groups:
        metric_logger.update(lr=param_group["lr"])

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        images = batch["images"].to(args.device)
        labels = batch["labels"].to(args.device)

        # Forward pass
        loss, logits = model({"images": images, "labels": labels})

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        metric_logger.update(loss=loss.item())
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total if total > 0 else 0.0
        metric_logger.update(accuracy=accuracy)

        # Save sample images occasionally
        if batch_idx % (print_freq * 5) == 0 and config:
            class_names = config["data"]["class_names"]
            save_sample_images(
                images, logits, labels, batch_idx, epoch, args.output_dir, class_names
            )

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Train stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_fn(
    args, data_loader, model, epoch, print_freq=100, results_path=None, log_dir="logs"
):
    """Evaluate model"""
    model.eval()

    metric_logger = MetricLogger(delimiter="  ", log_dir=log_dir)
    header = f"Test: [{epoch}]"

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            images = batch["images"].to(args.device)
            labels = batch["labels"].to(args.device)

            # Forward pass
            loss, logits = model({"images": images, "labels": labels})
            
            # Update loss metric
            metric_logger.update(loss=loss.item())

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total if total > 0 else 0.0
            metric_logger.update(accuracy=accuracy)

            # Store for detailed metrics
            all_logits.append(logits)
            all_labels.append(labels)

    # Calculate detailed metrics
    if all_logits:
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        detailed_metrics = calculate_metrics(all_logits, all_labels)
        
        # Update metric logger with detailed metrics
        for metric_name, metric_value in detailed_metrics.items():
            if metric_name != "confusion_matrix":  # Skip confusion matrix
                metric_logger.update(**{metric_name: metric_value})

    metric_logger.synchronize_between_processes()
    print(f"Test stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
