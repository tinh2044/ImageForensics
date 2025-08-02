import os
import argparse
import logging
import torch

from dataset import create_dataloaders
from models import create_model
from utils import (
    load_config,
    setup_logging,
    set_random_seed,
    get_device,
    get_optimizer,
    get_scheduler,
    save_checkpoints,
    load_checkpoints,
    train_epoch,
    evaluate,
    save_metrics,
    count_model_parameters,
)


def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/sample.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "eval", "test"],
        help="Task to perform: train, eval, or test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=379,
        help="Seed with which to initialize all the random components",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to specific checkpoint to load",
    )

    return parser


def load_model_from_config(config, device):
    model = create_model(config)
    model.to(device)

    checkpoint_config = config.get("checkpoint", {})
    if checkpoint_config.get("resume", False) or checkpoint_config.get(
        "resume_path", ""
    ):
        resume_path = checkpoint_config.get("resume_path", "")
        if resume_path and os.path.exists(resume_path):
            logging.info(f"Loading checkpoint from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)

    return model


def prepare_data_loaders_from_config(config, generator=None):
    return create_dataloaders(config, generator)


def train_model(config, model, train_loader, val_loader, device):
    train_config = config.get("training", {})
    total_epochs = train_config.get("total_epochs", 200)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    start_epoch = 0
    if config.get("checkpoint", {}).get("resume", False):
        start_epoch = load_checkpoints(model, optimizer, config, resume=True)
        logging.info(f"Resuming training from epoch {start_epoch}")

    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(start_epoch, total_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{total_epochs}")

        train_loss, train_acc, train_top5_acc = train_epoch(
            model, train_loader, optimizer, scheduler, config, epoch, total_epochs
        )

        model.eval()
        val_loss, val_acc, val_top5_acc, detailed_metrics = evaluate(
            model, val_loader, config, epoch, total_epochs
        )
        model.train()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logging.info(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
        )
        logging.info(
            f"Epoch {epoch + 1}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if detailed_metrics:
            logging.info(f"Detailed metrics: {detailed_metrics}")
            save_metrics(detailed_metrics, config, epoch)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            logging.info(f"New best validation accuracy: {best_val_acc:.4f}")

        save_checkpoints(
            model, optimizer, config, epoch, metrics=detailed_metrics, is_best=is_best
        )

    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "best_val_acc": best_val_acc,
    }


def evaluate_model(config, model, val_loader, device):
    logging.info("Starting model evaluation...")

    model.eval()
    val_loss, val_acc, val_top5_acc, detailed_metrics = evaluate(
        model, val_loader, config, epoch=0, epochs=1
    )

    logging.info(f"Final Evaluation Results:")
    logging.info(f"Loss: {val_loss:.4f}")
    logging.info(f"Accuracy: {val_acc:.4f}")
    logging.info(f"Top-5 Accuracy: {val_top5_acc:.4f}")

    if detailed_metrics:
        logging.info(f"Detailed metrics: {detailed_metrics}")
        save_metrics(detailed_metrics, config, epoch=0)

    return detailed_metrics


def main(args):
    config = load_config(args.config_path)

    setup_logging(config)
    logging.info(f"Starting {args.task} task with config: {args.config_path}")

    generator = set_random_seed(args.seed)

    device = get_device(config)
    logging.info(f"Using device: {device}")

    if args.resume:
        config["checkpoint"]["resume"] = True
    if args.checkpoint_path:
        config["checkpoint"]["resume_path"] = args.checkpoint_path

    model = load_model_from_config(config, device)

    param_counts = count_model_parameters(model)
    logging.info(f"Model parameters: {param_counts}")

    train_loader, val_loader = prepare_data_loaders_from_config(config, generator)
    logging.info(
        f"Data loaders created: Train: {len(train_loader)}, Val: {len(val_loader)}"
    )

    if args.task == "train":
        results = train_model(config, model, train_loader, val_loader, device)
        logging.info(
            f"Training completed. Best validation accuracy: {results['best_val_acc']:.4f}"
        )
    elif args.task == "test":
        logging.info("Testing model with sample data...")
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            sample_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in sample_batch.items()
            }
            loss, logits = model(sample_batch)
            logging.info(f"Test forward pass successful. Loss: {loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Image Forgery Detection", parents=[get_default_args()], add_help=False
    )
    args = parser.parse_args()
    main(args)
