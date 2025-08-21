#!/usr/bin/env python3
"""
Simple training script for image classification
"""

import torch
import yaml
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from models import create_model
from dataset import get_training_set, get_test_set
from optimizer import build_optimizer, build_scheduler
from train_opt import train_one_epoch, evaluate_model, save_training_state
from utils import get_device, set_random_seed, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Simple Image Classification Training")
    parser.add_argument(
        "--config", default="configs/aiot.yaml", help="Path to config file"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    config["training"]["total_epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["device"] = args.device

    # Setup
    set_random_seed(args.seed)
    setup_logging(config)
    device = get_device(config)

    print(f"Training on device: {device}")
    print(f"Configuration: {args.config}")

    # Create datasets
    print("Loading datasets...")
    train_dataset = get_training_set(config["data"]["root"], config)
    test_dataset = get_test_set(config["data"]["root"], config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=train_dataset.data_collator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=test_dataset.data_collator,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)

    # Create optimizer and scheduler
    print("Creating optimizer and scheduler...")
    optimizer = build_optimizer(config["training"]["optimization"], model)
    scheduler, scheduler_name = build_scheduler(
        config["training"]["optimization"], optimizer
    )

    print(f"Model: {type(model).__name__}")
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Scheduler: {scheduler_name}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_accuracy = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        # Train
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args, config
        )

        # Evaluate
        test_stats = evaluate_model(model, test_loader, device, epoch, args, config)

        # Step scheduler
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(train_stats["loss"])
        else:
            scheduler.step()

        # Print results
        print(
            f"Train Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['accuracy']:.4f}"
        )
        print(
            f"Test Loss: {test_stats['loss']:.4f}, Test Acc: {test_stats['accuracy']:.4f}"
        )

        # Save best model
        if test_stats["accuracy"] > best_accuracy:
            best_accuracy = test_stats["accuracy"]
            save_training_state(
                model,
                optimizer,
                scheduler,
                epoch,
                test_stats,
                args,
                config,
                is_best=True,
            )
            print(f"New best accuracy: {best_accuracy:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_training_state(
                model,
                optimizer,
                scheduler,
                epoch,
                test_stats,
                args,
                config,
                is_best=False,
            )

    print(f"\nTraining completed!")
    print(f"Best accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
