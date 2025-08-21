import os
import torch
import yaml
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config):
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))

    if log_config.get("save_logs", True):
        log_dir = Path(config["training"]["model_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{config['training']['experiment_name']}.log"

        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(
            level=level, format="%(asctime)s [%(levelname)s] %(message)s"
        )


def set_random_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device(config):
    device_name = config.get("device", "cuda")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def top_k_accuracy(logits, labels, k=5):
    top_k_preds = torch.topk(logits, k, dim=1).indices
    correct = (top_k_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
    total = labels.size(0)
    return correct / total


def calculate_metrics(logits, labels, class_names=None):
    preds = torch.argmax(logits, dim=1)

    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    acc = accuracy_score(labels_np, preds_np)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="weighted", zero_division=0
    )

    cm = confusion_matrix(labels_np, preds_np)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }

    return metrics


def save_checkpoints(model, optimizer, config, epoch, metrics=None, is_best=False):
    checkpoint_config = config.get("checkpoint", {})
    save_dir = Path(checkpoint_config.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "metrics": metrics,
    }

    if checkpoint_config.get("save_last", True):
        latest_path = save_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        logging.info(f"Saved latest checkpoint to {latest_path}")

    if is_best and checkpoint_config.get("save_best", True):
        best_path = save_dir / "best.pth"
        torch.save(checkpoint, best_path)
        logging.info(f"Saved best checkpoint to {best_path}")

    epoch_path = save_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, epoch_path)


def load_checkpoints(model, optimizer, config, resume=True):
    checkpoint_config = config.get("checkpoint", {})
    resume_path = checkpoint_config.get("resume_path", "")

    if not resume_path:
        if resume:
            save_dir = Path(checkpoint_config.get("save_dir", "./checkpoints"))
            latest_path = save_dir / "latest.pth"
            if latest_path.exists():
                resume_path = str(latest_path)

    if not resume_path or not os.path.exists(resume_path):
        if resume:
            logging.warning(f"No checkpoint found at {resume_path}")
        return 0

    logging.info(f"Loading checkpoint from {resume_path}")
    checkpoint = torch.load(resume_path, map_location="cpu")

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    if "optimizer" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if resume:
        return checkpoint.get("epoch", 0) + 1
    else:
        return 0


def train_epoch(
    model, dataloader, optimizer, scheduler=None, config=None, epoch=0, epochs=0
):
    model.train()
    all_loss, all_acc, all_top_5_acc = 0.0, 0.0, 0.0

    loop = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        leave=True,
        desc=f"Training epoch {epoch + 1}/{epochs}: ",
    )

    for i, data in loop:
        images = data["images"]
        labels = data["labels"]

        optimizer.zero_grad()

        loss, logits = model(data)
        loss.backward()
        optimizer.step()

        all_loss += loss.item()
        acc = accuracy(logits, labels)
        top_5_acc = top_k_accuracy(logits, labels, k=min(5, logits.shape[1]))

        all_acc += acc
        all_top_5_acc += top_5_acc

        loop.set_postfix_str(
            f"Loss: {loss.item():.3f}, Acc: {acc:.3f}, Top 5 Acc: {top_5_acc:.3f}"
        )

    if scheduler:
        scheduler.step(all_loss / len(dataloader))

    all_loss /= len(dataloader)
    all_acc /= len(dataloader)
    all_top_5_acc /= len(dataloader)

    return all_loss, all_acc, all_top_5_acc


def evaluate(model, dataloader, config=None, epoch=0, epochs=0):
    model.eval()
    all_loss, all_acc, all_top_5_acc = 0.0, 0.0, 0.0
    all_logits = []
    all_labels = []

    loop = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        leave=True,
        desc=f"Evaluation epoch {epoch + 1}/{epochs}: ",
    )

    with torch.no_grad():
        for i, data in loop:
            images = data["images"]
            labels = data["labels"]

            loss, logits = model(data)

            all_loss += loss.item()
            acc = accuracy(logits, labels)
            top_5_acc = top_k_accuracy(logits, labels, k=min(5, logits.shape[1]))

            all_acc += acc
            all_top_5_acc += top_5_acc

            all_logits.append(logits)
            all_labels.append(labels)

            loop.set_postfix_str(
                f"Loss: {loss.item():.3f}, Acc: {acc:.3f}, Top 5 Acc: {top_5_acc:.3f}"
            )

    all_loss /= len(dataloader)
    all_acc /= len(dataloader)
    all_top_5_acc /= len(dataloader)

    if all_logits:
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        detailed_metrics = calculate_metrics(all_logits, all_labels)
    else:
        detailed_metrics = {}

    return all_loss, all_acc, all_top_5_acc, detailed_metrics


def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total_params, "trainable": trainable_params}


def get_model_info(model, input_shape, device):
    """Get model information including parameters and FLOPs"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    model_info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }

    # Try to calculate FLOPs if thop is available
    try:
        from thop import profile

        input_tensor = torch.randn(input_shape).to(device)
        dummy_labels = torch.randint(0, 3, (input_shape[0],)).to(device)

        # Create wrapper function to handle model's expected input format
        def model_wrapper(x):
            return model({"images": x, "labels": dummy_labels})[1]  # Return logits

        flops, params = profile(model_wrapper, inputs=(input_tensor,), verbose=False)
        model_info.update(
            {
                "flops": flops,
                "flops_str": f"{flops / 1e9:.2f}G",
                "macs_str": f"{flops / 1e6:.2f}M",
                "params_str": f"{params / 1e6:.2f}M",
            }
        )
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs: {e}")

    return model_info


def save_metrics(metrics, config, epoch):
    if not config.get("evaluation", {}).get("save_predictions", False):
        return

    output_dir = Path(config["training"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if "confusion_matrix" in metrics and config.get("evaluation", {}).get(
        "save_confusion_matrix", True
    ):
        cm = metrics["confusion_matrix"]
        class_names = config["data"]["class_names"]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names
        )
        plt.title(f"Confusion Matrix - Epoch {epoch}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        cm_path = output_dir / f"confusion_matrix_epoch_{epoch}.png"
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Saved confusion matrix to {cm_path}")


def get_optimizer(model, config):
    opt_config = config["training"]["optimization"]
    optimizer_name = opt_config["optimizer"]
    lr = float(opt_config["learning_rate"])
    weight_decay = float(opt_config.get("weight_decay", 0))

    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(opt_config.get("betas", [0.9, 0.999])),
        )
    elif optimizer_name.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(opt_config.get("betas", [0.9, 0.999])),
        )
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=opt_config.get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(optimizer, config):
    opt_config = config["training"]["optimization"]
    scheduler_name = opt_config.get("scheduler", "ReduceLROnPlateau")

    if scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config["training"]["optimization"].get("scheduler_factor", 0.1),
            patience=config["training"]["optimization"].get("scheduler_patience", 5),
            verbose=True,
        )
    elif scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["total_epochs"],
            eta_min=opt_config.get("eta_min", 0),
        )
    else:
        return None


def init_distributed_mode(args):
    """Initialize distributed training if needed"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0

    args.distributed = args.world_size > 1

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        args.dist_backend = "nccl"
        print(
            "| distributed init (rank {}): {}".format(args.rank, args.dist_url),
            flush=True,
        )
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.distributed.barrier()
        setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """Setup for distributed training"""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_rank():
    """Get current rank for distributed training"""
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized"""
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def save_on_master(*args, **kwargs):
    """Save only on master process"""
    if is_main_process():
        torch.save(*args, **kwargs)


def is_main_process():
    """Check if current process is main process"""
    return get_rank() == 0


def check_state_dict(model, state_dict):
    """Check if model and state dict are compatible"""
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())

    # Check if all model keys are in state dict
    missing_keys = model_keys - state_dict_keys
    unexpected_keys = state_dict_keys - model_keys

    return len(missing_keys) == 0


if __name__ == "__main__":
    pass
