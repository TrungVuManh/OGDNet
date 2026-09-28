"""
Checkpoint utilities for saving and loading models.
"""

import os
import torch
from typing import Optional, Dict, Any


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metric: float,
    path: str,
    config: Optional[Dict] = None,
    metrics_history: Optional[Dict] = None
) -> None:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metric: Best metric value
        path: Path to save checkpoint
        config: Configuration dictionary
        metrics_history: History of metrics
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': metric,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if config is not None:
        checkpoint['config'] = config

    if metrics_history is not None:
        checkpoint['metrics_history'] = metrics_history

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda',
    strict: bool = False
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint to
        strict: Whether to strictly enforce that the keys in state_dict match

    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(path, map_location=device)

    # Use strict=False by default to handle architecture changes (e.g., added dropout layers)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    if missing_keys:
        print(f'Warning: Missing keys in checkpoint: {missing_keys}')
    if unexpected_keys:
        print(f'Warning: Unexpected keys in checkpoint: {unexpected_keys}')

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_metric': checkpoint.get('best_metric', 0.0),
        'config': checkpoint.get('config', None),
        'metrics_history': checkpoint.get('metrics_history', None)
    }
