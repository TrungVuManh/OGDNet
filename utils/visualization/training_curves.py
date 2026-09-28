"""
Training curves visualization with std shadow.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


def plot_loss_curve(
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    train_std: Optional[np.ndarray] = None,
    val_std: Optional[np.ndarray] = None,
    title: str = 'Training Loss',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot loss curve with optional std shadow.

    Args:
        train_losses: Training losses (mean across folds)
        val_losses: Validation losses (mean across folds)
        train_std: Training loss std (optional)
        val_std: Validation loss std (optional)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    epochs = np.arange(1, len(train_losses) + 1)

    # Plot training loss
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if train_std is not None:
        plt.fill_between(
            epochs,
            train_losses - train_std,
            train_losses + train_std,
            alpha=0.3,
            color='blue'
        )

    # Plot validation loss
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    if val_std is not None:
        plt.fill_between(
            epochs,
            val_losses - val_std,
            val_losses + val_std,
            alpha=0.3,
            color='red'
        )

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_iou_curve(
    val_ious: np.ndarray,
    val_std: Optional[np.ndarray] = None,
    title: str = 'Validation IoU',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot IoU curve with optional std shadow.

    Args:
        val_ious: Validation IoU (mean across folds)
        val_std: Validation IoU std (optional)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    epochs = np.arange(1, len(val_ious) + 1)

    plt.plot(epochs, val_ious, 'g-', label='Val MIoU', linewidth=2)
    if val_std is not None:
        plt.fill_between(
            epochs,
            val_ious - val_std,
            val_ious + val_std,
            alpha=0.3,
            color='green'
        )

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MIoU', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    metrics_history: Dict[str, np.ndarray],
    std_history: Optional[Dict[str, np.ndarray]] = None,
    title: str = 'Training Curves',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> None:
    """
    Plot training curves (loss and IoU) side by side.

    Args:
        metrics_history: Dictionary with 'train_loss', 'val_loss', 'val_miou'
        std_history: Dictionary with std values (optional)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs = np.arange(1, len(metrics_history['train_loss']) + 1)

    # Loss plot
    ax1 = axes[0]
    ax1.plot(epochs, metrics_history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, metrics_history['val_loss'], 'r-', label='Val', linewidth=2)

    if std_history:
        if 'train_loss' in std_history:
            ax1.fill_between(
                epochs,
                metrics_history['train_loss'] - std_history['train_loss'],
                metrics_history['train_loss'] + std_history['train_loss'],
                alpha=0.3, color='blue'
            )
        if 'val_loss' in std_history:
            ax1.fill_between(
                epochs,
                metrics_history['val_loss'] - std_history['val_loss'],
                metrics_history['val_loss'] + std_history['val_loss'],
                alpha=0.3, color='red'
            )

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curve', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # IoU plot - support both val_iou (crack class) and val_miou (legacy);
    # also plot train_iou alongside val_iou if available.
    ax2 = axes[1]
    iou_key = 'val_iou' if 'val_iou' in metrics_history else 'val_miou'
    iou_label = 'Val IoU (crack)' if iou_key == 'val_iou' else 'Val MIoU'

    if 'train_iou' in metrics_history:
        ax2.plot(epochs, metrics_history['train_iou'], 'b-', label='Train IoU', linewidth=2)
    ax2.plot(epochs, metrics_history[iou_key], 'g-', label=iou_label, linewidth=2)

    if std_history and iou_key in std_history:
        ax2.fill_between(
            epochs,
            metrics_history[iou_key] - std_history[iou_key],
            metrics_history[iou_key] + std_history[iou_key],
            alpha=0.3, color='green'
        )

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('IoU', fontsize=12)
    title2 = 'Train vs Val IoU (crack)' if 'train_iou' in metrics_history else 'Validation IoU (crack class)'
    ax2.set_title(title2, fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison_curves(
    methods_data: Dict[str, Dict[str, np.ndarray]],
    metric: str = 'val_miou',
    title: str = 'Method Comparison',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot comparison of multiple methods.

    Args:
        methods_data: Dictionary of method_name -> metrics_history
        metric: Metric to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_data)))

    for i, (method_name, history) in enumerate(methods_data.items()):
        if metric in history:
            epochs = np.arange(1, len(history[metric]) + 1)
            plt.plot(epochs, history[metric], label=method_name,
                    color=colors[i], linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
