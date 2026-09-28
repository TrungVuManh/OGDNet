"""
Prediction visualization utilities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List, Optional, Tuple, Union
import torch


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Denormalize image for visualization.

    Args:
        image: Normalized image (C, H, W) or (H, W, C)
        mean: Normalization mean
        std: Normalization std

    Returns:
        Denormalized image (H, W, C) in [0, 255]
    """
    if image.shape[0] == 3:  # (C, H, W)
        image = np.transpose(image, (1, 2, 0))

    mean = np.array(mean)
    std = np.array(std)

    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create overlay visualization with TP, FP, FN colored.

    Args:
        image: RGB image (H, W, 3)
        mask: Ground truth mask (H, W)
        pred: Prediction mask (H, W)
        alpha: Overlay transparency

    Returns:
        Overlay image (H, W, 3)
    """
    overlay = image.copy()

    # Binarize
    mask = (mask > 0.5).astype(bool)
    pred = (pred > 0.5).astype(bool)

    # True Positive (Green)
    tp_mask = mask & pred
    overlay[tp_mask] = (overlay[tp_mask] * (1 - alpha) +
                        np.array([0, 255, 0]) * alpha).astype(np.uint8)

    # False Positive (Red)
    fp_mask = ~mask & pred
    overlay[fp_mask] = (overlay[fp_mask] * (1 - alpha) +
                        np.array([255, 0, 0]) * alpha).astype(np.uint8)

    # False Negative (Blue)
    fn_mask = mask & ~pred
    overlay[fn_mask] = (overlay[fn_mask] * (1 - alpha) +
                        np.array([0, 0, 255]) * alpha).astype(np.uint8)

    return overlay


def save_prediction(
    image: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
    pred: Union[np.ndarray, torch.Tensor],
    save_path: str,
    image_id: str = '',
    figsize: Tuple[int, int] = (16, 4)
) -> None:
    """
    Save single prediction visualization.

    Args:
        image: Input image
        mask: Ground truth mask
        pred: Prediction
        save_path: Path to save figure
        image_id: Image identifier for title
        figsize: Figure size
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()

    # Remove batch and channel dims if needed
    if image.ndim == 4:
        image = image[0]
    if mask.ndim == 4:
        mask = mask[0]
    if pred.ndim == 4:
        pred = pred[0]

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]

    # Denormalize image
    image = denormalize_image(image)

    # Binarize prediction
    pred_binary = (pred > 0.5).astype(np.float32)

    # Create overlay
    overlay = create_overlay(image, mask, pred_binary)

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title('Input')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_binary, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (G:TP, R:FP, B:FN)')
    axes[3].axis('off')

    if image_id:
        fig.suptitle(f'Sample: {image_id}', fontsize=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_prediction_comparison(
    image: np.ndarray,
    mask: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: str,
    image_id: str = '',
    figsize_per_image: Tuple[float, float] = (2, 2)
) -> None:
    """
    Save comparison of predictions from multiple methods.

    Args:
        image: Input image
        mask: Ground truth mask
        predictions: Dictionary of method_name -> prediction
        save_path: Path to save figure
        image_id: Image identifier
        figsize_per_image: Size per subplot
    """
    # Convert if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Handle dimensions
    if image.ndim == 4:
        image = image[0]
    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    image = denormalize_image(image)

    n_methods = len(predictions)
    n_cols = n_methods + 2  # Input + GT + methods

    fig, axes = plt.subplots(1, n_cols,
                             figsize=(figsize_per_image[0] * n_cols,
                                     figsize_per_image[1]))

    # Input
    axes[0].imshow(image)
    axes[0].set_title('Input', fontsize=8)
    axes[0].axis('off')

    # Ground Truth
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('GT', fontsize=8)
    axes[1].axis('off')

    # Predictions
    for i, (method_name, pred) in enumerate(predictions.items()):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if pred.ndim == 4:
            pred = pred[0]
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]

        pred_binary = (pred > 0.5).astype(np.float32)
        axes[i + 2].imshow(pred_binary, cmap='gray')
        axes[i + 2].set_title(method_name, fontsize=8)
        axes[i + 2].axis('off')

    if image_id:
        fig.suptitle(f'Sample: {image_id}', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_image_with_planes(
    image: Union[np.ndarray, torch.Tensor],
    planes: List[Union[np.ndarray, torch.Tensor]],
    title_left: str = '(a) Input image.',
    title_right: str = '(b) Feature planes.',
    caption: str = '',
    cols: int = 4,
    cmap_planes: str = 'gray',
    normalize_planes: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Visualize an input image alongside K derived planes in a grid.

    Produces the same layout as MTSCrack Fig. 8:
      - Left  : original image (tall, spanning all rows)
      - Right : K planes arranged in a ceil(K/cols) × cols grid

    Works for any kind of per-channel outputs:
      * Encoder feature maps  (one channel per plane)
      * Spectral-gate activations
      * Binary threshold planes (like the paper's ERIS output)
      * Attention maps

    Args:
        image:            Input image.  Accepted shapes:
                            (C, H, W) float  — will be denormalized
                            (H, W, C) uint8
                            (H, W)    grayscale
        planes:           List of K arrays/tensors, each (H, W) or (1, H, W).
                          Can also be a single 4-D tensor (1, K, H, W) or
                          (K, H, W).
        title_left:       Sub-caption under the input image.
        title_right:      Sub-caption under the planes grid.
        caption:          Full figure caption shown at the bottom.
        cols:             Number of columns in the planes grid.
        cmap_planes:      Colormap for planes ('gray', 'viridis', 'jet', …).
        normalize_planes: If True, normalise each plane independently to [0,1].
        save_path:        Save path.  If None, shows interactively.
        dpi:              Output DPI.
    """
    # ── Normalise inputs ──────────────────────────────────────────────
    def _to_numpy_image(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 4:
            x = x[0]
        if x.ndim == 3 and x.shape[0] in (1, 3):   # (C, H, W)
            x = np.transpose(x, (1, 2, 0))
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]
        return x

    def _to_numpy_plane(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 4:
            x = x[0]
        if x.ndim == 3:
            x = x[0] if x.shape[0] == 1 else x.mean(0)
        return x.astype(np.float32)

    img = _to_numpy_image(image)

    # Handle bulk tensor input (K, H, W) or (1, K, H, W)
    if isinstance(planes, (np.ndarray, torch.Tensor)):
        bulk = planes
        if isinstance(bulk, torch.Tensor):
            bulk = bulk.detach().cpu().numpy()
        if bulk.ndim == 4:
            bulk = bulk[0]      # (K, H, W)
        planes = [bulk[k] for k in range(bulk.shape[0])]

    plane_list = [_to_numpy_plane(p) for p in planes]
    K = len(plane_list)

    if normalize_planes:
        normed = []
        for p in plane_list:
            lo, hi = p.min(), p.max()
            normed.append((p - lo) / (hi - lo + 1e-8))
        plane_list = normed

    # Denormalize RGB image if float in (C,H,W) format
    if img.dtype != np.uint8:
        if img.ndim == 2:
            img = np.clip(img, 0, 1)
        else:
            img = denormalize_image(
                np.transpose(img, (2, 0, 1)) if img.shape[-1] == 3 else img
            )

    # ── Layout ────────────────────────────────────────────────────────
    rows = (K + cols - 1) // cols
    # Width ratio: left image ≈ same width as 2 plane columns
    left_width = max(2, cols // 2)

    fig = plt.figure(figsize=(left_width * 2 + cols * 1.8, rows * 1.8 + 0.8))

    # GridSpec: 1 column for the input image, cols columns for planes
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        rows, left_width + cols,
        figure=fig,
        wspace=0.05, hspace=0.05,
    )

    # ── Left: input image ─────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[:, :left_width])
    if img.ndim == 2:
        ax_img.imshow(img, cmap='gray', vmin=0, vmax=255 if img.max() > 1 else 1)
    else:
        ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_xlabel(title_left, fontsize=10, labelpad=6)
    ax_img.xaxis.set_label_position('bottom')
    ax_img.tick_params(bottom=False)

    # ── Right: planes grid ────────────────────────────────────────────
    plane_axes = []
    for idx in range(K):
        r, c = divmod(idx, cols)
        ax = fig.add_subplot(gs[r, left_width + c])
        ax.imshow(plane_list[idx], cmap=cmap_planes, vmin=0, vmax=1)
        ax.axis('off')
        plane_axes.append(ax)

    # Hide unused cells
    for idx in range(K, rows * cols):
        r, c = divmod(idx, cols)
        ax = fig.add_subplot(gs[r, left_width + c])
        ax.axis('off')

    # Right sub-caption below the planes block
    # Place it as a figure text below the right block
    right_center_x = (left_width + cols / 2) / (left_width + cols)
    fig.text(
        right_center_x, 0.01, title_right,
        ha='center', va='bottom', fontsize=10,
    )
    fig.text(
        left_width / (2 * (left_width + cols)), 0.01, title_left,
        ha='center', va='bottom', fontsize=10,
    )

    if caption:
        fig.text(0.5, -0.02, caption, ha='center', va='bottom',
                 fontsize=9, style='italic')

    plt.subplots_adjust(bottom=0.08)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_qualitative_comparison(
    raw_image: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor],
    predictions: Dict[str, Union[np.ndarray, torch.Tensor]],
    image_id: str = '',
    threshold: float = 0.5,
    col_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    figsize_per_col: Tuple[float, float] = (2.0, 2.0),
) -> None:
    """
    Qualitative comparison figure: Raw | GT | model1 | model2 | …

    Matches the style of MTSCrack Fig. 17 / Fig. 20:
      Columns: (RW) Raw image, (GT) Ground truth, then one column per model.
    Column abbreviations are shown at the bottom (like the paper's captions).

    Args:
        raw_image:    Input image tensor/array.
        ground_truth: Binary ground-truth mask.
        predictions:  Ordered dict of model_name -> raw probability map.
        image_id:     Optional title shown above the figure.
        threshold:    Binarisation threshold for predictions.
        col_labels:   Override column labels.  Defaults to ['RW','GT'] + model keys.
        save_path:    Save path; None = interactive.
        dpi:          Output DPI.
        figsize_per_col: (width, height) per subplot column.
    """
    def _prep_image(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 4:
            x = x[0]
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = np.transpose(x, (1, 2, 0))
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]
        return x

    def _prep_mask(x):
        x = _prep_image(x)
        if x.dtype != np.uint8:
            x = (x > threshold).astype(np.float32)
        return x

    img = _prep_image(raw_image)
    if img.dtype != np.uint8:
        img = denormalize_image(
            np.transpose(img, (2, 0, 1)) if img.ndim == 3 and img.shape[-1] == 3
            else img
        )

    gt = _prep_mask(ground_truth)

    n_models = len(predictions)
    n_cols = 2 + n_models
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(figsize_per_col[0] * n_cols, figsize_per_col[1]),
    )

    abbrevs = col_labels or (
        ['RW', 'GT'] + [k[:4] for k in predictions.keys()]
    )

    # Raw image
    axes[0].imshow(img if img.ndim == 3 else img, cmap=None if img.ndim == 3 else 'gray')
    axes[0].axis('off')

    # Ground truth
    axes[1].imshow(gt, cmap='gray', vmin=0, vmax=1)
    axes[1].axis('off')

    # Model predictions
    for col, (model_name, pred) in enumerate(predictions.items(), start=2):
        pred_mask = _prep_mask(pred)
        axes[col].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
        axes[col].axis('off')

    # Column labels at the bottom (matching paper caption style)
    label_line = '   '.join(
        f'({ab}) {name}' for ab, name in zip(abbrevs, ['Raw image', 'Ground truth'] +
                                                        list(predictions.keys()))
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.text(0.5, 0.01, label_line, ha='center', va='bottom', fontsize=7.5)

    if image_id:
        fig.suptitle(image_id, fontsize=10, y=1.01)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_comparison_grid(
    samples: List[Dict],
    method_names: List[str],
    save_path: str,
    n_samples: int = 5,
    figsize_per_image: Tuple[float, float] = (1.5, 1.5)
) -> None:
    """
    Create grid comparison of multiple samples and methods.

    Args:
        samples: List of sample dictionaries with 'image', 'mask', 'predictions'
        method_names: List of method names
        save_path: Path to save figure
        n_samples: Number of samples to include
        figsize_per_image: Size per subplot
    """
    n_samples = min(n_samples, len(samples))
    n_cols = len(method_names) + 2  # Input + GT + methods
    n_rows = n_samples

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_image[0] * n_cols,
                                     figsize_per_image[1] * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Column headers
    col_titles = ['Input', 'GT'] + method_names

    for row_idx in range(n_samples):
        sample = samples[row_idx]

        image = sample['image']
        mask = sample['mask']
        predictions = sample['predictions']

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        if image.ndim == 4:
            image = image[0]
        if mask.ndim == 4:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]

        image = denormalize_image(image)

        # Input
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title(col_titles[0], fontsize=9)

        # GT
        axes[row_idx, 1].imshow(mask, cmap='gray')
        axes[row_idx, 1].axis('off')
        if row_idx == 0:
            axes[row_idx, 1].set_title(col_titles[1], fontsize=9)

        # Predictions
        for col_idx, method_name in enumerate(method_names):
            pred = predictions.get(method_name, np.zeros_like(mask))
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            if pred.ndim == 4:
                pred = pred[0]
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred[0]

            pred_binary = (pred > 0.5).astype(np.float32)
            axes[row_idx, col_idx + 2].imshow(pred_binary, cmap='gray')
            axes[row_idx, col_idx + 2].axis('off')
            if row_idx == 0:
                axes[row_idx, col_idx + 2].set_title(col_titles[col_idx + 2], fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
