"""
Precision-Recall curve visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


def plot_pr_curve(
    precisions: np.ndarray,
    recalls: np.ndarray,
    ap: Optional[float] = None,
    label: str = 'Model',
    title: str = 'Precision-Recall Curve',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot single precision-recall curve.

    Args:
        precisions: Precision values (sorted by recall, monotonically decreasing)
        recalls: Recall values (sorted ascending)
        ap: Average precision (optional, for legend)
        label: Label for the curve
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Ensure sorted by recall (ascending) if not already
    if len(recalls) > 1 and recalls[0] > recalls[-1]:
        sorted_idx = np.argsort(recalls)
        recalls = recalls[sorted_idx]
        precisions = precisions[sorted_idx]

    if ap is not None:
        label = f'{label} (AP={ap:.3f})'

    # Plot with step style for proper PR curve appearance
    plt.step(recalls, precisions, where='post', linewidth=2, label=label)

    # Add iso-F1 curves
    f1_scores = np.linspace(0.2, 0.8, 4)
    for f1 in f1_scores:
        x = np.linspace(0.01, 1, 100)
        y = f1 * x / (2 * x - f1)
        y[y < 0] = np.nan
        y[y > 1] = np.nan
        plt.plot(x, y, '--', color='gray', alpha=0.3)
        plt.annotate(f'F1={f1:.1f}', xy=(0.9, f1 * 0.9 / (2 * 0.9 - f1)),
                    fontsize=8, color='gray')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pr_curves_comparison(
    pr_data: Dict[str, Dict[str, np.ndarray]],
    title: str = 'Precision-Recall Curves Comparison',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot multiple precision-recall curves for comparison.

    Args:
        pr_data: Dictionary of method_name -> {'precisions': array, 'recalls': array, 'ap': float}
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(pr_data)))

    # Sort methods by AP for legend ordering
    sorted_methods = sorted(pr_data.items(),
                           key=lambda x: x[1].get('ap', 0),
                           reverse=True)

    for i, (method_name, data) in enumerate(sorted_methods):
        precisions = data['precisions'].copy()
        recalls = data['recalls'].copy()
        ap = data.get('ap', None)

        # Ensure sorted by recall (ascending) if not already
        if len(recalls) > 1 and recalls[0] > recalls[-1]:
            sorted_idx = np.argsort(recalls)
            recalls = recalls[sorted_idx]
            precisions = precisions[sorted_idx]

        label = method_name
        if ap is not None:
            label = f'{method_name} (AP={ap:.3f})'

        # Plot with step style for proper PR curve appearance
        plt.step(recalls, precisions, where='post', color=colors[i], linewidth=2, label=label)

    # Add iso-F1 curves
    f1_scores = np.linspace(0.2, 0.8, 4)
    for f1 in f1_scores:
        x = np.linspace(0.01, 1, 100)
        y = f1 * x / (2 * x - f1)
        y[y < 0] = np.nan
        y[y > 1] = np.nan
        plt.plot(x, y, '--', color='gray', alpha=0.2)

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metric_vs_threshold(
    threshold_data: Dict[str, Dict[str, np.ndarray]],
    metric: str = 'f1',
    title: str = 'F1-Score vs Decision Threshold',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    with_std: bool = False,
) -> None:
    """
    Plot a metric (F1, IoU, Precision, Recall) vs decision threshold for multiple models.

    Produces the same chart style as MTSCrack Fig. 14-16 ("FSIM on normal/high/wet-shadow
    illumination images"), but using crack-detection metrics instead of FSIM.

    Args:
        threshold_data: dict of method_name ->
            {
              'thresholds': 1-D array of threshold values (ascending),
              'f1':         1-D array of F1 at each threshold,
              'precision':  1-D array,
              'recall':     1-D array,
              'iou':        1-D array,
              'f1_std':     1-D array (optional, for shaded std band),
            }
        metric:     One of 'f1', 'precision', 'recall', 'iou'.
        title:      Plot title.
        save_path:  If given, saves the figure; otherwise calls plt.show().
        figsize:    Figure size (width, height).
        with_std:   If True, draw a shaded ±std band around each line
                    (requires '<metric>_std' key in threshold_data).
    """
    METRIC_LABELS = {
        'f1': 'F1-Score',
        'precision': 'Precision',
        'recall': 'Recall',
        'iou': 'IoU',
        'miou': 'mIoU',
        'pa': 'PA',
        'mpa': 'MPA',
    }

    # Colour + marker scheme that matches the paper's style
    STYLE_CYCLE = [
        {'color': '#2196F3', 'marker': 'o', 'ls': '-'},   # blue  circle
        {'color': '#F44336', 'marker': 's', 'ls': '-'},   # red   square
        {'color': '#4CAF50', 'marker': '^', 'ls': '-'},   # green triangle
        {'color': '#FF9800', 'marker': 'D', 'ls': '-'},   # orange diamond
        {'color': '#9C27B0', 'marker': 'v', 'ls': '-'},   # purple inverted-tri
        {'color': '#00BCD4', 'marker': 'P', 'ls': '-'},   # cyan  plus
        {'color': '#795548', 'marker': '*', 'ls': '-'},   # brown star
        {'color': '#607D8B', 'marker': 'h', 'ls': '-'},   # grey  hexagon
    ]

    fig, ax = plt.subplots(figsize=figsize)

    # Sort methods by peak metric value (descending) for a clean legend
    sorted_methods = sorted(
        threshold_data.items(),
        key=lambda kv: float(np.max(kv[1].get(metric, [0]))),
        reverse=True,
    )

    for idx, (method_name, data) in enumerate(sorted_methods):
        thresholds = np.asarray(data['thresholds'])
        values = np.asarray(data.get(metric, np.zeros_like(thresholds)))
        style = STYLE_CYCLE[idx % len(STYLE_CYCLE)]

        # Sort by threshold ascending
        order = np.argsort(thresholds)
        thresholds = thresholds[order]
        values = values[order]

        peak = float(np.max(values))
        label = f'{method_name} (peak={peak:.3f})'

        ax.plot(
            thresholds, values,
            color=style['color'],
            marker=style['marker'],
            linestyle=style['ls'],
            linewidth=2,
            markersize=5,
            markevery=max(1, len(thresholds) // 10),
            label=label,
        )

        if with_std and f'{metric}_std' in data:
            std = np.asarray(data[f'{metric}_std'])[order]
            ax.fill_between(
                thresholds,
                values - std,
                values + std,
                color=style['color'],
                alpha=0.15,
            )

    ylabel = METRIC_LABELS.get(metric, metric.upper())
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metric_vs_threshold_grid(
    threshold_data_per_condition: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    metric: str = 'f1',
    titles: Optional[List[str]] = None,
    suptitle: str = '',
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cols: int = 2,
    with_std: bool = False,
) -> None:
    """
    Plot a 2×N grid of metric-vs-threshold charts (one per condition),
    mirroring the 2×2 grid in MTSCrack Fig. 11-12 and the
    single charts in Fig. 14-16.

    Args:
        threshold_data_per_condition: dict of condition_name ->
            threshold_data (same format as plot_metric_vs_threshold).
        metric:     Metric to plot ('f1', 'iou', 'precision', 'recall').
        titles:     Optional list of subplot titles; defaults to condition names.
        suptitle:   Overall figure title.
        save_path:  Save path.
        figsize:    Figure size; auto-computed if None.
        cols:       Number of columns in the grid.
        with_std:   Draw ±std bands.
    """
    conditions = list(threshold_data_per_condition.keys())
    n = len(conditions)
    rows = (n + cols - 1) // cols

    if figsize is None:
        figsize = (6 * cols, 5 * rows)

    if titles is None:
        titles = conditions

    METRIC_LABELS = {
        'f1': 'F1-Score', 'precision': 'Precision',
        'recall': 'Recall', 'iou': 'IoU', 'miou': 'mIoU',
        'pa': 'PA', 'mpa': 'MPA',
    }
    STYLE_CYCLE = [
        {'color': '#2196F3', 'marker': 'o'},
        {'color': '#F44336', 'marker': 's'},
        {'color': '#4CAF50', 'marker': '^'},
        {'color': '#FF9800', 'marker': 'D'},
        {'color': '#9C27B0', 'marker': 'v'},
        {'color': '#00BCD4', 'marker': 'P'},
        {'color': '#795548', 'marker': '*'},
        {'color': '#607D8B', 'marker': 'h'},
    ]

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for cond_idx, (condition, ax) in enumerate(
        zip(conditions, axes.flatten())
    ):
        threshold_data = threshold_data_per_condition[condition]
        sorted_methods = sorted(
            threshold_data.items(),
            key=lambda kv: float(np.max(kv[1].get(metric, [0]))),
            reverse=True,
        )

        for idx, (method_name, data) in enumerate(sorted_methods):
            thresholds = np.asarray(data['thresholds'])
            values = np.asarray(data.get(metric, np.zeros_like(thresholds)))
            style = STYLE_CYCLE[idx % len(STYLE_CYCLE)]
            order = np.argsort(thresholds)
            thresholds, values = thresholds[order], values[order]

            peak = float(np.max(values))
            ax.plot(
                thresholds, values,
                color=style['color'], marker=style['marker'],
                linewidth=2, markersize=5,
                markevery=max(1, len(thresholds) // 10),
                label=f'{method_name} (peak={peak:.3f})',
            )
            if with_std and f'{metric}_std' in data:
                std = np.asarray(data[f'{metric}_std'])[order]
                ax.fill_between(
                    thresholds, values - std, values + std,
                    color=style['color'], alpha=0.15,
                )

        ylabel = METRIC_LABELS.get(metric, metric.upper())
        ax.set_xlabel('Decision Threshold', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(titles[cond_idx], fontsize=12)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8, framealpha=0.8)

    # Hide unused subplots
    for ax in axes.flatten()[n:]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.01)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_radar_chart(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = 'Multi-Metric Comparison',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot radar chart for multi-metric comparison.

    Args:
        results: Dictionary of method_name -> metrics
        metrics: List of metrics to include
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if metrics is None:
        metrics = ['precision', 'recall', 'f1', 'dice', 'miou']

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for i, (method_name, method_metrics) in enumerate(results.items()):
        values = [method_metrics.get(m, 0) for m in metrics]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=method_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
