"""
Logging utilities for training.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional


class Logger:
    """
    Logger for training progress and metrics.
    """

    def __init__(self, log_file: Optional[str] = None, name: str = 'SCAFormer'):
        """
        Initialize logger.

        Args:
            log_file: Path to log file (optional)
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)

        self.log_file = log_file

    def log(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def log_metrics(self, epoch: int, metrics: dict, prefix: str = '') -> None:
        """
        Log metrics in a formatted way.

        Args:
            epoch: Current epoch
            metrics: Dictionary of metric names and values
            prefix: Prefix for metric names (e.g., 'train', 'val')
        """
        metric_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        if prefix:
            self.logger.info(f'Epoch {epoch} [{prefix}] {metric_str}')
        else:
            self.logger.info(f'Epoch {epoch} | {metric_str}')

    def log_config(self, config: dict) -> None:
        """Log configuration."""
        self.logger.info('=' * 60)
        self.logger.info('Configuration:')
        self._log_dict(config, indent=2)
        self.logger.info('=' * 60)

    def _log_dict(self, d: dict, indent: int = 0) -> None:
        """Recursively log dictionary."""
        for key, value in d.items():
            if isinstance(value, dict):
                self.logger.info(' ' * indent + f'{key}:')
                self._log_dict(value, indent + 2)
            else:
                self.logger.info(' ' * indent + f'{key}: {value}')
