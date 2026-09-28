"""
Random seed utilities for reproducibility.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def worker_init_fn(worker_id: int, seed: int = 42) -> None:
    """
    Worker initialization function for DataLoader.

    Args:
        worker_id: Worker ID
        seed: Base random seed
    """
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
