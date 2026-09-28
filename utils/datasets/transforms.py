"""
Data augmentation transforms using Albumentations.
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

import warnings
warnings.filterwarnings("ignore", message="A NumPy version")
warnings.filterwarnings("ignore", message="Error fetching version info", category=UserWarning)
warnings.filterwarnings("ignore", module="albumentations")

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict


def get_train_transforms(config: Dict) -> A.Compose:
    """
    Get training augmentation pipeline.

    Args:
        config: Configuration dictionary

    Returns:
        Albumentations Compose object
    """
    aug_config = config.get('augmentation', {})
    data_config = config.get('data', config)

    input_size = data_config.get('input_size', [640, 640])
    if isinstance(input_size, int):
        input_size = [input_size, input_size]

    # Get augmentation parameters
    h_flip_prob = aug_config.get('horizontal_flip', 0.5)
    v_flip_prob = aug_config.get('vertical_flip', 0.5)
    rotation_prob = aug_config.get('random_rotation', 0.25)
    crop_scale = aug_config.get('random_crop_scale', [0.75, 1.0])
    elastic_prob = aug_config.get('elastic_transform', 0.2)
    grid_prob = aug_config.get('grid_distortion', 0.2)

    transforms_list = [
        # Horizontal flip
        A.HorizontalFlip(p=h_flip_prob),

        # Vertical flip
        A.VerticalFlip(p=v_flip_prob),

        # Random 90 degree rotations
        A.RandomRotate90(p=rotation_prob),

        # Random resized crop (MUST BE AFTER ROTATE to ensure consistent output size)
        A.RandomResizedCrop(
            size=(input_size[0], input_size[1]),
            scale=(crop_scale[0], crop_scale[1]),
            ratio=(0.9, 1.1),
            p=1.0
        ),

        # Color augmentation (helps prevent overfitting on IR images)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.3),

        # Gaussian noise (simulates sensor noise in IR images)
        A.GaussNoise(p=0.2),

        # Gaussian blur (simulates slight defocus)
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),

        # Elastic transform (important for crack-like structures)
        A.ElasticTransform(
            alpha=50,
            sigma=10,
            p=elastic_prob
        ),

        # Grid distortion (simulates surface irregularities)
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=grid_prob
        ),

        # Normalize
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        # Convert to tensor
        ToTensorV2()
    ]

    return A.Compose(transforms_list)


def get_val_transforms(config: Dict) -> A.Compose:
    """
    Get validation transform pipeline (no augmentation).

    Args:
        config: Configuration dictionary

    Returns:
        Albumentations Compose object
    """
    data_config = config.get('data', config)
    input_size = data_config.get('input_size', [640, 640])
    if isinstance(input_size, int):
        input_size = [input_size, input_size]

    transforms_list = [
        # Resize
        A.Resize(height=input_size[0], width=input_size[1]),

        # Normalize
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        # Convert to tensor
        ToTensorV2()
    ]

    return A.Compose(transforms_list)


def get_test_transforms(config: Dict) -> A.Compose:
    """
    Get test transform pipeline (same as validation).

    Args:
        config: Configuration dictionary

    Returns:
        Albumentations Compose object
    """
    return get_val_transforms(config)
