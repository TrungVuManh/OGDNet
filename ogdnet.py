"""Model/configuration entry points shared by the public command-line tools."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path

import torch
import yaml

METHODS = (
    "ogdnet", "unet", "resnet", "deeplabv3plus", "crackformer",
    "swin_unet", "segformer", "segnext", "convnext_unet", "upernet",
)


def load_config(path):
    """All relative paths in a config are relative to the repository working directory."""
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["method"]["name"] not in METHODS:
        raise ValueError(f"Unknown method; choose one of {METHODS}")
    return config


def build_model(config, pretrained=None):
    config = deepcopy(config)
    if pretrained is not None:
        config.setdefault("model", {})["pretrained"] = pretrained
    name = config["method"]["name"]
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}")
    return import_module(f"methods.{name}.model").build_model(config)


def load_model(checkpoint_path, device="cpu"):
    """Load a weights-only-compatible project checkpoint with strict key validation."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = build_model(config, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), config
