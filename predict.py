"""Predict an RGB image and save a 0/255 PNG mask at its original size."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from ogdnet import load_model
from utils.datasets import get_val_transforms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new filename")
    if args.output.suffix.lower() != ".png" or not 0 <= args.threshold <= 1:
        parser.error("Output must be PNG and threshold must be in [0, 1]")
    model, config = load_model(args.checkpoint, args.device)
    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode input image")
    tensor = get_val_transforms(config)(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))["image"]
    with torch.inference_mode():
        probabilities = model(tensor.unsqueeze(0).to(args.device))[0, 0].cpu().numpy()
    probabilities = cv2.resize(probabilities, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = (probabilities > args.threshold).astype(np.uint8) * 255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise OSError("PNG encoding failed")
    encoded.tofile(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
