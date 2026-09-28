"""Evaluate a checkpoint on an explicit target split, without fine-tuning."""
import argparse
import csv
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ogdnet import load_config, load_model
from utils.datasets import CrackDataset, get_val_transforms
from utils.metrics.segmentation import SegmentationMetrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Override target data config for cross-dataset evaluation")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("Threshold must be between zero and one")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty")
    model, config = load_model(args.checkpoint, args.device)
    if args.config:
        config["data"] = load_config(args.config)["data"]
    dataset = CrackDataset(config, args.split, get_val_transforms(config))
    metrics, rows = SegmentationMetrics(args.threshold), []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1):
            predictions = model(batch["image"].to(args.device))
            metrics.update(predictions, batch["mask"])
            per_image = SegmentationMetrics(args.threshold)
            per_image.update(predictions, batch["mask"])
            rows.append({"image_id": batch["image_id"][0], **per_image.compute()})
    result = {"split": args.split, "n_images": len(dataset), "threshold": args.threshold,
              "aggregation": "global pixel confusion matrix", **metrics.compute()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
