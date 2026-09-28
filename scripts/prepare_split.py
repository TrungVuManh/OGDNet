"""Create an explicit train/validation manifest; optionally keep groups together."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

EXTENSIONS = {".png", ".jpg", ".jpeg"}


def index_images(folder):
    result = {}
    for path in sorted(Path(folder).iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in result:
                raise ValueError(f"Duplicate image ID: {path.stem}")
            result[path.stem] = path
    if not result:
        raise ValueError(f"No images found in {folder}")
    return result


def create_split(root, seed=42, fraction=0.8, groups=None, order="random"):
    root = Path(root)
    images, masks = index_images(root / "Image"), index_images(root / "Label")
    if images.keys() != masks.keys():
        raise ValueError("Image and Label directories must contain matching image IDs")
    if not 0 < fraction < 1:
        raise ValueError("Train fraction must be strictly between 0 and 1")
    if groups:
        with open(groups, newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len({row["image_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate image_id in group manifest")
        mapping = {row["image_id"]: row["group_id"] for row in rows}
        if mapping.keys() != images.keys() or any(not value for value in mapping.values()):
            raise ValueError("Group CSV must map every image ID to a nonempty group_id")
    else:
        mapping = {key: key for key in images}
    units = sorted(set(mapping.values()))
    if len(units) < 2:
        raise ValueError("At least two independent images/groups are required")
    if order == "random":
        random.Random(seed).shuffle(units)
    cut = max(1, min(len(units) - 1, int(len(units) * fraction)))
    train_groups = set(units[:cut])
    train = sorted(key for key in images if mapping[key] in train_groups)
    val = sorted(key for key in images if mapping[key] not in train_groups)
    # Reject byte-identical images across subsets, even with different filenames.
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in images.items()}
    if {hashes[key] for key in train} & {hashes[key] for key in val}:
        raise ValueError("Byte-identical images cross the split; group them together")
    return {"train": train, "val": val, "metadata": {
        "seed": seed, "order": order, "train_fraction": fraction,
        "grouped": bool(groups), "image_sha256": hashes,
        "note": "New split; not asserted to be the historical manuscript split.",
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--groups", type=Path, help="CSV columns: image_id,group_id")
    parser.add_argument("--order", choices=["random", "sorted"], default="random")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new filename to preserve the previous split")
    result = create_split(args.data, args.seed, args.train_fraction, args.groups, args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {len(result['train'])} train / {len(result['val'])} validation images")


if __name__ == "__main__":
    main()
