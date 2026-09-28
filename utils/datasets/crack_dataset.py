"""RGB concrete images, binary crack masks, and explicit split manifests."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, config, split="train", transform=None):
        data = config.get("data", config)
        root = Path(data["dataset_path"]).expanduser()
        self.image_dir = root / data.get("image_folder", "Image")
        self.mask_dir = root / data.get("mask_folder", "Label")
        self.transform = transform
        manifest = Path(data["split_file"])
        if not manifest.is_file():
            raise FileNotFoundError(f"Split missing: {manifest}. Run scripts/prepare_split.py first.")
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        seen = set()
        for subset in ("train", "val", "test"):
            ids = entries.get(subset, [])
            if not isinstance(ids, list) or any(not isinstance(key, str) or not key or '/' in key or '\\' in key or key in ('.', '..') for key in ids):
                raise ValueError(f"{subset} must contain image stems, not paths")
            if len(ids) != len(set(ids)) or seen.intersection(ids):
                raise ValueError("Duplicate/overlapping image IDs in split manifest")
            seen.update(ids)
        self.image_ids = entries.get(split, [])
        if not self.image_ids:
            raise ValueError(f"Split {split!r} is missing or empty")
        self.images, self.masks = self._index(self.image_dir), self._index(self.mask_dir)
        missing = [key for key in self.image_ids if key not in self.images or key not in self.masks]
        if missing:
            raise FileNotFoundError(f"Missing image/mask pairs: {missing[:10]}")

    @staticmethod
    def _index(folder):
        result = {}
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                if path.stem in result:
                    raise ValueError(f"Duplicate image stem: {path.stem}")
                result[path.stem] = path
        return result

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        key = self.image_ids[index]
        image = cv2.imdecode(np.fromfile(self.images[key], dtype=np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(self.masks[key], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Cannot decode image/mask for {key}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mask.shape != image.shape[:2]:
            raise ValueError(f"Image/mask size mismatch for {key}; align annotations explicitly")
        mask = (mask > (127 if mask.max() > 1 else 0)).astype(np.float32)
        if self.transform:
            result = self.transform(image=image, mask=mask)
            image, mask = result["image"], result["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255
            mask = torch.from_numpy(mask)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return {"image": image, "mask": mask, "image_id": key}
