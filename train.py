"""Train OGDNet or a comparison model using an explicit data split."""
import argparse
import json
import platform
import shutil
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from ogdnet import build_model, load_config
from utils.common.seed import set_seed
from utils.datasets import CrackDataset, get_train_transforms, get_val_transforms
from utils.losses import build_criterion
from utils.metrics.segmentation import SegmentationMetrics

ABLATIONS = {
    "full": {},
    "no_orientation": {"use_orientation": False, "use_base_offset": False},
    "no_residual": {"use_residual_offset": False},
    "no_bound": {"bound_residual": False},
    "no_warm_start": {"warm_start_base": False},
    "plain_decoder": {"use_orientation": False, "use_deformable": False,
                      "use_base_offset": False, "use_residual_offset": False},
}


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, clip=1.0):
    training = optimizer is not None
    model.train(training)
    metrics, losses = SegmentationMetrics(), []
    for batch in tqdm(loader, desc="Train" if training else "Validation"):
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device.type, enabled=training and device.type == "cuda"):
                predictions = model(images)
            loss = criterion(predictions, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
        metrics.update(predictions.detach(), masks)
        losses.append(loss.item())
    if not losses:
        raise ValueError("Empty dataloader; check split and batch size")
    return {"loss": sum(losses) / len(losses), **metrics.compute()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="methods/ogdnet/config.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.ablation != "full" and config["method"]["name"] != "ogdnet":
        parser.error("Ablations are defined only for OGDNet")
    if config["method"]["name"] == "ogdnet":
        config["model"]["ablation"] = ABLATIONS[args.ablation]
    if args.no_pretrained:
        config["model"]["pretrained"] = False
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    epochs = config["training"]["epochs"]
    if epochs < 1 or args.workers < 0:
        parser.error("Epochs must be positive and workers non-negative")
    output = args.output or Path(config["experiment"]["output_dir"])
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; use a new directory for each run")
    set_seed(config["experiment"]["seed"])
    device = torch.device(args.device)
    transform = get_train_transforms(config)
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(config["experiment"]["seed"])
    training = CrackDataset(config, "train", transform)
    validation = CrackDataset(config, "val", get_val_transforms(config))
    batch_size = config["training"]["batch_size"]
    if len(training) < batch_size:
        parser.error("Training split is smaller than batch_size")
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(validation, batch_size=1, num_workers=args.workers)
    model = build_model(config).to(device)
    criterion = build_criterion(config)
    encoder = getattr(model, "encoder", getattr(model, "backbone", None))
    encoder_parameters = list(encoder.parameters()) if encoder is not None else []
    encoder_ids = {id(p) for p in encoder_parameters}
    base_lr = config["training"]["learning_rate"]
    groups = [{"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": base_lr}]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": base_lr * config["training"].get("encoder_lr_scale", 0.1)})
    optimizer = torch.optim.AdamW(groups, weight_decay=config["training"]["weight_decay"])
    target_lrs = [group["lr"] for group in optimizer.param_groups]
    warmup = min(config["training"].get("warmup_epochs", 10), max(0, epochs - 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup),
                                                          eta_min=config["training"].get("min_lr", 1e-6))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    shutil.copyfile(config["data"]["split_file"], output / "split.json")
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "cuda": torch.version.cuda, "device": str(device), "seed": config["experiment"]["seed"]}
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    history, best, stale = [], -1.0, 0
    patience = config["experiment"].get("early_stopping_patience", epochs)
    for epoch in range(epochs):
        if epoch < warmup:
            for group, target in zip(optimizer.param_groups, target_lrs):
                group["lr"] = target * (epoch + 1) / warmup
        train = run_epoch(model, train_loader, criterion, device, optimizer, scaler,
                          config["training"].get("gradient_clip", 1.0))
        val = run_epoch(model, val_loader, criterion, device)
        if epoch >= warmup:
            scheduler.step()
        history.append({"epoch": epoch + 1, "train": train, "val": val})
        improved = val["iou"] > best
        best, stale = max(best, val["iou"]), 0 if improved else stale + 1
        checkpoint = {"model_state_dict": model.state_dict(), "config": config,
                      "epoch": epoch + 1, "best_metric": best, "history": history}
        if improved:
            torch.save(checkpoint, output / "best_model.pth")
        torch.save(checkpoint, output / "last_model.pth")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch + 1}/{epochs}: validation IoU={val['iou']:.6f}; best={best:.6f}")
        if patience > 0 and stale >= patience:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
