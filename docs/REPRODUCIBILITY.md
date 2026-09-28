# Reproducibility and release provenance

## What is included

- Ten model implementation files copied from the project's `Code_2` source tree.
  Original SHA-256 values are in [source_manifest.json](source_manifest.json).
- Shared focal/Dice/morphological-boundary loss and segmentation metrics.
- Portable configurations, explicit split creation/validation, public training,
  evaluation and prediction tools, and synthetic-data tests.

The original research tree also contained earlier code copies and unavailable
cloud-backed artifacts. This is a curated code release, not a complete archive
of the training machine or proof that one particular source revision produced
every manuscript table.

## Changes in the public tooling

Model parameter names/forward implementations were retained. The public CLI and
data pipeline require explicit manifests and reject overlapping IDs or mismatched
image/mask sizes. Historical loaders could fall back to a sorted 80/20 split and
resize masks silently. Data and output paths are now portable. Checkpoint loading
uses `weights_only=True` and strict state-dictionary validation. Checkpoints from
the public trainer contain weights/configuration/history, not optimizer resume
state; restart training in a new output directory rather than treating a weights
checkpoint as an exact resume.

CPU autocast is disabled; CUDA training uses mixed precision. Public training
initializes the Albumentations random generator explicitly. Default configs run
100 epochs with patience 100 rather than the older patience 50. Each run saves
its split, configuration, environment summary, and epoch metrics. These changes
can affect reproducibility and are not represented as retraining the manuscript.

## Actual preprocessing

Inputs are RGB and normalized with ImageNet mean/std. Validation uses resize and
normalization only. Input size is specified as `[height, width]`.

The inherited training transform contains horizontal/vertical flips, random
90-degree rotation, resized crop (scale 0.75–1.0, ratio 0.9–1.1), brightness/
contrast or gamma adjustment, Gaussian noise, Gaussian blur, elastic transform,
and grid distortion. The manuscript's shorter augmentation description does
not list every inherited transform; this needs reconciliation against the
historical logs before claiming exact paper reproduction.

The orientation head uses `F.normalize(..., eps=1e-8)` followed by half `atan2`.
Normalization does not make the zero-vector angle well-defined or remove the
angle branch cut. The residual branch is zero-initialized and bounded with tanh;
the rotation scale starts at sigmoid(-4), approximately 0.018, and need not
increase monotonically. The square grid is rotated, not elongated by rotation.

## Metrics and scientific limits

`evaluate.py` accumulates TP/FP/FN/TN over the selected subset and computes global
precision, recall, F1, crack IoU, mean IoU, pixel accuracy and mean pixel accuracy.
Per-image metrics are exported separately. Across-image standard deviation is
not variation across training seeds. IoU/F1 do not quantify topology/connectivity
or calibrated crack-width accuracy.

Reported in-domain scores used validation-based checkpoint selection. A separate
test set and repeated seeds are needed for stronger comparative inference. No
significance tests or uncertainty estimates have been manufactured for this release.

## Missing artifacts

Exact historical split IDs, checkpoints, per-run environment lockfiles, original
training predictions and all underlying curve values remain to be archived.
Public data availability should name those artifacts when they become available.
Do not replace that statement with “all data are in this repository.”

## Validation

The repository provides CPU tests for dataset integrity, grouped splitting,
model outputs, OGDNet ablation gradients, checkpoint reload, and metric arithmetic.
The synthetic smoke test exercises the command-line pipeline end to end. Full
dataset training, GPU determinism, published accuracy, and throughput have not
been revalidated during release preparation.
