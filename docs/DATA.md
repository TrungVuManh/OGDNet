# Data preparation

Obtain each dataset from its provider and retain its attribution/license.
The Özgenel record is https://data.mendeley.com/datasets/jwsn7tfbrp/1 (CC BY 4.0).
The Concrete400 DOI listed in the manuscript is 10.5281/zenodo.21193945; this
release does not assert that the record is currently downloadable.

Use RGB images in `Image/` and grayscale binary masks in `Label/`. Match file
stems exactly, e.g. `Image/001.jpg` and `Label/001.png`. Supported extensions are
PNG/JPG/JPEG (case-insensitive). Masks must have the same spatial dimensions as
their images. Multiclass palette masks need conversion to binary before use.

The loader treats values greater than 127 as foreground for 8-bit masks and
values greater than 0 for 0/1 masks. It rejects mismatched dimensions instead of
silently resizing annotations.

The split manifest has this structure:

```json
{"train": ["001", "002"], "val": ["003"], "test": ["004"]}
```

`test` is optional. Lists must be nonoverlapping and selected lists nonempty.
IDs are basenames without extension, not filesystem paths.

`scripts/prepare_split.py` generates a seeded 80/20 train/validation split and
records image SHA-256 hashes. `--groups` keeps every acquisition group together;
with groups, the fraction applies to groups and image counts may differ from 80/20.
Group CSV example:

```csv
image_id,group_id
001,building_a_session_1
002,building_a_session_1
003,building_b_session_1
```

Byte-identical images crossing subsets are rejected. This does not detect near
duplicates, adjacent crops, or repeated views; group these using acquisition
metadata. The optional `--order sorted` reproduces a sorted 80/20 splitting rule,
but is not evidence that a particular historical experiment used that rule.

Exact historical split files, original image archives, and manuscript checkpoints
are not included. Do not describe newly generated files as recovered original splits.
