# Historical aggregate results

`concrete400_comparison.csv` is copied from the research project's
`Code_2/results/comparison/crack_segment_data/all_results.csv`, without changing
the numerical cells. Metrics use fractions, not percentages; FPS/parameter/FLOP
columns retain the source units. This file is a historical aggregate, not output
of the public release tests. `*_std` must not be interpreted as repeated-training
uncertainty; per-image and timing variability require their original protocols.

There is at least one difference from the manuscript: UPerNet mIoU is 0.8952 in
this CSV versus 89.50% in the manuscript table. Neither value has been silently
overwritten. Reconcile the generating checkpoints, predictions and aggregation
before using the table as evidence of exact manuscript reproduction.

The README's OGDNet summary is transcribed from the manuscript. The earlier
Özgenel comparison CSV was not readable from the cloud-backed source during
release preparation and is not represented as an independently recovered result.
