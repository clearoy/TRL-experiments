# VCBench benchmark for PolicyInduction

Founder-success prediction on VCBench: 4,500 public founders (train) and 4,500
private founders (test), 9% positive. Always-NO scores 91.0% accuracy, so
precision and F0.5 are the metrics that matter.

Same layout as [`compas/`](../compas/) and [`cmv/`](../cmv/).

## Status

- [x] `prepare_data.py` — public → train, private → test
- [x] `baselines.py` — traditional-ML baselines on the same split
- [x] `run_policy_induction.py` — the TRL algorithm under test
- [x] `compare.py` — comparison table

## Results

_Not run yet._ Earlier PolicyInduction runs (library repo,
`old_vcbench_evaluation/`) reached test F0.5 of 0.17–0.25.
