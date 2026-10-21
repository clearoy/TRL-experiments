# TRL-experiments

Benchmarks for [`think-reason-learn`](https://github.com/Vela-Research/think-reason-learn)
on public datasets with published baselines, so its algorithms can be measured
against something other than themselves.

Each dataset gets its own directory with the same four-script layout, its own
README, and results that are reproducible from the scripts alone.

## Layout

```
<dataset>/
├── README.md                  setup decisions, results, known gaps
├── prepare_data.py            download -> filter -> split
├── baselines.py               traditional-ML baselines on the same split
├── run_policy_induction.py    the TRL algorithm under test
├── compare.py                 renders the comparison table
├── data/                      (gitignored) datasets + split_meta.json
├── results/                   (gitignored) metrics + per-row predictions
└── run*/                      (gitignored) per-run model artifacts
```

| Dataset | Task | Status |
|---|---|---|
| [`compas/`](compas/) | 2-year recidivism, 6,172 rows, 45.5% positive | PolicyInduction at parity with logistic regression |
| [`cmv/`](cmv/) | ChangeMyView persuasion, 807 heldout pairs | PolicyInduction 0.6241 vs 0.5967 word count, level with an embedding baseline and with Tan et al. 2016. Result hinges on the scorer model |

## Setup

These scripts import `think_reason_learn` from a sibling checkout:

```
vela/vcbench/
├── think-reason-learn/     the library
└── experiments/            this repo
```

`run_policy_induction.py` adds the sibling repo to `sys.path`, so no install is
needed as long as that layout holds. Run scripts with the library's interpreter:

```bash
../think-reason-learn/.venv/bin/python compas/prepare_data.py
```

Alternatively, `make install` in the library repo installs it editable, after
which any interpreter with those dependencies works from any directory.

API keys are read from the library repo's `.env` (`GOOGLE_AI_API_KEY`,
`OPENAI_API_KEY`).

## Conventions

**Baselines are fitted twice** — on all training rows, and on the same subsample
the LLM method can afford. The subsample row is the valid head-to-head
comparison; comparing a 500-row method against a 4,320-row one is not.

**Both methods get identical inputs.** Structured rows go to PolicyInduction
as-is (`_render_sample()` flattens them to `column: value` lines), so any
difference is attributable to the method rather than to a preprocessing step.

**Every result records the library commit** it was produced with (`trl_commit` in
the metrics JSON). With the library and experiments in separate repos, a result
without that stamp is not reproducible.

**Nothing regenerable is tracked.** Datasets, metrics, and model artifacts are
gitignored; `data/split_meta.json` is the exception, since it records the exact
filter, split, and feature choices needed to rebuild a split.

## A caveat that applies to every LLM result here

Policy generation runs at `temperature=1.0`, and `random_state` seeds only sample
selection — not the LLM. Two runs of an *identical* configuration have been
observed to differ by ~0.08 precision and ~0.15 recall. Single runs are
indicative; ≥3 are needed before drawing a conclusion.
