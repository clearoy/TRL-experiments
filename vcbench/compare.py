"""Print the TF-IDF baseline vs PolicyInduction comparison table for VCBench.

Reads results/baselines.json and results/policy_induction_*.json.

Run:
    python vcbench/compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
COLS = ("f0.5", "precision", "recall", "f1", "accuracy")


def fmt(row: dict) -> str:
    cells = " | ".join(f"{row.get(c, float('nan')):.4f}" for c in COLS)
    auc = f"{row['roc_auc']:.4f}" if "roc_auc" in row else "--"
    return f" {cells} | {auc} |"


def main() -> None:
    base_path = RESULTS_DIR / "baselines.json"
    if not base_path.exists():
        raise SystemExit("results/baselines.json not found. Run baselines.py first.")
    base = json.loads(base_path.read_text())

    lines = [
        f"VCBench -- test set n={base['n_test']}, "
        f"{base['positive_rate_test']:.1%} positive",
        "",
        "| Model | Train rows | F0.5 | Precision | Recall | F1 | Accuracy | ROC-AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for regime, block in base["regimes"].items():
        for name, m in block.items():
            if name == "n_train":
                continue
            lines.append(f"| {name} tfidf ({regime}) | {block['n_train']} |{fmt(m)}")

    pi_paths = sorted(RESULTS_DIR.glob("policy_induction_*.json"))
    if pi_paths:
        for p in pi_paths:
            pi = json.loads(p.read_text())
            run = p.stem.replace("policy_induction_", "")
            label = f"**PolicyInduction** ({run})"
            if pi.get("n_scored", 0) != pi.get("n_test", 0):
                label += f" [{pi['n_scored']}/{pi['n_test']} scored]"
            lines.append(f"| {label} | {pi['n_train']} |{fmt(pi)}")
    else:
        lines.append("| *PolicyInduction* | -- | *not run yet* | | | | | |")

    lines.append(
        f"| majority class (all NO) | -- | -- | -- | -- | -- | "
        f"{base['majority_class_accuracy']:.4f} | -- |"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
