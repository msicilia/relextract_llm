"""
Relation-type confusion analysis.

For each dataset with more than 2 relation types, reads the per-experiment
prediction JSONL files (results/predictions/*.jsonl) and builds a confusion
matrix over relation types.

Confusion is defined at the entity-pair level:
  - entity pair in gold AND prediction with SAME type    → TP (correct)
  - entity pair in gold AND prediction with DIFFERENT type → type confusion
  - entity pair in gold but ABSENT from prediction        → MISSED
  - entity pair in prediction but ABSENT from gold        → HALLUCINATED

Only datasets with > 2 relation types are shown (binary datasets yield
a trivial 2×2 matrix that adds little information).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from relextract_llm.util import DATASET_RELATION_TYPES, _canonical_ent

PREDICTIONS_DIR = Path(__file__).resolve().parents[2] / "results" / "predictions"
SLUG_RE = re.compile(
    r"^(?P<dataset>.+?)__(?P<model>.+?)__t(?P<temp>.+?)__(?P<mode>.+?)__(?P<mk>.+?)__"
    r"(?P<backend>.+?)__(?P<cot>cot|nocot)"
    r"(?:__(?P<prior>prior|noprior))?"
    r"(?:__(?P<infmode>[^_]+(?:_[^_]+)*?))?"
    r"(?:__(?P<conf>conf|noconf))?"
    r"\.jsonl$"
)

# Sentinel labels used in the confusion matrix for non-predictions
MISSED       = "<MISSED>"
HALLUCINATED = "<HALLUCINATED>"


def _entity_key(rel: dict) -> tuple:
    """Canonical entity pair (sorted for symmetric types, raw otherwise)."""
    e1, e2 = rel["entity_1"], rel["entity_2"]
    return (min(e1, e2), max(e1, e2))


def build_confusion(pred_file: Path) -> dict[tuple[str, str], int]:
    """Return {(true_type, pred_type): count} for one experiment file."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    with open(pred_file, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gold_by_pair: dict[tuple, str] = {}
            for r in rec["ground_truth"]:
                gold_by_pair[_entity_key(r)] = r["relation_type"]

            pred_by_pair: dict[tuple, str] = {}
            for r in rec["predicted"]:
                pred_by_pair[_entity_key(r)] = r["relation_type"]

            for pair, true_type in gold_by_pair.items():
                pred_type = pred_by_pair.get(pair, MISSED)
                counts[(true_type, pred_type)] += 1

            for pair, pred_type in pred_by_pair.items():
                if pair not in gold_by_pair:
                    counts[(HALLUCINATED, pred_type)] += 1

    return dict(counts)


def merge_confusions(per_file: list[dict]) -> dict[tuple[str, str], int]:
    merged: dict[tuple[str, str], int] = defaultdict(int)
    for c in per_file:
        for k, v in c.items():
            merged[k] += v
    return dict(merged)


def print_confusion_matrix(dataset: str, counts: dict[tuple[str, str], int]) -> None:
    all_true  = sorted({t for t, _ in counts})
    all_pred  = sorted({p for _, p in counts})
    all_labels = sorted(set(all_true) | set(all_pred))

    col_w = max(len(l) for l in all_labels) + 2
    row_label_w = col_w

    # header
    header = f"{'':>{row_label_w}}" + "".join(f"{l:>{col_w}}" for l in all_labels)
    print(header)
    print("-" * len(header))

    for true_type in all_labels:
        row = f"{true_type:>{row_label_w}}"
        for pred_type in all_labels:
            v = counts.get((true_type, pred_type), 0)
            row += f"{v if v else '':>{col_w}}"
        print(row)


def top_confusions(counts: dict[tuple[str, str], int], n: int = 10) -> list[tuple[int, str, str]]:
    """Return top-n (true, pred) pairs that are NOT correct matches, sorted by count desc."""
    errors = [(v, t, p) for (t, p), v in counts.items() if t != p]
    return sorted(errors, reverse=True)[:n]


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main() -> None:
    if not PREDICTIONS_DIR.exists():
        print(f"No predictions directory found at {PREDICTIONS_DIR}")
        print("Run experiments first to generate per-example prediction logs.")
        return

    files = sorted(PREDICTIONS_DIR.glob("*.jsonl"))
    if not files:
        print("No prediction files found.")
        return

    # Group files by dataset
    by_dataset: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        m = SLUG_RE.match(f.name)
        if m:
            by_dataset[m.group("dataset")].append(f)

    multi_type_datasets = [
        ds for ds in by_dataset
        if ds in DATASET_RELATION_TYPES and len(DATASET_RELATION_TYPES[ds]) > 2
    ]

    if not multi_type_datasets:
        print("No datasets with more than 2 relation types found in predictions.")
        return

    for dataset in sorted(multi_type_datasets):
        n_types = len(DATASET_RELATION_TYPES[dataset])
        pred_files = by_dataset[dataset]
        section(f"{dataset}  ({n_types} relation types, {len(pred_files)} experiment configs)")

        per_file = [build_confusion(pf) for pf in pred_files]
        counts = merge_confusions(per_file)

        total = sum(counts.values())
        correct = sum(v for (t, p), v in counts.items() if t == p)
        print(f"  Total relation slots : {total}")
        print(f"  Correct (type match) : {correct}  ({100*correct/total:.1f}%)")
        print(f"  Errors               : {total - correct}")

        print("\n  Top confusions (true → predicted):")
        for count, true_t, pred_t in top_confusions(counts, n=15):
            print(f"    {true_t:>35}  →  {pred_t:<35}  {count:>5}x")

        print("\n  Confusion matrix (rows=true, cols=predicted):")
        print_confusion_matrix(dataset, counts)


if __name__ == "__main__":
    main()
