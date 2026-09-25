"""
Cross-model ensemble analysis.

Groups prediction JSONL files by everything EXCEPT the model, then majority-votes
the predicted relation triple for each example.  Reports ensemble F1 versus the
best and average individual-model F1 for each config group.

Usage:
    python -m relextract_llm.ensemble_analysis
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Iterator

from relextract_llm.util import RelationSpec, _canonical, _canonical_ent, MatchCounter

PREDICTIONS_DIR = Path(__file__).resolve().parents[2] / "results" / "predictions"

# Experiment slug; the prior, inference-mode and confidence components are optional.
SLUG_RE = re.compile(
    r"^(?P<dataset>.+?)__(?P<model>.+?)__t(?P<temp>.+?)__(?P<mode>.+?)__(?P<mk>.+?)__"
    r"(?P<backend>.+?)__(?P<cot>cot|nocot)"
    r"(?:__(?P<prior>prior|noprior))?"
    r"(?:__(?P<infmode>[^_]+(?:_[^_]+)*?))?"
    r"(?:__(?P<conf>conf|noconf))?"
    r"\.jsonl$"
)

# Config key = everything that must be the same across models in a group
_GROUP_FIELDS = ("dataset", "temp", "mode", "mk", "backend", "cot", "prior", "infmode", "conf")


def _parse_slug(filename: str) -> dict | None:
    m = SLUG_RE.match(filename)
    return m.groupdict() if m else None


def _load_predictions(path: Path) -> dict[str, dict]:
    """Return {example_id: {"ground_truth": [...], "predicted": [...]}} for one file."""
    records: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("failed"):
                continue
            records[rec["id"]] = {
                "ground_truth": [RelationSpec(**r) for r in rec["ground_truth"]],
                "predicted":    [RelationSpec(**r) for r in rec["predicted"]],
            }
    return records


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _compute_f1(predictions: dict[str, dict]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for rec in predictions.values():
        truth = {_canonical(r) for r in rec["ground_truth"]}
        pred  = {_canonical(r) for r in rec["predicted"]}
        tp += len(truth & pred)
        fp += len(pred  - truth)
        fn += len(truth - pred)
    return _prf(tp, fp, fn)


def majority_vote(
    per_model: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    """For each example id seen in any model's predictions, majority-vote the triple.

    Only examples with predictions from ≥2 models are included.
    """
    # Collect all example ids
    all_ids: set[str] = set()
    for recs in per_model.values():
        all_ids.update(recs.keys())

    ensemble: dict[str, dict] = {}
    for eid in all_ids:
        votes: Counter = Counter()
        ground_truth = None
        n_models_with_pred = 0
        for model_recs in per_model.values():
            rec = model_recs.get(eid)
            if rec is None:
                continue
            n_models_with_pred += 1
            if ground_truth is None:
                ground_truth = rec["ground_truth"]
            for r in rec["predicted"]:
                votes[_canonical(r)] += 1

        if n_models_with_pred < 2 or not votes:
            continue

        # Take the triple with the most votes; ties broken by canonical order
        top_triple = votes.most_common(1)[0][0]
        e1, e2, rt = top_triple
        ensemble[eid] = {
            "ground_truth": ground_truth,
            "predicted":    [RelationSpec(entity_1=e1, entity_2=e2, relation_type=rt)],
        }
    return ensemble


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def main() -> None:
    if not PREDICTIONS_DIR.exists():
        print("No predictions directory found. Run experiments first.")
        return

    files = sorted(PREDICTIONS_DIR.glob("*.jsonl"))
    if not files:
        print("No prediction files found.")
        return

    # Group files by config-except-model
    groups: dict[tuple, dict[str, Path]] = defaultdict(dict)
    for fpath in files:
        info = _parse_slug(fpath.name)
        if info is None:
            continue
        group_key = tuple(info.get(f) or "" for f in _GROUP_FIELDS)
        model_name = info["model"]
        groups[group_key][model_name] = fpath

    # Only analyse groups with ≥2 models
    multi_model_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not multi_model_groups:
        print("No groups with ≥2 models found — run more models first.")
        return

    print(f"Found {len(multi_model_groups)} config groups with ≥2 models.\n")
    header = f"{'Config':<60} {'Models':>6}  {'Best-ind F1':>11}  {'Ensemble F1':>11}  {'Delta':>7}"
    print(header)
    print("-" * len(header))

    for group_key, model_files in sorted(multi_model_groups.items()):
        label = "  ".join(f for f in group_key if f)

        # Load per-model predictions
        per_model: dict[str, dict] = {}
        for model_name, fpath in model_files.items():
            per_model[model_name] = _load_predictions(fpath)

        # Individual F1 per model
        ind_f1s = {m: _compute_f1(recs)[2] for m, recs in per_model.items()}
        best_ind = max(ind_f1s.values())

        # Ensemble
        ensemble_preds = majority_vote(per_model)
        if not ensemble_preds:
            continue
        _, _, ens_f1 = _compute_f1(ensemble_preds)

        delta = ens_f1 - best_ind
        print(f"{label:<60} {len(model_files):>6}  {best_ind:>11.4f}  {ens_f1:>11.4f}  {delta:>+7.4f}")

    print()
    print("Delta = Ensemble F1 − Best-individual F1  (positive = ensemble wins)")


if __name__ == "__main__":
    main()
