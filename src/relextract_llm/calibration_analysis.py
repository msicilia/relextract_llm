"""
Confidence calibration analysis.

Reads prediction JSONL files that contain a non-empty "confidences" field
(produced when use_confidence=True or inference_mode=self_consistency).

For each such file computes:
  - Expected Calibration Error (ECE) with 10 equal-width bins
  - Maximum Calibration Error (MCE)
  - Average confidence vs actual accuracy
  - Per-bin breakdown

Usage:
    python -m relextract_llm.calibration_analysis
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from relextract_llm.util import RelationSpec, _canonical

PREDICTIONS_DIR = Path(__file__).resolve().parents[2] / "results" / "predictions"
N_BINS = 10

SLUG_RE = re.compile(
    r"^(?P<dataset>.+?)__(?P<model>.+?)__t(?P<temp>.+?)__(?P<mode>.+?)__(?P<mk>.+?)__"
    r"(?P<backend>.+?)__(?P<cot>cot|nocot)"
    r"(?:__(?P<prior>prior|noprior))?"
    r"(?:__(?P<infmode>[^_]+(?:_[^_]+)*?))?"
    r"(?:__(?P<conf>conf|noconf))?"
    r"\.jsonl$"
)


def _is_correct(ground_truth: list[RelationSpec], predicted: list[RelationSpec]) -> bool:
    truth = {_canonical(r) for r in ground_truth}
    pred  = {_canonical(r) for r in predicted}
    return truth == pred


def _load_confidence_records(path: Path) -> list[tuple[float, bool]]:
    """Return (confidence, is_correct) pairs from one JSONL file."""
    pairs: list[tuple[float, bool]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("failed"):
                continue
            confs = rec.get("confidences", [])
            if not confs:
                continue
            gt   = [RelationSpec(**r) for r in rec["ground_truth"]]
            pred = [RelationSpec(**r) for r in rec["predicted"]]
            correct = _is_correct(gt, pred)
            # Use the first (and usually only) confidence value
            pairs.append((float(confs[0]), correct))
    return pairs


def compute_ece(pairs: list[tuple[float, bool]], n_bins: int = N_BINS) -> dict:
    """Compute ECE, MCE, and per-bin stats."""
    bins: dict[int, list] = defaultdict(list)
    for conf, correct in pairs:
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, correct))

    total = len(pairs)
    ece = 0.0
    mce = 0.0
    bin_stats = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        items = bins.get(b, [])
        if not items:
            bin_stats.append({"range": f"[{lo:.1f},{hi:.1f})", "n": 0,
                               "avg_conf": None, "accuracy": None, "gap": None})
            continue
        avg_conf = sum(c for c, _ in items) / len(items)
        accuracy = sum(1 for _, ok in items if ok) / len(items)
        gap      = abs(avg_conf - accuracy)
        weight   = len(items) / total
        ece     += weight * gap
        mce      = max(mce, gap)
        bin_stats.append({"range": f"[{lo:.1f},{hi:.1f})",
                           "n": len(items), "avg_conf": avg_conf,
                           "accuracy": accuracy, "gap": gap})

    avg_conf_overall = sum(c for c, _ in pairs) / total
    accuracy_overall = sum(1 for _, ok in pairs if ok) / total

    return {
        "n": total,
        "ece": ece,
        "mce": mce,
        "avg_confidence": avg_conf_overall,
        "accuracy": accuracy_overall,
        "bins": bin_stats,
    }


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def main() -> None:
    if not PREDICTIONS_DIR.exists():
        print("No predictions directory. Run experiments first.")
        return

    files = sorted(PREDICTIONS_DIR.glob("*.jsonl"))
    calibration_files = []
    for fpath in files:
        # Quick check: scan first few lines for non-empty confidences
        with open(fpath, encoding="utf-8") as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                if rec.get("confidences"):
                    calibration_files.append(fpath)
                    break
                if i > 10:
                    break

    if not calibration_files:
        print("No prediction files with confidence scores found.")
        print("Run experiments with use_confidence=True or inference_mode=self_consistency.")
        return

    print(f"Found {len(calibration_files)} file(s) with confidence data.\n")
    header = (f"{'Experiment':<55}  {'N':>5}  {'Acc':>6}  {'AvgConf':>7}  "
              f"{'ECE':>6}  {'MCE':>6}")
    print(header)
    print("-" * len(header))

    for fpath in calibration_files:
        pairs = _load_confidence_records(fpath)
        if len(pairs) < 5:
            continue
        stats = compute_ece(pairs)
        slug = fpath.stem[:55]
        print(f"{slug:<55}  {stats['n']:>5}  {stats['accuracy']:>6.3f}  "
              f"{stats['avg_confidence']:>7.3f}  {stats['ece']:>6.4f}  {stats['mce']:>6.4f}")

    # Detailed breakdown for each file
    for fpath in calibration_files:
        pairs = _load_confidence_records(fpath)
        if len(pairs) < 5:
            continue
        stats = compute_ece(pairs)
        section(fpath.stem)
        print(f"  Examples : {stats['n']}")
        print(f"  Accuracy : {stats['accuracy']:.4f}")
        print(f"  Avg conf : {stats['avg_confidence']:.4f}")
        print(f"  ECE      : {stats['ece']:.4f}  (lower is better)")
        print(f"  MCE      : {stats['mce']:.4f}  (worst-bin gap)")
        print()
        print(f"  {'Bin range':<12} {'N':>5}  {'Avg conf':>8}  {'Accuracy':>8}  {'Gap':>6}")
        print(f"  {'-'*46}")
        for b in stats["bins"]:
            if b["n"] == 0:
                continue
            bar = "█" * round(b["gap"] * 20)
            print(f"  {b['range']:<12} {b['n']:>5}  {b['avg_conf']:>8.3f}  "
                  f"{b['accuracy']:>8.3f}  {b['gap']:>6.3f}  {bar}")


if __name__ == "__main__":
    main()
