"""
BLURB-style evaluation of the stored predictions.

The complete-match P/R/F1 written by MatchCounter treats the null class
(CPR:false, DDI-false, negative, Other) as an ordinary relation type.  Because
every example carries exactly one gold relation, that score is close to
all-class accuracy and is not comparable to the positive-class micro-F1
reported for fine-tuned models on BLURB.  This script recomputes, from
results/predictions/*.jsonl:

  - positive-class micro P/R/F1: only relations whose type is not a null
    class count as TP/FP/FN (BLURB convention for ChemProt, DDI, GAD, EU-ADR;
    SemEval is scored the same way, excluding Other);
  - failures (no structured output) are scored as empty predictions, so a
    failed positive example is a false negative;
  - 95% bootstrap confidence intervals over examples;
  - majority-class, most-frequent-positive-type and stratified-random
    baselines evaluated on the same examples as each configuration.

Output: results/blurb_metrics.csv (one row per configuration) and a summary
printed to stdout.
"""

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from relextract_llm.confusion_analysis import SLUG_RE
from relextract_llm.util import NON_DIRECTIONAL_TYPES

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = ROOT / "results" / "predictions"
OUT_PATH = ROOT / "results" / "blurb_metrics.csv"

NULL_TYPES = {"CPR:false", "DDI-false", "negative", "Other"}
N_BOOT = 1000
N_RANDOM = 200
SEED = 0


def _canon(rel: dict) -> tuple:
    e1, e2, t = rel["entity_1"], rel["entity_2"], rel["relation_type"]
    if t in NON_DIRECTIONAL_TYPES and e2 < e1:
        e1, e2 = e2, e1
    return (e1, e2, t)


def _positive(rels: list) -> set:
    return {_canon(r) for r in rels if r["relation_type"] not in NULL_TYPES}


def _counts(gold: list, pred: list) -> tuple:
    g, p = _positive(gold), _positive(pred)
    return len(g & p), len(p - g), len(g - p)


def _prf(tp: int, fp: int, fn: int) -> tuple:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _f1_of(rows: list) -> float:
    tp = sum(r[0] for r in rows)
    fp = sum(r[1] for r in rows)
    fn = sum(r[2] for r in rows)
    return _prf(tp, fp, fn)[2]


def _bootstrap_ci(rows: list, rng: random.Random) -> tuple:
    n = len(rows)
    stats = sorted(_f1_of([rows[rng.randrange(n)] for _ in range(n)]) for _ in range(N_BOOT))
    return stats[int(0.025 * N_BOOT)], stats[int(0.975 * N_BOOT) - 1]


def _relabel(gold: list, label: str) -> list:
    """Predict the gold entity pair with a fixed label."""
    return [{**gold[0], "relation_type": label}] if gold else []


def _baselines(golds: list, rng: random.Random) -> dict:
    labels = Counter(g[0]["relation_type"] for g in golds if g)
    majority = labels.most_common(1)[0][0]
    positive = [l for l, _ in labels.most_common() if l not in NULL_TYPES]
    top_pos = positive[0] if positive else majority

    maj = _f1_of([_counts(g, _relabel(g, majority)) for g in golds])
    pos = _f1_of([_counts(g, _relabel(g, top_pos)) for g in golds])

    population, weights = zip(*labels.items())
    rand = sum(
        _f1_of([_counts(g, _relabel(g, rng.choices(population, weights)[0])) for g in golds])
        for _ in range(N_RANDOM)
    ) / N_RANDOM
    return {"majority_label": majority, "f1_majority": maj,
            "top_positive_label": top_pos, "f1_top_positive": pos,
            "f1_random": rand}


def evaluate_file(path: Path, rng: random.Random) -> dict | None:
    m = SLUG_RE.match(path.name)
    if not m:
        return None
    records = [json.loads(line) for line in path.open()]
    if not records:
        return None
    rows = [_counts(r["ground_truth"], [] if r.get("failed") else r["predicted"]) for r in records]
    tp, fp, fn = (sum(x[i] for x in rows) for i in range(3))
    p, r, f = _prf(tp, fp, fn)
    lo, hi = _bootstrap_ci(rows, rng)
    return {
        "dataset": m["dataset"], "model": m["model"], "temperature": m["temp"],
        "mode": m["mode"], "backend": m["backend"], "use_cot": m["cot"] == "cot",
        "use_prior": m["prior"] == "prior", "inference_mode": m["infmode"] or "standard",
        "n": len(records), "failures": sum(bool(x.get("failed")) for x in records),
        "tp": tp, "fp": fp, "fn": fn,
        "precision_pos": round(p, 4), "recall_pos": round(r, 4), "f1_pos": round(f, 4),
        "f1_pos_ci_low": round(lo, 4), "f1_pos_ci_high": round(hi, 4),
        **{k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in _baselines([x["ground_truth"] for x in records], rng).items()},
    }


def main() -> None:
    rng = random.Random(SEED)
    rows = [row for path in sorted(PREDICTIONS_DIR.glob("*.jsonl"))
            if (row := evaluate_file(path, rng)) is not None]
    df = pd.DataFrame(rows).sort_values(["dataset", "f1_pos"], ascending=[True, False])
    df.to_csv(OUT_PATH, index=False)

    cols = ["model", "mode", "backend", "inference_mode", "use_cot", "use_prior", "n", "failures",
            "precision_pos", "recall_pos", "f1_pos", "f1_pos_ci_low", "f1_pos_ci_high",
            "f1_majority", "f1_top_positive", "f1_random"]
    pd.set_option("display.width", 250)
    for ds, g in df.groupby("dataset"):
        print(f"\n=== {ds} ===")
        print(g[g["failures"] < g["n"]][cols].head(8).to_string(index=False))
    print(f"\nWritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
