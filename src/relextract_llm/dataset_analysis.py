import csv
import json
import statistics
from pathlib import Path

from relextract_llm.util import get_relations

DATASETS = ["chemprot_BLURB", "GAD_BLURB", "DDI_BLURB", "EU-ADR_BioBERT", "SemEval2010_task8"]

def _load_negative_type(dataset_name: str) -> str | None:
    """Return the relation type label with id=0 (the negative/false class), or None."""
    path = Path(__file__).resolve().parents[2] / "data" / dataset_name / "relation_types.json"
    if not path.exists():
        return None
    rel_types = json.loads(path.read_text(encoding="utf-8"))
    return next((label for label, meta in rel_types.items() if meta["id"] == 0), None)

rows = []
for dataset_name in DATASETS:
    # get_relations already merges lines sharing the same text
    examples = get_relations(dataset_name)
    negative_type = _load_negative_type(dataset_name)

    n = len(examples)
    all_relations = [r for ex in examples for r in ex.relations]
    relations_per_example = [len(ex.relations) for ex in examples]
    text_lengths = [len(ex.text) for ex in examples]
    relation_types = {r.relation_type for r in all_relations}
    total_relations = len(all_relations)
    positive_count = (
        sum(1 for r in all_relations if r.relation_type != negative_type)
        if negative_type else len(all_relations)
    )

    rows.append({
        "dataset":                dataset_name,
        "num_examples":           n,
        "total_relations":        total_relations,
        "positive_ratio":         round(positive_count / total_relations, 4) if total_relations > 0 else 0,
        "avg_relations_per_text": round(statistics.fmean(relations_per_example), 4) if n > 0 else 0,
        "std_relations_per_text": round(statistics.pstdev(relations_per_example), 4) if n > 0 else 0,
        "avg_text_length":        round(statistics.fmean(text_lengths), 4) if n > 0 else 0,
        "std_text_length":        round(statistics.pstdev(text_lengths), 4) if n > 0 else 0,
        "num_relation_types":     len(relation_types),
    })

output_path = Path(__file__).resolve().parents[2] / "results" / "dataset_metrics.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Metrics saved to: {output_path}\n")

# Console summary
col = {"dataset": 16, "n": 8, "total": 8, "pos": 8, "avg_rel": 8, "std_rel": 8, "avg_len": 8, "std_len": 8, "types": 6}
header = (f"{'Dataset':<{col['dataset']}} {'#Examples':>{col['n']}} {'TotRel':>{col['total']}}"
          f" {'PosRatio':>{col['pos']}} {'AvgRel':>{col['avg_rel']}} {'StdRel':>{col['std_rel']}}"
          f" {'AvgLen':>{col['avg_len']}} {'StdLen':>{col['std_len']}} {'Types':>{col['types']}}")
sep = "-" * len(header)
print(sep)
print(header)
print(sep)
for r in rows:
    print(f"{r['dataset']:<{col['dataset']}} {r['num_examples']:>{col['n']}}"
          f" {r['total_relations']:>{col['total']}}"
          f" {r['positive_ratio']:>{col['pos']}.4f}"
          f" {r['avg_relations_per_text']:>{col['avg_rel']}.4f}"
          f" {r['std_relations_per_text']:>{col['std_rel']}.4f}"
          f" {r['avg_text_length']:>{col['avg_len']}.1f}"
          f" {r['std_text_length']:>{col['std_len']}.1f}"
          f" {r['num_relation_types']:>{col['types']}}")
print(sep)
