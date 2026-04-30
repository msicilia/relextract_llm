import csv
import itertools
import re
import random
from pathlib import Path
from typing import Annotated, List

import ollama
from tqdm import tqdm
import instructor
import outlines
from pydantic import BaseModel, Field, ValidationError

import math

from relextract_llm.util import (get_relations, get_prompt_instructions, get_relation_types,
                                  RelationSpec, MatchCounter, _canonical, _canonical_ent)

# ---------------------------------------------------------------------------
# Outlines requires a Pydantic BaseModel (not a bare list), so we wrap the
# list of relations in a container.  Instructor can use list[RelationSpec]
# directly.
# ---------------------------------------------------------------------------
class RelationList(BaseModel):
    # max_length=30 adds maxItems:30 to the JSON schema, preventing Ollama from
    # generating unbounded arrays that hit the context limit and produce truncated JSON.
    relations: Annotated[List[RelationSpec], Field(max_length=30)]


# ---------------------------------------------------------------------------
# Configuration grid
# ---------------------------------------------------------------------------
models       = [
                "ollama/medgemma:4b"
                #"ollama/MedAIBase/MedGemma1.5:4b"] # , "ollama/mistral"
                ]
datasets     = ["chemprot_BLURB", "GAD_BLURB", "DDI_BLURB", "EU-ADR_BioBERT", "SemEval2010_task8"]
temperatures = [0, 0.5] # , 1.0]
# Modes:
#   zero_shot          — no examples in the prompt
#   few_shot_partial   — examples covering ~half the relation types (ceil(n_types/2))
#   few_shot_exhaustive — examples covering every relation type (at least one each)
modes        = [
                "zero_shot", 
                "few_shot_partial", 
                "few_shot_exhaustive"]
markers_options = [True] # [True, False]
backends     = ["instructor", "outlines"]

RESULTS_PATH       = Path(__file__).resolve().parents[2] / "results" / "results.csv"
EXAMPLES_DIR       = Path(__file__).resolve().parents[2] / "results" / "examples"
config_sample_pct  = 0.3 # fraction of pending configs to run in this session (0.0–1.0)
examples_sample_pct = 0.1 # fraction of eval examples to sample per experiment (0.0–1.0)
# N_EVAL_SAMPLES     = 100     # exact number of eval examples to sample per experiment
MAX_LOG_EXAMPLES   = 10     # max examples written to the per-experiment log file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_done_configs(path: Path) -> set:
    """Return the set of (dataset, model, temperature, mode, with_markers, backend)
    tuples already recorded in results.csv."""
    if not path.exists():
        return set()
    done = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            done.add((
                row["dataset"],
                row["model"],
                float(row["temperature"]),
                row["mode"],
                row["with_markers"] == "True",
                row.get("backend", "instructor"),   # backwards-compat for rows without backend
            ))
    return done


def append_result(result: dict, path: Path) -> None:
    """Append a single result row, writing the header only if the file is new."""
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def _config_slug(dataset: str, model: str, temp: float, mode: str,
                 with_markers: bool, backend: str) -> str:
    """Return a filesystem-safe identifier for an experiment configuration."""
    model_short = model.split("/")[-1].replace(":", "-")
    mk = "mk" if with_markers else "nomk"
    return f"{dataset}__{model_short}__t{temp}__{mode}__{mk}__{backend}"


def select_shots(all_data: List, mode: str, dataset: str) -> List:
    """Select few-shot examples according to the mode.

    zero_shot           — returns []
    few_shot_partial    — greedily covers ceil(n_types / 2) distinct relation types,
                          choosing one example per needed type in random order.
    few_shot_exhaustive — greedily covers every relation type defined for the dataset,
                          choosing one example per needed type in random order.

    The greedy strategy shuffles the candidate pool, then picks the first example
    that covers at least one still-uncovered type, marking all types in that example
    as covered.  This minimises the number of shot examples while still meeting
    the coverage target.
    """
    if mode == "zero_shot":
        return []

    all_types = get_relation_types(dataset)
    n_target  = len(all_types) if mode == "few_shot_exhaustive" else math.ceil(len(all_types) / 2)

    # Randomly pick which types must be covered (preserves diversity across runs).
    target_types = set(random.sample(all_types, n_target))

    candidates = list(all_data)
    random.shuffle(candidates)

    remaining = set(target_types)
    shots: List = []
    for example in candidates:
        if not remaining:
            break
        ex_types = {r.relation_type for r in example.relations}
        if ex_types & remaining:          # covers at least one still-needed type
            shots.append(example)
            remaining -= ex_types

    return shots


def _match_label(ground_truth: List[RelationSpec], predicted: List[RelationSpec]) -> str:
    """Classify the match between ground truth and prediction for one example."""
    truth_full = {_canonical(r) for r in ground_truth}
    pred_full  = {_canonical(r) for r in predicted}
    truth_ent  = {_canonical_ent(r) for r in ground_truth}
    pred_ent   = {_canonical_ent(r) for r in predicted}

    if pred_full == truth_full:
        return "COMPLETE MATCH"
    if pred_ent == truth_ent and truth_ent:
        return "ENTITY MATCH  (pairs correct, relation types wrong)"
    if pred_ent & truth_ent:
        return "PARTIAL MATCH (some entity pairs overlap)"
    if not predicted:
        return "MISSED        (no prediction)"
    if not ground_truth:
        return "FALSE POS     (nothing expected, something predicted)"
    return "NO MATCH      (entity pairs do not overlap)"


def _fmt_relations(relations: List[RelationSpec]) -> str:
    if not relations:
        return "  (none)"
    return "\n".join(
        f"  entity_1={r.entity_1!r:20s}  entity_2={r.entity_2!r:20s}  relation={r.relation_type!r}"
        for r in relations
    )


def write_example_log(slug: str, shots: list, example_records: list) -> None:
    """Write a human-readable log of sampled examples for one experiment.

    Parameters
    ----------
    slug:
        Unique filename stem for this experiment.
    shots:
        The few-shot examples used in the prompt (may be empty).
    example_records:
        List of dicts with keys: id, prompt, ground_truth, predicted.
        At most MAX_LOG_EXAMPLES are written.
    """
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = EXAMPLES_DIR / f"{slug}.txt"

    to_log = example_records[:MAX_LOG_EXAMPLES]
    n_total = len(example_records)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {slug}\n")
        f.write(f"Logged {len(to_log)} of {n_total} processed examples\n")
        if shots:
            f.write(f"Few-shot examples used: {[s.id for s in shots]}\n")
        f.write("=" * 72 + "\n\n")

        for i, rec in enumerate(to_log, 1):
            label = _match_label(rec["ground_truth"], rec["predicted"])
            f.write(f"--- Example {i}/{len(to_log)}  (ID: {rec['id']}) ---\n")
            f.write(f"MATCH: {label}\n\n")
            f.write("PROMPT\n")
            f.write("-" * 60 + "\n")
            f.write(rec["prompt"] + "\n")
            f.write("-" * 60 + "\n\n")
            f.write("GROUND TRUTH\n")
            f.write(_fmt_relations(rec["ground_truth"]) + "\n\n")
            f.write("PREDICTED\n")
            f.write(_fmt_relations(rec["predicted"]) + "\n")
            f.write("\n" + "=" * 72 + "\n\n")

    print(f"  Example log  → {path}")


_ENTITY_BRACKET_RE = re.compile(r"\[/?E\d\]")

def _normalize_entity(s: str) -> str:
    """Strip [E1], [/E1], [E2], [/E2] wrappers that appear in text_with_entity_marker
    but should NOT appear in extracted entity values."""
    return _ENTITY_BRACKET_RE.sub("", s).strip()

def _normalize_relations(rels: List[RelationSpec]) -> List[RelationSpec]:
    """Strip entity bracket markers and deduplicate predicted relation triples."""
    seen: set = set()
    result = []
    for r in rels:
        norm = RelationSpec(
            entity_1=_normalize_entity(r.entity_1),
            entity_2=_normalize_entity(r.entity_2),
            relation_type=r.relation_type,
        )
        key = (norm.entity_1, norm.entity_2, norm.relation_type)
        if key not in seen:
            seen.add(key)
            result.append(norm)
    return result


def run_instructor(model: str, prompt: str, temp: float) -> List[RelationSpec]:
    """Call the model via instructor (retry-based structured output)."""
    client = instructor.from_provider(model=model, mode=instructor.Mode.JSON)
    result = client.create(
        messages=[{"role": "user", "content": prompt}],
        response_model=list[RelationSpec],
        max_retries=5,
        timeout=120.0,
        temperature=temp,
    )
    return _normalize_relations(result)


def run_outlines(model: str, prompt: str, temp: float) -> List[RelationSpec]:
    """Call the model via outlines (JSON-schema-constrained generation).

    outlines passes **kwargs from __call__ through to ollama.Client.chat(),
    so temperature must be supplied as options={"temperature": temp} on the
    call, NOT on from_ollama() which only accepts (client, model_name).
    The return value of generate() is a raw JSON string.
    """
    ollama_model = model.removeprefix("ollama/")
    ol_model = outlines.from_ollama(ollama.Client(), ollama_model)
    raw = ol_model(prompt, RelationList, options={"temperature": temp})
    if isinstance(raw, RelationList):
        result = raw.relations
    elif isinstance(raw, dict):
        result = RelationList.model_validate(raw).relations
    else:
        result = RelationList.model_validate_json(raw).relations
    return _normalize_relations(result)


# ---------------------------------------------------------------------------
# Build the list of configs to run this session
# ---------------------------------------------------------------------------
all_configs  = list(itertools.product(datasets, models, temperatures, modes, markers_options, backends))
done_configs = load_done_configs(RESULTS_PATH)
pending      = [c for c in all_configs if c not in done_configs]
random.shuffle(pending)
to_run       = pending[:max(1, round(len(pending) * config_sample_pct))]

print(f"Total configs : {len(all_configs)}")
print(f"Already done  : {len(done_configs)}")
print(f"Pending       : {len(pending)}")
print(f"Running       : {len(to_run)}  ({config_sample_pct:.0%} of pending)")

# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
stats = MatchCounter()

pbar = tqdm(to_run, unit="cfg")
for dataset, model, temp, mode, with_markers, backend in pbar:
    model_short = model.split("/")[-1]
    pbar.set_description(f"{dataset} | {model_short} | t={temp} | {mode}")

    stats.start_experiment(
        model=model, dataset=dataset, temperature=temp,
        mode=mode, with_markers=with_markers, backend=backend,
    )

    all_data = get_relations(dataset)

    shots     = select_shots(all_data, mode, dataset)
    shot_ids  = {s.id for s in shots}
    eval_data = [ex for ex in all_data if ex.id not in shot_ids]

    prompt_instr = get_prompt_instructions(dataset, with_markers=with_markers, shots=shots)

    sample_size   = max(1, round(len(eval_data) * examples_sample_pct))
    random_sample = random.sample(eval_data, min(len(eval_data), sample_size))

    example_records: list = []

    for example in tqdm(random_sample, desc="examples", unit="ex", leave=False):
        text   = example.text_with_entity_marker if with_markers else example.text
        prompt = f"{prompt_instr}\n\nExtract from this text:\n\n{text}"
        try:
            if backend == "instructor":
                resp = run_instructor(model, prompt, temp)
            else:
                resp = run_outlines(model, prompt, temp)
        except ValidationError as e:
            resp = []
        except Exception as e:
            resp = []

        stats.update(example.relations, resp)
        example_records.append({
            "id":           example.id,
            "prompt":       prompt,
            "ground_truth": example.relations,
            "predicted":    resp,
        })

    stats.finish_experiment()
    append_result(stats._results[-1], RESULTS_PATH)

    slug = _config_slug(dataset, model, temp, mode, with_markers, backend)
    write_example_log(slug, shots, example_records)

stats.print_summary()
