import csv
import json
import itertools
import re
import random
from collections import Counter
from pathlib import Path
from typing import Annotated, List

import ollama
from tqdm import tqdm
import instructor
import outlines
from pydantic import BaseModel, Field, ValidationError

import math

from relextract_llm.util import (
    get_relations, get_prompt_instructions, get_relation_types,
    get_binary_prompt, compute_prior,
    DATASET_NULL_TYPE, RelationSpec, MatchCounter, _canonical, _canonical_ent,
)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class RelationList(BaseModel):
    relations: Annotated[List[RelationSpec], Field(max_length=30)]


class CoTRelationList(BaseModel):
    # reasoning FIRST so the model generates chain-of-thought before values
    reasoning: Annotated[str, Field(max_length=600)]
    relations: Annotated[List[RelationSpec], Field(max_length=30)]


class ConfidenceRelationSpec(BaseModel):
    """Like RelationSpec but carries a model-estimated confidence score."""
    entity_1: str
    entity_2: str
    relation_type: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class ConfidenceRelationList(BaseModel):
    relations: Annotated[List[ConfidenceRelationSpec], Field(max_length=30)]


class CoTConfidenceRelationList(BaseModel):
    reasoning: Annotated[str, Field(max_length=600)]
    relations: Annotated[List[ConfidenceRelationSpec], Field(max_length=30)]


class HasRelation(BaseModel):
    """Stage-1 response for two-stage inference."""
    has_relation: bool


class BinaryAnswer(BaseModel):
    """Per-type response for binary-decomposed inference."""
    is_match: bool
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


# ---------------------------------------------------------------------------
# Configuration grid
# ---------------------------------------------------------------------------
models = [
    "mixtral",
    "mistral",
    "qwen2.5",
    "phi4",
    "gemma3",
    "medgemma",
    "gemma4",
]
datasets     = ["chemprot_BLURB", "GAD_BLURB", "DDI_BLURB", "EU-ADR_BioBERT", "SemEval2010_task8"]
temperatures = [0, 0.5, 1.0]
# Modes:
#   zero_shot           — no examples in the prompt
#   few_shot_partial    — examples covering ceil(n_types/2) relation types
#   few_shot_exhaustive — examples covering every relation type
modes           = ["zero_shot", "few_shot_partial", "few_shot_exhaustive"]
markers_options = [True]
cot_options     = [False, True]
backends        = ["instructor", "outlines"]

# --- Additional experimental factors ---
# Prior-informed prompting: append empirical class distribution to prompt
use_prior_options   = [False, True]

# Inference mode
#   standard          — single structured-output call (baseline)
#   two_stage         — binary has_relation? call, then type classification
#   self_consistency  — N samples at temp>0, majority vote (requires temp>0)
#   binary_decomposed — N binary calls, one per positive type (expensive; off by default)
inference_modes     = ["standard", "two_stage", "self_consistency"]
# inference_modes   += ["binary_decomposed"]  # uncomment to include in grid

# Confidence elicitation: model outputs confidence per relation (standard mode only)
use_conf_options    = [False, True]

# Self-consistency samples
SC_SAMPLES = 5

RESULTS_PATH     = Path(__file__).resolve().parents[2] / "results" / "results.csv"
EXAMPLES_DIR     = Path(__file__).resolve().parents[2] / "results" / "examples"
PREDICTIONS_DIR  = Path(__file__).resolve().parents[2] / "results" / "predictions"
config_sample_pct = 0.3
N_EVAL_SAMPLES    = 300
MAX_LOG_EXAMPLES  = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_config(dataset, model, temp, mode, with_markers, backend,
                  use_cot, use_prior, inference_mode, use_confidence) -> bool:
    """Filter out logically invalid or redundant config combinations."""
    # self_consistency requires meaningful output variance; temp=0 approximates greedy
    # decoding and produces near-identical samples (GPU fp non-determinism aside),
    # making majority voting uninformative.
    if inference_mode == "self_consistency" and temp == 0:
        return False
    # CoT and confidence output only make sense with standard single-call inference
    if inference_mode != "standard" and use_cot:
        return False
    if inference_mode != "standard" and use_confidence:
        return False
    return True


def load_done_configs(path: Path) -> set:
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
                row.get("backend", "instructor"),
                row.get("use_cot", "False") == "True",
                row.get("use_prior", "False") == "True",
                row.get("inference_mode", "standard"),
                row.get("use_confidence", "False") == "True",
            ))
    return done


def append_result(result: dict, path: Path) -> None:
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def _config_slug(dataset, model, temp, mode, with_markers, backend,
                 use_cot, use_prior, inference_mode, use_confidence) -> str:
    model_short = model.split("/")[-1].replace(":", "-")
    mk    = "mk"    if with_markers   else "nomk"
    cot   = "cot"   if use_cot        else "nocot"
    prior = "prior" if use_prior      else "noprior"
    conf  = "conf"  if use_confidence else "noconf"
    return (f"{dataset}__{model_short}__t{temp}__{mode}__{mk}__{backend}"
            f"__{cot}__{prior}__{inference_mode}__{conf}")


def select_shots(all_data: List, mode: str, dataset: str) -> List:
    if mode == "zero_shot":
        return []
    all_types = get_relation_types(dataset)
    n_target  = len(all_types) if mode == "few_shot_exhaustive" else math.ceil(len(all_types) / 2)
    target_types = set(random.sample(all_types, n_target))
    candidates = list(all_data)
    random.shuffle(candidates)
    remaining = set(target_types)
    shots: List = []
    for example in candidates:
        if not remaining:
            break
        if {example.relation.relation_type} & remaining:
            shots.append(example)
            remaining -= {example.relation.relation_type}
    return shots


# ---------------------------------------------------------------------------
# Entity-bracket normalisation (strip [E1]/[/E1] etc. from predicted values)
# ---------------------------------------------------------------------------

_ENTITY_BRACKET_RE = re.compile(r"\[/?E\d\]")

def _normalize_entity(s: str) -> str:
    return _ENTITY_BRACKET_RE.sub("", s).strip()

def _normalize_relations(rels: List[RelationSpec]) -> List[RelationSpec]:
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

def _normalize_conf_relations(
    conf_rels: List[ConfidenceRelationSpec],
) -> tuple[List[RelationSpec], List[float]]:
    """Deduplicate ConfidenceRelationSpec → parallel (RelationSpec, confidence) lists."""
    seen: set = set()
    rels, confs = [], []
    for r in conf_rels:
        e1 = _normalize_entity(r.entity_1)
        e2 = _normalize_entity(r.entity_2)
        key = (e1, e2, r.relation_type)
        if key not in seen:
            seen.add(key)
            rels.append(RelationSpec(entity_1=e1, entity_2=e2, relation_type=r.relation_type))
            confs.append(r.confidence)
    return rels, confs


# ---------------------------------------------------------------------------
# Run functions — all return (List[RelationSpec], reasoning: str, confidences: List[float])
# ---------------------------------------------------------------------------

def run_instructor(model: str, prompt: str, temp: float,
                   use_cot: bool = False, use_confidence: bool = False,
                   ) -> tuple[List[RelationSpec], str, List[float]]:
    client = instructor.from_provider(model=model, mode=instructor.Mode.JSON)
    if use_cot and use_confidence:
        result = client.create(
            messages=[{"role": "user", "content": prompt}],
            response_model=CoTConfidenceRelationList,
            max_retries=3, timeout=240.0, temperature=temp,
        )
        rels, confs = _normalize_conf_relations(result.relations)
        return rels, result.reasoning, confs
    if use_cot:
        result = client.create(
            messages=[{"role": "user", "content": prompt}],
            response_model=CoTRelationList,
            max_retries=3, timeout=240.0, temperature=temp,
        )
        return _normalize_relations(result.relations), result.reasoning, []
    if use_confidence:
        result = client.create(
            messages=[{"role": "user", "content": prompt}],
            response_model=ConfidenceRelationList,
            max_retries=5, timeout=120.0, temperature=temp,
        )
        rels, confs = _normalize_conf_relations(result.relations)
        return rels, "", confs
    result = client.create(
        messages=[{"role": "user", "content": prompt}],
        response_model=list[RelationSpec],
        max_retries=5, timeout=120.0, temperature=temp,
    )
    return _normalize_relations(result), "", []


def run_outlines(model: str, prompt: str, temp: float,
                 use_cot: bool = False, use_confidence: bool = False,
                 ) -> tuple[List[RelationSpec], str, List[float]]:
    if use_cot and use_confidence:
        schema = CoTConfidenceRelationList
    elif use_cot:
        schema = CoTRelationList
    elif use_confidence:
        schema = ConfidenceRelationList
    else:
        schema = RelationList

    ollama_model = model.removeprefix("ollama/")
    ol_model = outlines.from_ollama(ollama.Client(), ollama_model)
    raw = ol_model(prompt, schema, options={"temperature": temp})
    if isinstance(raw, dict):
        parsed = schema.model_validate(raw)
    elif isinstance(raw, str):
        parsed = schema.model_validate_json(raw)
    else:
        parsed = raw

    reasoning = parsed.reasoning if use_cot else ""
    if use_confidence:
        rels, confs = _normalize_conf_relations(parsed.relations)
        return rels, reasoning, confs
    return _normalize_relations(parsed.relations), reasoning, []


# --- Two-stage inference ---

def run_two_stage_instructor(
    model: str, binary_prompt: str, full_prompt: str, text: str,
    temp: float, null_type: str | None, entity_1: str, entity_2: str,
) -> tuple[List[RelationSpec], str, List[float]]:
    """Stage 1: binary has_relation? Stage 2: type classification (if yes)."""
    client = instructor.from_provider(model=model, mode=instructor.Mode.JSON)
    stage1 = client.create(
        messages=[{"role": "user", "content": f"{binary_prompt}\n\nText: {text}"}],
        response_model=HasRelation,
        max_retries=3, timeout=60.0, temperature=temp,
    )
    if not stage1.has_relation:
        if null_type:
            return [RelationSpec(entity_1=entity_1, entity_2=entity_2,
                                 relation_type=null_type)], "", []
        return [], "", []
    result = client.create(
        messages=[{"role": "user", "content": full_prompt}],
        response_model=list[RelationSpec],
        max_retries=5, timeout=120.0, temperature=temp,
    )
    return _normalize_relations(result), "", []


def run_two_stage_outlines(
    model: str, binary_prompt: str, full_prompt: str, text: str,
    temp: float, null_type: str | None, entity_1: str, entity_2: str,
) -> tuple[List[RelationSpec], str, List[float]]:
    ollama_model = model.removeprefix("ollama/")
    ol = outlines.from_ollama(ollama.Client(), ollama_model)

    raw1 = ol(f"{binary_prompt}\n\nText: {text}", HasRelation, options={"temperature": temp})
    stage1 = HasRelation.model_validate(raw1) if isinstance(raw1, dict) else (
        HasRelation.model_validate_json(raw1) if isinstance(raw1, str) else raw1)

    if not stage1.has_relation:
        if null_type:
            return [RelationSpec(entity_1=entity_1, entity_2=entity_2,
                                 relation_type=null_type)], "", []
        return [], "", []

    raw2 = ol(full_prompt, RelationList, options={"temperature": temp})
    parsed = RelationList.model_validate(raw2) if isinstance(raw2, dict) else (
        RelationList.model_validate_json(raw2) if isinstance(raw2, str) else raw2)
    return _normalize_relations(parsed.relations), "", []


# --- Self-consistency decoding ---

def run_self_consistency_instructor(
    model: str, prompt: str, temp: float, use_cot: bool = False, n: int = SC_SAMPLES,
) -> tuple[List[RelationSpec], str, List[float]]:
    """Run N samples, majority-vote the predicted relation triple.
    Returns confidence = fraction of samples that agreed on the winner.
    """
    votes: Counter = Counter()
    for _ in range(n):
        try:
            rels, _, _ = run_instructor(model, prompt, temp, use_cot=use_cot)
            for r in rels:
                votes[_canonical(r)] += 1
        except Exception:
            pass
    if not votes:
        raise RuntimeError("All self-consistency samples failed")
    (e1, e2, rt), top_count = votes.most_common(1)[0]
    consistency = top_count / n
    return [RelationSpec(entity_1=e1, entity_2=e2, relation_type=rt)], "", [consistency]


def run_self_consistency_outlines(
    model: str, prompt: str, temp: float, use_cot: bool = False, n: int = SC_SAMPLES,
) -> tuple[List[RelationSpec], str, List[float]]:
    ollama_model = model.removeprefix("ollama/")
    ol = outlines.from_ollama(ollama.Client(), ollama_model)
    schema = CoTRelationList if use_cot else RelationList
    votes: Counter = Counter()
    for _ in range(n):
        try:
            raw = ol(prompt, schema, options={"temperature": temp})
            parsed = schema.model_validate(raw) if isinstance(raw, dict) else (
                schema.model_validate_json(raw) if isinstance(raw, str) else raw)
            for r in _normalize_relations(parsed.relations):
                votes[_canonical(r)] += 1
        except Exception:
            pass
    if not votes:
        raise RuntimeError("All self-consistency samples failed")
    (e1, e2, rt), top_count = votes.most_common(1)[0]
    consistency = top_count / n
    return [RelationSpec(entity_1=e1, entity_2=e2, relation_type=rt)], "", [consistency]


# --- Binary-decomposed inference ---

def run_binary_decomposed_instructor(
    model: str, base_prompt: str, text: str, temp: float,
    dataset: str, entity_1: str, entity_2: str,
) -> tuple[List[RelationSpec], str, List[float]]:
    """Ask one binary question per positive type; predict highest-confidence yes."""
    client = instructor.from_provider(model=model, mode=instructor.Mode.JSON)
    all_types  = get_relation_types(dataset)
    null_type  = DATASET_NULL_TYPE.get(dataset)
    pos_types  = [t for t in all_types if t != null_type]
    best_type  = null_type or all_types[0]
    best_conf  = 0.0
    for t in pos_types:
        q = (f"{base_prompt}\n\n"
             f"For the text below, answer ONLY whether the relation type is exactly '{t}'.\n"
             f"is_match=true if '{t}' is the correct type, false otherwise.")
        try:
            ans = client.create(
                messages=[{"role": "user", "content": f"{q}\n\nText: {text}"}],
                response_model=BinaryAnswer,
                max_retries=3, timeout=60.0, temperature=temp,
            )
            if ans.is_match and ans.confidence > best_conf:
                best_type, best_conf = t, ans.confidence
        except Exception:
            pass
    conf = best_conf if best_conf > 0 else 0.5
    return [RelationSpec(entity_1=entity_1, entity_2=entity_2,
                         relation_type=best_type)], "", [conf]


def run_binary_decomposed_outlines(
    model: str, base_prompt: str, text: str, temp: float,
    dataset: str, entity_1: str, entity_2: str,
) -> tuple[List[RelationSpec], str, List[float]]:
    ollama_model = model.removeprefix("ollama/")
    ol = outlines.from_ollama(ollama.Client(), ollama_model)
    all_types  = get_relation_types(dataset)
    null_type  = DATASET_NULL_TYPE.get(dataset)
    pos_types  = [t for t in all_types if t != null_type]
    best_type  = null_type or all_types[0]
    best_conf  = 0.0
    for t in pos_types:
        q = (f"{base_prompt}\n\n"
             f"For the text below, answer ONLY whether the relation type is exactly '{t}'.\n"
             f"is_match=true if '{t}' is the correct type, false otherwise.")
        try:
            raw = ol(f"{q}\n\nText: {text}", BinaryAnswer, options={"temperature": temp})
            ans = BinaryAnswer.model_validate(raw) if isinstance(raw, dict) else (
                BinaryAnswer.model_validate_json(raw) if isinstance(raw, str) else raw)
            if ans.is_match and ans.confidence > best_conf:
                best_type, best_conf = t, ans.confidence
        except Exception:
            pass
    conf = best_conf if best_conf > 0 else 0.5
    return [RelationSpec(entity_1=entity_1, entity_2=entity_2,
                         relation_type=best_type)], "", [conf]


# ---------------------------------------------------------------------------
# Match-label helper (for human-readable example logs)
# ---------------------------------------------------------------------------

def _match_label(ground_truth, predicted) -> str:
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


def _fmt_relations(relations) -> str:
    if not relations:
        return "  (none)"
    return "\n".join(
        f"  entity_1={r.entity_1!r:20s}  entity_2={r.entity_2!r:20s}  relation={r.relation_type!r}"
        for r in relations
    )


def write_example_log(slug: str, shots: list, example_records: list) -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = EXAMPLES_DIR / f"{slug}.txt"
    to_log = example_records[:MAX_LOG_EXAMPLES]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {slug}\n")
        f.write(f"Logged {len(to_log)} of {len(example_records)} processed examples\n")
        if shots:
            f.write(f"Few-shot examples used: {[s.id for s in shots]}\n")
        f.write("=" * 72 + "\n\n")
        for i, rec in enumerate(to_log, 1):
            if rec.get("failed"):
                label = "FAILED        (structured output not produced)"
            else:
                label = _match_label(rec["ground_truth"], rec["predicted"])
            f.write(f"--- Example {i}/{len(to_log)}  (ID: {rec['id']}) ---\n")
            f.write(f"MATCH: {label}\n\n")
            f.write("PROMPT\n" + "-" * 60 + "\n")
            f.write(rec["prompt"] + "\n" + "-" * 60 + "\n\n")
            if rec.get("reasoning"):
                f.write("REASONING\n  " + rec["reasoning"].replace("\n", "\n  ") + "\n\n")
            if rec.get("confidences"):
                f.write(f"CONFIDENCES: {rec['confidences']}\n\n")
            f.write("GROUND TRUTH\n" + _fmt_relations(rec["ground_truth"]) + "\n\n")
            if not rec.get("failed"):
                f.write("PREDICTED\n" + _fmt_relations(rec["predicted"]) + "\n")
            f.write("\n" + "=" * 72 + "\n\n")
    print(f"  Example log  → {path}")


# ---------------------------------------------------------------------------
# Build the config list for this session
# ---------------------------------------------------------------------------

all_configs = [
    c for c in itertools.product(
        datasets, models, temperatures, modes,
        markers_options, backends, cot_options,
        use_prior_options, inference_modes, use_conf_options,
    )
    if _valid_config(*c)
]

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
pbar  = tqdm(to_run, unit="cfg")

for (dataset, model, temp, mode, with_markers, backend,
     use_cot, use_prior, inference_mode, use_confidence) in pbar:

    model_short = model.split("/")[-1]
    pbar.set_description(
        f"{dataset} | {model_short} | t={temp} | {mode} | {inference_mode}"
        + (" cot" if use_cot else "")
        + (" prior" if use_prior else "")
        + (" conf" if use_confidence else "")
    )

    stats.start_experiment(
        model=model, dataset=dataset, temperature=temp,
        mode=mode, with_markers=with_markers, backend=backend,
        use_cot=use_cot, use_prior=use_prior,
        inference_mode=inference_mode, use_confidence=use_confidence,
    )

    all_data = get_relations(dataset)
    shots    = select_shots(all_data, mode, dataset)
    shot_ids = {s.id for s in shots}
    eval_data = [ex for ex in all_data if ex.id not in shot_ids]

    # Compute class prior from full dataset if requested
    class_prior = compute_prior(all_data) if use_prior else None

    prompt_instr = get_prompt_instructions(
        dataset, with_markers=with_markers, shots=shots,
        use_cot=use_cot, class_prior=class_prior,
    )

    # For two-stage: build binary prompt once per config
    binary_prompt = (get_binary_prompt(dataset, with_markers=with_markers)
                     if inference_mode == "two_stage" else None)

    random_sample = random.sample(eval_data, min(len(eval_data), N_EVAL_SAMPLES))

    example_records: list = []
    _logged_errors: dict[str, int] = {}
    EARLY_STOP_AFTER  = 20
    EARLY_STOP_THRESH = 0.20
    early_stopped = False

    for example in tqdm(random_sample, desc="examples", unit="ex", leave=False):
        text   = example.text_with_entity_marker if with_markers else example.text
        prompt = f"{prompt_instr}\n\nExtract from this text:\n\n{text}"

        null_type = DATASET_NULL_TYPE.get(dataset)
        e1, e2    = example.relation.entity_1, example.relation.entity_2

        try:
            if inference_mode == "standard":
                if backend == "instructor":
                    resp, reasoning, confs = run_instructor(
                        model, prompt, temp, use_cot=use_cot, use_confidence=use_confidence)
                else:
                    resp, reasoning, confs = run_outlines(
                        model, prompt, temp, use_cot=use_cot, use_confidence=use_confidence)

            elif inference_mode == "two_stage":
                if backend == "instructor":
                    resp, reasoning, confs = run_two_stage_instructor(
                        model, binary_prompt, prompt, text, temp, null_type, e1, e2)
                else:
                    resp, reasoning, confs = run_two_stage_outlines(
                        model, binary_prompt, prompt, text, temp, null_type, e1, e2)

            elif inference_mode == "self_consistency":
                if backend == "instructor":
                    resp, reasoning, confs = run_self_consistency_instructor(
                        model, prompt, temp, use_cot=use_cot)
                else:
                    resp, reasoning, confs = run_self_consistency_outlines(
                        model, prompt, temp, use_cot=use_cot)

            elif inference_mode == "binary_decomposed":
                if backend == "instructor":
                    resp, reasoning, confs = run_binary_decomposed_instructor(
                        model, prompt_instr, text, temp, dataset, e1, e2)
                else:
                    resp, reasoning, confs = run_binary_decomposed_outlines(
                        model, prompt_instr, text, temp, dataset, e1, e2)

            else:
                raise ValueError(f"Unknown inference_mode: {inference_mode}")

        except ValidationError as exc:
            resp, reasoning, confs = None, "", []
            err_key = f"ValidationError: {str(exc)[:120]}"
            _logged_errors[err_key] = _logged_errors.get(err_key, 0) + 1
            if _logged_errors[err_key] == 1:
                tqdm.write(f"  [failure] {err_key}")
        except Exception as exc:
            resp, reasoning, confs = None, "", []
            err_key = f"{type(exc).__name__}: {str(exc)[:120]}"
            _logged_errors[err_key] = _logged_errors.get(err_key, 0) + 1
            if _logged_errors[err_key] == 1:
                tqdm.write(f"  [failure] {err_key}")

        failed = resp is None
        if failed:
            stats.record_failure()
        else:
            stats.update([example.relation], resp)

        example_records.append({
            "id":           example.id,
            "prompt":       prompt,
            "ground_truth": [example.relation],
            "predicted":    resp if not failed else [],
            "reasoning":    reasoning,
            "confidences":  confs,
            "failed":       failed,
        })

        n_so_far = len(example_records)
        if n_so_far == EARLY_STOP_AFTER:
            n_failed = sum(1 for r in example_records if r["failed"])
            if n_failed / n_so_far >= EARLY_STOP_THRESH:
                tqdm.write(
                    f"  [early-stop] {n_failed}/{n_so_far} failures "
                    f"({100*n_failed/n_so_far:.0f}%) — skipping remaining examples"
                )
                early_stopped = True
                break

    if _logged_errors:
        total_failures = sum(_logged_errors.values())
        tqdm.write(f"  [failures] {total_failures} total"
                   f"{' (early-stopped)' if early_stopped else ''} — breakdown:")
        for err_key, count in sorted(_logged_errors.items(), key=lambda x: -x[1]):
            tqdm.write(f"    {count:>4}x  {err_key}")

    stats.finish_experiment()
    append_result(stats._results[-1], RESULTS_PATH)

    slug = _config_slug(dataset, model, temp, mode, with_markers, backend,
                        use_cot, use_prior, inference_mode, use_confidence)
    write_example_log(slug, shots, example_records)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = PREDICTIONS_DIR / f"{slug}.jsonl"
    with open(pred_path, "w", encoding="utf-8") as pf:
        for rec in example_records:
            pf.write(json.dumps({
                "id":           rec["id"],
                "ground_truth": [r.model_dump() for r in rec["ground_truth"]],
                "predicted":    [r.model_dump() for r in rec["predicted"]],
                "reasoning":    rec.get("reasoning", ""),
                "confidences":  rec.get("confidences", []),
                "failed":       rec.get("failed", False),
            }) + "\n")

stats.print_summary()
