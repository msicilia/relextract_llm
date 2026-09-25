import csv
import json
from collections import defaultdict
from pathlib import Path
from pydantic import BaseModel
from typing import List, Set, Tuple

class RelationSpec(BaseModel):
    """Simplified relation specification for relation extraction tasks."""
    entity_1: str
    entity_2: str
    relation_type: str
  
class RelationExample(BaseModel):
    id: str
    text: str
    text_with_entity_marker: str
    text_with_typed_entity_marker: str
    relation: RelationSpec

# Canonical relation types for each dataset, in a stable order.
DATASET_RELATION_TYPES: dict[str, List[str]] = {
    "chemprot_BLURB":  ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "CPR:false"],
    "GAD_BLURB":       ["positive", "negative"],
    "DDI_BLURB":       ["DDI-advise", "DDI-effect", "DDI-int", "DDI-mechanism", "DDI-false"],
    "EU-ADR_BioBERT":  ["positive", "negative"],
    "SemEval2010_task8": [
        "Cause-Effect(e1,e2)", "Cause-Effect(e2,e1)",
        "Component-Whole(e1,e2)", "Component-Whole(e2,e1)",
        "Content-Container(e1,e2)", "Content-Container(e2,e1)",
        "Entity-Destination(e1,e2)", "Entity-Destination(e2,e1)",
        "Entity-Origin(e1,e2)", "Entity-Origin(e2,e1)",
        "Instrument-Agency(e1,e2)", "Instrument-Agency(e2,e1)",
        "Member-Collection(e1,e2)", "Member-Collection(e2,e1)",
        "Message-Topic(e1,e2)", "Message-Topic(e2,e1)",
        "Product-Producer(e1,e2)", "Product-Producer(e2,e1)",
        "Other",
    ],
}


def get_relation_types(dataset_name: str) -> List[str]:
    """Return the ordered list of relation type codes for a dataset."""
    if dataset_name not in DATASET_RELATION_TYPES:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_RELATION_TYPES[dataset_name]


# The "null / no-relation" type for each dataset (used by two-stage and binary-decomposed).
# Datasets without a null class map to None.
DATASET_NULL_TYPE: dict[str, str | None] = {
    "chemprot_BLURB":    "CPR:false",
    "DDI_BLURB":         "DDI-false",
    "SemEval2010_task8": "Other",
    "GAD_BLURB":         None,
    "EU-ADR_BioBERT":    None,
}


def compute_prior(examples: List["RelationExample"]) -> dict[str, float]:
    """Compute empirical class distribution from a list of RelationExample objects."""
    counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        counts[ex.relation.relation_type] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in sorted(counts.items(), key=lambda x: -x[1])}


def get_relations(dataset_name: str) -> List[RelationExample]:
    """Load all JSONL examples from a dataset, merging lines that share the same text."""
    base_path = Path(__file__).resolve().parents[2] / "data" / dataset_name
    print(f"Loading relations from dataset: {dataset_name}")

    # Use text as key to merge entity pairs from different lines into one example.
    # Intermediate dicts accumulate relations lists before conversion to RelationExample.
    by_text: dict[str, dict] = {}

    target_files = []
    for prefix in ["train", "dev", "test"]:
        target_files.extend(list(base_path.glob(f"{prefix}*")))

    for file_path in target_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    specs = [
                        RelationSpec(
                            entity_1=rel["entity_1"],
                            entity_2=rel["entity_2"],
                            relation_type=rel["relation_type"]
                        )
                        for rel in entry.get("relation", [])
                    ]
                    text = entry["text"]
                    if text in by_text:
                        by_text[text]["relations"].extend(specs)
                    else:
                        by_text[text] = {
                            "id":   entry["id"],
                            "text": text,
                            "text_with_entity_marker":       entry["text_with_entity_marker"],
                            "text_with_typed_entity_marker": entry["text_with_typed_entity_marker"],
                            "relations": specs,
                        }
                except json.JSONDecodeError as e:
                    print(f"Error parsing {file_path.name} at line {line_num}: {e}")

    # Deduplicate relations within each example (EU-ADR 10-fold CV can repeat triples).
    for data in by_text.values():
        seen: set = set()
        unique = []
        for r in data["relations"]:
            key = (r.entity_1, r.entity_2, r.relation_type)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        data["relations"] = unique

    # Remove examples with more than one relation and report.
    multi = [d for d in by_text.values() if len(d["relations"]) != 1]
    if multi:
        print(f"  Removed {len(multi)} multi-relation examples from {dataset_name} "
              f"(kept {len(by_text) - len(multi)})")

    return [
        RelationExample(
            id=d["id"],
            text=d["text"],
            text_with_entity_marker=d["text_with_entity_marker"],
            text_with_typed_entity_marker=d["text_with_typed_entity_marker"],
            relation=d["relations"][0],
        )
        for d in by_text.values()
        if len(d["relations"]) == 1
    ]

def get_prompt_instructions(dataset_name: str, with_markers: bool,
                            shots: List["RelationExample"] | None = None,
                            use_cot: bool = False,
                            class_prior: dict[str, float] | None = None) -> str:
    """Returns a dataset-specific prompt for relation extraction, optionally with few-shot examples."""
    if dataset_name == "chemprot_BLURB":
        intro = ("Classify the relation between the chemical entity and the protein entity in the text below. "
                 "You MUST output exactly one relation for the entity pair — even if no positive interaction "
                 "is stated (use CPR:false in that case). "
                 "entity_1 = the chemical, entity_2 = the protein/gene, "
                 "relation_type = one of the codes below, EXACTLY as written.")
        markers = ("Chemicals are marked with @CHEMICAL$ and proteins with @GENE$. "
                   "Use exactly those marker strings as entity values, not any other text.")
        types = ("Relation type codes — first check whether any of CPR:3–CPR:9 fits; "
                 "if none does, output CPR:false.\n"
                 "  CPR:3 — Chemical upregulates, activates, or increases expression/activity of the protein "
                 "(keywords: activates, induces, upregulates, increases expression of, enhances activity of)\n"
                 "  CPR:4 — Chemical downregulates, inhibits, or decreases expression/activity of the protein "
                 "(keywords: inhibits, suppresses, downregulates, reduces expression of, blocks activity of)\n"
                 "  CPR:5 — Chemical is an agonist of the protein receptor "
                 "(keywords: agonist, mimics, activates receptor, binds and activates). "
                 "Use CPR:5 only when receptor binding/agonism is explicitly stated; use CPR:3 for general activation.\n"
                 "  CPR:6 — Chemical is an antagonist of the protein receptor "
                 "(keywords: antagonist, blocks receptor, receptor blocker). "
                 "Use CPR:6 only when receptor antagonism is explicit; use CPR:4 for general inhibition.\n"
                 "  CPR:9 — Chemical is a substrate of the protein, or the protein produces/metabolises the chemical "
                 "(keywords: substrate of, metabolised by, converted by, product of, catalyses)\n"
                 "  CPR:false — DEFAULT when no positive interaction is stated. "
                 "Use CPR:false whenever: (a) the chemical and protein are merely co-mentioned with no mechanism; "
                 "(b) the sentence is about study design, background, or methodology without asserting an interaction; "
                 "(c) interaction is explicitly denied or unclear. "
                 "Signal phrases for CPR:false: 'was studied', 'was measured', 'was detected', 'used in', "
                 "'reported', 'in the presence of', 'compared to', 'no significant effect', 'did not'. "
                 "IMPORTANT: do NOT skip the entity pair — if CPR:3–CPR:9 do not clearly fit, output CPR:false.")
    elif dataset_name == "GAD_BLURB":
        intro = ("Classify the relationship between the gene entity and the disease entity in the text below. "
                 "You MUST output exactly one relation for the entity pair — always choose either positive or negative. "
                 "entity_1 = the gene, entity_2 = the disease, "
                 "relation_type = one of the two codes below, EXACTLY as written.")
        markers = ("Genes are marked with @GENE$ and diseases with @DISEASE$. "
                   "Use exactly those marker strings as entity values, not any other text.")
        types = ("Relation type codes — choose the best fit:\n"
                 "  positive — The text states or implies that the gene is associated with, contributes to, "
                 "increases risk of, or causes the disease "
                 "(keywords: associated with, linked to, risk factor, mutation causes, predisposes to, "
                 "susceptibility gene, polymorphism in … disease)\n"
                 "  negative — The text states there is NO association between the gene and disease, "
                 "or the gene has a protective/inverse effect. "
                 "Use negative even when the sentence discusses the gene and disease together but explicitly "
                 "denies the association. "
                 "(keywords: not associated, no significant association, protective, reduces risk of, "
                 "inversely associated, failed to replicate, no evidence of)\n"
                 "IMPORTANT: you must always output one of these two types — never skip the entity pair.\n"
                 )
    elif dataset_name == "DDI_BLURB":
        intro = ("Classify the relation between every pair of drug entities in the text below. "
                 "You MUST output a relation for every drug pair — even when no pharmacological interaction "
                 "is stated (use DDI-false in that case). "
                 "entity_1 = the first drug, entity_2 = the second drug, "
                 "relation_type = one of the codes below, EXACTLY as written.")
        markers = ("Drug entities are marked with @DRUG$. "
                   "Use exactly that marker string as the entity value, not any drug name.")
        types = ("Relation type codes — first check DDI-mechanism/effect/advise/int; "
                 "if none clearly fits, output DDI-false.\n"
                 "  DDI-mechanism — The text describes the pharmacokinetic mechanism of the interaction: "
                 "one drug alters the absorption, distribution, metabolism (e.g. CYP enzymes), or excretion of the other "
                 "(keywords: inhibits metabolism of, induces CYP, increases/decreases plasma levels of, "
                 "reduces clearance of, bioavailability)\n"
                 "  DDI-effect — The text states that one drug increases or decreases the clinical effect "
                 "or toxicity of the other, without explaining the mechanism "
                 "(keywords: potentiates, enhances the effect of, increases toxicity of, reduces efficacy of)\n"
                 "  DDI-advise — The text gives a clinical recommendation about co-administration: "
                 "should not be used together, contraindicated, caution advised, dosage adjustment needed "
                 "(keywords: should not be combined, contraindicated with, avoid concomitant use, "
                 "monitor closely when used with)\n"
                 "  DDI-int — An interaction is mentioned but is too vague for any of the above "
                 "(keywords: interacts with, interaction between — with no further detail)\n"
                 "  DDI-false — DEFAULT when no pharmacological interaction is asserted. "
                 "Use DDI-false whenever: (a) the drugs are merely co-mentioned (e.g. listed as comparators, "
                 "study arms, concomitant medications, or controls); "
                 "(b) the sentence describes experimental co-administration without claiming an interaction; "
                 "(c) the text explicitly denies an interaction. "
                 "Signal phrases for DDI-false: 'was administered with', 'patients also received', "
                 "'compared to', 'in addition to', 'along with', 'no significant interaction', 'did not interact'. "
                 "IMPORTANT: do NOT skip the drug pair — if DDI-mechanism/effect/advise/int do not clearly fit, "
                 "output DDI-false.\n"
                 "")
    elif dataset_name == "EU-ADR_BioBERT":
        intro = ("Classify the relationship between the gene entity and the disease entity in the text below. "
                 "You MUST output exactly one relation for the entity pair — always choose either positive or negative. "
                 "entity_1 = the gene, entity_2 = the disease, "
                 "relation_type = one of the two codes below, EXACTLY as written.")
        markers = ("Genes are marked with @GENE$ and diseases with @DISEASE$. "
                   "Use exactly those marker strings as entity values, not any other text.")
        types = ("Relation type codes — choose the best fit:\n"
                 "  positive — The text states or implies that the gene is associated with, contributes to, "
                 "increases risk of, or causes the disease "
                 "(keywords: associated with, linked to, risk factor, mutation causes, predisposes to, "
                 "susceptibility gene)\n"
                 "  negative — The text states there is NO association between the gene and disease, "
                 "or the gene has a protective/inverse effect. "
                 "Use negative even when the sentence discusses the gene and disease together but explicitly "
                 "denies the association. "
                 "(keywords: not associated, no significant association, protective, reduces risk of, "
                 "inversely associated, failed to replicate, no evidence of)\n"
                 "IMPORTANT: you must always output one of these two types — never skip the entity pair.\n"
                 "")
    elif dataset_name == "SemEval2010_task8":
        intro = ("Classify the semantic relation between the two marked entities in the text below. "
                 "You MUST output exactly one relation — use Other if none of the named types clearly fits. "
                 "entity_1 must be the text inside [E1]...[/E1] and entity_2 the text inside [E2]...[/E2]. "
                 "relation_type must be one of the codes below, EXACTLY as written.")
        markers = ("The first entity is marked with [E1]...[/E1] and the second with [E2]...[/E2]. "
                   "Use the literal text inside those markers as the entity values.")
        types = ("Relation type codes — check each named type; if none clearly fits, use Other.\n"
                 "  [E1] is e1, [E2] is e2 in the direction notation:\n"
                 "  Cause-Effect(e1,e2) — e1 causes e2\n"
                 "  Cause-Effect(e2,e1) — e2 causes e1\n"
                 "  Component-Whole(e1,e2) — e1 is a component of e2\n"
                 "  Component-Whole(e2,e1) — e2 is a component of e1\n"
                 "  Content-Container(e1,e2) — e1 is contained in e2\n"
                 "  Content-Container(e2,e1) — e2 is contained in e1\n"
                 "  Entity-Destination(e1,e2) — e1 moves to e2\n"
                 "  Entity-Destination(e2,e1) — e2 moves to e1\n"
                 "  Entity-Origin(e1,e2) — e1 originates from e2\n"
                 "  Entity-Origin(e2,e1) — e2 originates from e1\n"
                 "  Instrument-Agency(e1,e2) — e1 is the instrument, e2 is the agent\n"
                 "  Instrument-Agency(e2,e1) — e2 is the instrument, e1 is the agent\n"
                 "  Member-Collection(e1,e2) — e1 is a member of e2\n"
                 "  Member-Collection(e2,e1) — e2 is a member of e1\n"
                 "  Message-Topic(e1,e2) — e1 is the message, e2 is the topic\n"
                 "  Message-Topic(e2,e1) — e2 is the message, e1 is the topic\n"
                 "  Product-Producer(e1,e2) — e1 is the product, e2 is the producer\n"
                 "  Product-Producer(e2,e1) — e2 is the product, e1 is the producer\n"
                 "  Other — DEFAULT when none of the above named relations clearly hold. "
                 "IMPORTANT: always output one relation — use Other rather than skipping.")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dedup_rule = ("Each unique (entity_1, entity_2, relation_type) triple must appear AT MOST ONCE "
                  "in the output list. Do not repeat the same triple.")

    cot_instruction = (
        "Before listing relations, reason step by step in the 'reasoning' field: "
        "identify the entities, note the key words or phrases that signal a relation, "
        "explicitly consider whether the text asserts a positive interaction or merely co-mentions the entities, "
        "and justify which relation type best fits (including the 'false'/'negative' type if no positive type holds). "
        "Only then fill in the relations list.")

    parts = [intro]
    if with_markers:
        parts.append(markers)
    parts.append(types)
    if class_prior:
        prior_lines = ["Empirical class distribution in this dataset (use as a calibration prior):"]
        for t, p in class_prior.items():
            prior_lines.append(f"  {t}: {p:.0%}")
        parts.append("\n".join(prior_lines))
    parts.append(dedup_rule)
    if use_cot:
        parts.append(cot_instruction)

    if shots:
        parts.append("\nHere are some examples of the expected output format:")
        for i, shot in enumerate(shots, 1):
            text = shot.text_with_entity_marker if with_markers else shot.text
            rels = [{"entity_1": shot.relation.entity_1,
                     "entity_2": shot.relation.entity_2,
                     "relation_type": shot.relation.relation_type}]
            parts.append(f"Example {i}:\nText: {text}\nOutput: {json.dumps(rels)}")

    return "\n".join(parts)


def get_binary_prompt(dataset_name: str, with_markers: bool) -> str:
    """Binary yes/no prompt for stage-1 of two-stage inference.

    Asks whether the entity pair has any specific (non-null) relation.
    The caller pairs this with a HasRelation response model.
    """
    if dataset_name == "chemprot_BLURB":
        question = (
            "Read the text below and decide: does it assert a direct chemical-protein "
            "interaction (activation, inhibition, agonism, antagonism, or a substrate/enzyme "
            "relationship) between @CHEMICAL$ and @GENE$?\n"
            "has_relation=true  → an interaction is explicitly stated.\n"
            "has_relation=false → they are merely co-mentioned with no interaction stated."
        )
        marker_note = "Chemicals are marked with @CHEMICAL$ and proteins with @GENE$."
    elif dataset_name == "DDI_BLURB":
        question = (
            "Read the text below and decide: does it assert a pharmacological interaction "
            "between the @DRUG$ entities (a mechanism, a clinical effect, a recommendation, "
            "or an explicit statement that they interact)?\n"
            "has_relation=true  → a pharmacological interaction is explicitly stated.\n"
            "has_relation=false → the drugs are merely co-mentioned, listed, or co-administered "
            "without claiming an effect."
        )
        marker_note = "Drug entities are marked with @DRUG$."
    elif dataset_name == "SemEval2010_task8":
        question = (
            "Read the text below and decide: does it express one of the 9 named semantic "
            "relations (Cause-Effect, Component-Whole, Content-Container, Entity-Destination, "
            "Entity-Origin, Instrument-Agency, Member-Collection, Message-Topic, Product-Producer) "
            "between [E1] and [E2]?\n"
            "has_relation=true  → one of the 9 named types clearly applies.\n"
            "has_relation=false → the relation is 'Other' (none of the 9 types fit)."
        )
        marker_note = "The first entity is marked with [E1]...[/E1] and the second with [E2]...[/E2]."
    elif dataset_name in ("GAD_BLURB", "EU-ADR_BioBERT"):
        question = (
            "Read the text below and decide: does the gene have a POSITIVE association "
            "with the disease (the gene contributes to, causes, or increases risk of it)?\n"
            "has_relation=true  → positive association.\n"
            "has_relation=false → negative association, null result, or protective effect."
        )
        marker_note = "Genes are marked with @GENE$ and diseases with @DISEASE$."
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    parts = [question]
    if with_markers:
        parts.append(marker_note)
    return "\n".join(parts)


# Relation types for which entity order is irrelevant (symmetric / non-directional).
# For these, (A, B, t) and (B, A, t) are treated as the same relation.
NON_DIRECTIONAL_TYPES: Set[str] = {
    "CPR:false",   # ChemProt — no interaction
    "DDI-false",   # DDI — no interaction
    "positive",    # GAD / EU-ADR — gene–disease association (symmetric)
    "negative",    # GAD / EU-ADR — no/inverse association (symmetric)
}


def _canonical(r: RelationSpec) -> Tuple[str, str, str]:
    """Return a canonical (entity_1, entity_2, relation_type) tuple.

    For non-directional relation types the entity pair is sorted so that
    swapped predictions still match the ground truth.
    """
    if r.relation_type in NON_DIRECTIONAL_TYPES:
        e1, e2 = (r.entity_1, r.entity_2) if r.entity_1 <= r.entity_2 else (r.entity_2, r.entity_1)
    else:
        e1, e2 = r.entity_1, r.entity_2
    return (e1, e2, r.relation_type)


def _canonical_ent(r: RelationSpec) -> Tuple[str, str]:
    """Return a canonical (entity_1, entity_2) pair, sorted for non-directional types."""
    if r.relation_type in NON_DIRECTIONAL_TYPES:
        return (r.entity_1, r.entity_2) if r.entity_1 <= r.entity_2 else (r.entity_2, r.entity_1)
    return (r.entity_1, r.entity_2)


class MatchCounter:
    """Accumulates two sets of P/R/F1 metrics across all experiment configurations.

    Match types
    -----------
    complete : entity_1, entity_2 *and* relation_type must all match.
    entities : entity_1 and entity_2 match regardless of relation_type.

    For non-directional relation types (see NON_DIRECTIONAL_TYPES) the entity
    pair is canonicalised before comparison, so swapped predictions still count
    as correct.
    """

    def __init__(self):
        self._results: List[dict] = []
        self._current_config: dict = {}
        self._reset_counters()

    def _reset_counters(self):
        # complete-match counters
        self._tp = self._fp = self._fn = 0
        self._complete_match_count = 0
        # entity-only counters
        self._tp_ent = self._fp_ent = self._fn_ent = 0
        self._ent_match_count = 0
        self._total_examples = 0
        self._failure_count = 0

    def start_experiment(self, model: str, temperature: float, dataset: str, mode: str,
                         with_markers: bool, backend: str = "instructor", use_cot: bool = False,
                         use_prior: bool = False, inference_mode: str = "standard",
                         use_confidence: bool = False):
        self._current_config = {
            "model": model, "temperature": temperature, "dataset": dataset,
            "mode": mode, "with_markers": with_markers, "backend": backend,
            "use_cot": use_cot, "use_prior": use_prior,
            "inference_mode": inference_mode, "use_confidence": use_confidence,
        }
        self._reset_counters()

    def _to_set(self, relations: List[RelationSpec]) -> Set[Tuple[str, str, str]]:
        """Full (entity_1, entity_2, relation_type) tuples, canonicalised for non-directional types."""
        return {_canonical(r) for r in relations}

    def _to_entity_set(self, relations: List[RelationSpec]) -> Set[Tuple[str, str]]:
        """Entity-pair-only tuples, canonicalised for non-directional types."""
        return {_canonical_ent(r) for r in relations}

    def record_failure(self) -> None:
        """Record one example where structured output could not be produced.
        Failures are counted separately and excluded from P/R/F1 computation."""
        self._failure_count += 1

    def update(self, ground_truth: List[RelationSpec], inferred: List[RelationSpec]):
        self._total_examples += 1

        # --- complete match ---
        truth_full    = self._to_set(ground_truth)
        inferred_full = self._to_set(inferred)
        if truth_full == inferred_full:
            self._complete_match_count += 1
        self._tp += len(truth_full & inferred_full)
        self._fp += len(inferred_full - truth_full)
        self._fn += len(truth_full - inferred_full)

        # --- entity-only match ---
        truth_ent    = self._to_entity_set(ground_truth)
        inferred_ent = self._to_entity_set(inferred)
        if truth_ent == inferred_ent:
            self._ent_match_count += 1
        self._tp_ent += len(truth_ent & inferred_ent)
        self._fp_ent += len(inferred_ent - truth_ent)
        self._fn_ent += len(truth_ent - inferred_ent)

    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    def finish_experiment(self):
        """Compute both sets of metrics and store the result row."""
        p, r, f1      = self._prf(self._tp, self._fp, self._fn)
        p_e, r_e, f1_e = self._prf(self._tp_ent, self._fp_ent, self._fn_ent)
        n = self._total_examples
        self._results.append({
            **self._current_config,
            "failures": self._failure_count,
            # complete-match metrics
            "tp": self._tp, "fp": self._fp, "fn": self._fn,
            "precision":          round(p,    4),
            "recall":             round(r,    4),
            "f1":                 round(f1,   4),
            "complete_match_acc": round(self._complete_match_count / n if n else 0, 4),
            # entity-only metrics
            "tp_ent": self._tp_ent, "fp_ent": self._fp_ent, "fn_ent": self._fn_ent,
            "precision_ent":  round(p_e,  4),
            "recall_ent":     round(r_e,  4),
            "f1_ent":         round(f1_e, 4),
            "acc_ent":        round(self._ent_match_count / n if n else 0, 4),
            "total_examples": n,
        })

    def save_to_csv(self, output_path: str | None = None):
        """Save all experiment results to a single CSV file."""
        if not self._results:
            return
        path = Path(output_path) if output_path else Path(__file__).resolve().parents[2] / "results" / "results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self._results[0].keys())
            writer.writeheader()
            writer.writerows(self._results)
        print(f"Results saved to: {path}")

    def print_summary(self):
        """Print a formatted table with both complete-match and entity-only metrics."""
        if not self._results:
            print("No results to display.")
            return
        c = {"ds": 16, "model": 30, "temp": 5, "mode": 20, "mk": 3, "bk": 10,
             "P": 6, "R": 6, "F": 6, "A": 6, "N": 5}
        hdr = (f"{'Dataset':<{c['ds']}} {'Model':<{c['model']}} {'Tmp':>{c['temp']}}"
               f" {'Mode':>{c['mode']}} {'Mk':>{c['mk']}} {'Backend':>{c['bk']}}"
               f"  {'P':>{c['P']}} {'R':>{c['R']}} {'F1':>{c['F']}} {'Acc':>{c['A']}}"
               f"  {'Pe':>{c['P']}} {'Re':>{c['R']}} {'Fe':>{c['F']}} {'Ae':>{c['A']}}"
               f"  {'N':>{c['N']}}")
        sep = "-" * len(hdr)
        print(f"\n{'EXPERIMENT SUMMARY  (full | ent)':^{len(hdr)}}")
        print(sep)
        print(hdr)
        print(sep)
        for r in self._results:
            mode_label = str(r['mode'])
            mk = "Y" if r['with_markers'] else "N"
            model_short = r['model'].split("/")[-1]
            print(f"{r['dataset']:<{c['ds']}} {model_short:<{c['model']}} {r['temperature']:>{c['temp']}}"
                  f" {mode_label:>{c['mode']}} {mk:>{c['mk']}} {r.get('backend','instructor'):>{c['bk']}}"
                  f"  {r['precision']:>{c['P']}.4f} {r['recall']:>{c['R']}.4f}"
                  f" {r['f1']:>{c['F']}.4f} {r['complete_match_acc']:>{c['A']}.4f}"
                  f"  {r.get('precision_ent', 0):>{c['P']}.4f} {r.get('recall_ent', 0):>{c['R']}.4f}"
                  f" {r.get('f1_ent', 0):>{c['F']}.4f} {r.get('acc_ent', 0):>{c['A']}.4f}"
                  f"  {r['total_examples']:>{c['N']}}")
        print(sep)
        print("  full = complete match (entity_1, entity_2, relation_type)")
        print("  ent  = entity-pair match only (entity_1, entity_2)")