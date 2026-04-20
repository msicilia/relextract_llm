import csv
import json
from pathlib import Path
from pydantic import BaseModel
from typing import List, Set, Tuple

# Fixed: Changed 'def' to 'class' so these function as Pydantic models
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
    relations: List[RelationSpec]

# Canonical relation types for each dataset, in a stable order.
DATASET_RELATION_TYPES: dict[str, List[str]] = {
    "chemprot_BLURB":  ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "CPR:false"],
    "GAD_BLURB":       ["positive", "negative"],
    "DDI_BLURB":       ["DDI-advise", "DDI-effect", "DDI-int", "DDI-mechanism", "DDI-false"],
    "EU-ADR_BioBERT":  ["positive", "negative"],
}


def get_relation_types(dataset_name: str) -> List[str]:
    """Return the ordered list of relation type codes for a dataset."""
    if dataset_name not in DATASET_RELATION_TYPES:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_RELATION_TYPES[dataset_name]


def get_relations(dataset_name: str) -> List[RelationExample]:
    """Load all JSONL examples from a dataset, merging lines that share the same text."""
    base_path = Path(__file__).resolve().parents[2] / "data" / dataset_name
    print(f"Loading relations from dataset: {dataset_name}")

    # Use text as key to merge entity pairs from different lines into one example
    by_text: dict[str, RelationExample] = {}

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
                        by_text[text].relations.extend(specs)
                    else:
                        by_text[text] = RelationExample(
                            id=entry["id"],
                            text=text,
                            text_with_entity_marker=entry["text_with_entity_marker"],
                            text_with_typed_entity_marker=entry["text_with_typed_entity_marker"],
                            relations=specs
                        )
                except json.JSONDecodeError as e:
                    print(f"Error parsing {file_path.name} at line {line_num}: {e}")

    # Deduplicate relations within each example.  The same text can appear in
    # multiple fold files (e.g. EU-ADR 10-fold CV), causing identical relation
    # triples to be appended repeatedly during the merge above.
    for example in by_text.values():
        seen: set = set()
        unique = []
        for r in example.relations:
            key = (r.entity_1, r.entity_2, r.relation_type)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        example.relations = unique

    return list(by_text.values())

def get_prompt_instructions(dataset_name: str, with_markers: bool,
                            shots: List["RelationExample"] | None = None) -> str:
    """Returns a dataset-specific prompt for relation extraction, optionally with few-shot examples."""
    if dataset_name == "chemprot_BLURB":
        intro = ("Extract chemical-protein interactions from the text below. "
                 "Only extract interactions explicitly mentioned in the text. "
                 "Return a list of interactions; each must have entity_1 = the chemical, "
                 "entity_2 = the protein/gene, and relation_type = one of the codes below. "
                 "Use the relation_type code string EXACTLY as written.")
        markers = ("Chemicals are marked with @CHEMICAL$ and proteins with @GENE$. "
                   "Use exactly those marker strings as entity values, not any other text.")
        types = ("Relation type codes — choose the best fit:\n"
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
                 "  CPR:false — The chemical and protein are mentioned together but NO interaction is stated. "
                 "Use CPR:false when the text explicitly denies an interaction or merely co-mentions the entities.")
    elif dataset_name == "GAD_BLURB":
        intro = ("Extract gene-disease associations from the text below. "
                 "Only extract associations explicitly mentioned in the text. "
                 "Return a list of associations; each must have entity_1 = the gene, "
                 "entity_2 = the disease, and relation_type = one of the codes below. "
                 "Use the relation_type string EXACTLY as written.")
        markers = ("Genes are marked with @GENE$ and diseases with @DISEASE$. "
                   "Use exactly those marker strings as entity values, not any other text.")
        types = ("Relation type codes — choose the best fit:\n"
                 "  positive — The text states or implies that the gene is associated with, contributes to, "
                 "increases risk of, or causes the disease "
                 "(keywords: associated with, linked to, risk factor, mutation causes, predisposes to, "
                 "susceptibility gene, polymorphism in … disease)\n"
                 "  negative — The text states there is NO association, or the gene has a protective/inverse "
                 "effect against the disease "
                 "(keywords: not associated, no significant association, protective, reduces risk of, "
                 "inversely associated)\n"
                 "When in doubt between the two, prefer 'positive' only if an association is clearly stated.")
    elif dataset_name == "DDI_BLURB":
        intro = ("Extract drug-drug interactions from the text below. "
                 "Only extract interactions explicitly mentioned in the text. "
                 "Return a list of interactions; each must have entity_1 = the first drug, "
                 "entity_2 = the second drug, and relation_type = one of the codes below. "
                 "Use the relation_type code string EXACTLY as written.")
        markers = ("Drug entities are marked with @DRUG$. "
                   "Use exactly that marker string as the entity value, not any drug name.")
        types = ("Relation type codes — choose the MOST SPECIFIC that fits:\n"
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
                 "  DDI-false — No interaction is stated; the drugs are only co-mentioned. "
                 "Use DDI-false when the text explicitly says there is no interaction, or when the drugs appear "
                 "in the same sentence purely for context (e.g. clinical trial co-administration without interaction claim).\n"
                 "Priority order when unsure: DDI-mechanism > DDI-effect > DDI-advise > DDI-int > DDI-false.")
    elif dataset_name == "EU-ADR_BioBERT":
        intro = ("Extract gene-disease associations from the text below. "
                 "Only extract associations explicitly mentioned in the text. "
                 "Return a list of associations; each must have entity_1 = the gene, "
                 "entity_2 = the disease, and relation_type = one of the codes below. "
                 "Use the relation_type string EXACTLY as written.")
        markers = ("Genes are marked with @GENE$ and diseases with @DISEASE$. "
                   "Use exactly those marker strings as entity values, not any other text.")
        types = ("Relation type codes — choose the best fit:\n"
                 "  positive — The text states or implies that the gene is associated with, contributes to, "
                 "increases risk of, or causes the disease "
                 "(keywords: associated with, linked to, risk factor, mutation causes, predisposes to, "
                 "susceptibility gene)\n"
                 "  negative — The text states there is NO association, or the gene has a protective/inverse "
                 "effect against the disease "
                 "(keywords: not associated, no significant association, protective, reduces risk of, "
                 "inversely associated)\n"
                 "When in doubt between the two, prefer 'positive' only if an association is clearly stated.")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dedup_rule = ("Each unique (entity_1, entity_2, relation_type) triple must appear AT MOST ONCE "
                  "in the output list. Do not repeat the same triple.")

    parts = [intro]
    if with_markers:
        parts.append(markers)
    parts.append(types)
    parts.append(dedup_rule)

    if shots:
        parts.append("\nHere are some examples of the expected output format:")
        for i, shot in enumerate(shots, 1):
            text = shot.text_with_entity_marker if with_markers else shot.text
            rels = [{"entity_1": r.entity_1, "entity_2": r.entity_2, "relation_type": r.relation_type}
                    for r in shot.relations]
            parts.append(f"Example {i}:\nText: {text}\nOutput: {json.dumps(rels)}")

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

    def start_experiment(self, model: str, temperature: float, dataset: str, mode: str,
                         with_markers: bool, backend: str = "instructor"):
        self._current_config = {"model": model, "temperature": temperature, "dataset": dataset,
                                "mode": mode, "with_markers": with_markers, "backend": backend}
        self._reset_counters()

    def _to_set(self, relations: List[RelationSpec]) -> Set[Tuple[str, str, str]]:
        """Full (entity_1, entity_2, relation_type) tuples, canonicalised for non-directional types."""
        return {_canonical(r) for r in relations}

    def _to_entity_set(self, relations: List[RelationSpec]) -> Set[Tuple[str, str]]:
        """Entity-pair-only tuples, canonicalised for non-directional types."""
        return {_canonical_ent(r) for r in relations}

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
            # complete-match metrics (original column names kept for backward compat)
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