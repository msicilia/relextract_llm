# relextract-llm

Benchmarks LLM-based relation extraction on biomedical text using locally-hosted models via [Ollama](https://ollama.com) and structured outputs via [Instructor](https://github.com/jxnl/instructor).

## Datasets

- **ChemProt** — chemical-protein interactions (CPR:3/4/5/6/9)
- **GAD** — gene-disease associations (positive/negative)

Both are from [BLURB](https://microsoft.github.io/BLURB/) and live under `data/`.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Ollama must be running locally with the target models pulled (e.g. `ollama pull mistral`).

## Running

```bash
python -m relextract_llm.experiment
```

Runs a grid of 2 datasets × 2 models × 3 temperatures. Results (precision, recall, F1) are written to `results/` as CSV files, one per configuration.

## Models evaluated

- `MedAIBase/MedGemma1.5:4b`
- `mistral`
