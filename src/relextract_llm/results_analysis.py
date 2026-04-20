"""
Analysis of LLM relation extraction experiment results.

Reads results/results.csv and produces:
  1. Best overall configuration (highest F1)
  2. Best model overall (mean F1 across all configs)
  3. Effect of temperature on F1
  4. Effect of mode (zero-shot vs few-shot) on F1
  5. Effect of entity markers on F1
  6. Dataset-level differences with Kruskal-Wallis significance test
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats


RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "results.csv"


def load_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["mode_label"] = df["mode"].apply(lambda m: "zero-shot" if m == 0 else f"few-shot-{m}")
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")


def fmt_row(row: pd.Series) -> str:
    model_short = row["model"].split("/")[-1]
    return (
        f"  dataset={row['dataset']:<18}  model={model_short:<25}"
        f"  temp={row['temperature']}  mode={row['mode_label']:<12}"
        f"  markers={'yes' if row['with_markers'] else 'no '}"
        f"  P={row['precision']:.3f}  R={row['recall']:.3f}  F1={row['f1']:.3f}"
    )


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def best_configuration(df: pd.DataFrame) -> None:
    section("1. BEST CONFIGURATION (highest F1)")
    top = df.loc[df["f1"].idxmax()]
    print(fmt_row(top))
    print(f"\n  Top-5 configurations by F1:")
    top5 = df.nlargest(5, "f1")
    for _, row in top5.iterrows():
        print(fmt_row(row))


def best_model(df: pd.DataFrame) -> None:
    section("2. BEST MODEL OVERALL")
    model_stats = (
        df.groupby("model")["f1"]
        .agg(mean="mean", median="median", std="std", n="count")
        .sort_values("mean", ascending=False)
    )
    model_stats.index = model_stats.index.str.split("/").str[-1]
    print(model_stats.round(4).to_string())

    # Mann-Whitney U between models (if exactly 2)
    models = df["model"].unique()
    if len(models) == 2:
        subsection("Mann-Whitney U test (model comparison)")
        a = df[df["model"] == models[0]]["f1"].dropna()
        b = df[df["model"] == models[1]]["f1"].dropna()
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  {models[0].split('/')[-1]} vs {models[1].split('/')[-1]}")
        print(f"  U={stat:.2f},  p={p:.4f}  {'(significant at 0.05)' if p < 0.05 else '(not significant)'}")


def temperature_effect(df: pd.DataFrame) -> None:
    section("3. EFFECT OF TEMPERATURE")
    temp_stats = (
        df.groupby("temperature")["f1"]
        .agg(mean="mean", median="median", std="std", n="count")
        .sort_values("mean", ascending=False)
    )
    print(temp_stats.round(4).to_string())

    temps = df["temperature"].unique()
    if len(temps) >= 3:
        subsection("Kruskal-Wallis test across temperatures")
        groups = [df[df["temperature"] == t]["f1"].dropna().values for t in sorted(temps)]
        stat, p = stats.kruskal(*groups)
        print(f"  H={stat:.3f},  p={p:.4f}  {'(significant at 0.05)' if p < 0.05 else '(not significant)'}")
    elif len(temps) == 2:
        subsection("Mann-Whitney U test (temperature comparison)")
        a = df[df["temperature"] == temps[0]]["f1"].dropna()
        b = df[df["temperature"] == temps[1]]["f1"].dropna()
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  U={stat:.2f},  p={p:.4f}  {'(significant at 0.05)' if p < 0.05 else '(not significant)'}")

    subsection("Mean F1 per temperature, broken down by model")
    pivot = df.pivot_table(values="f1", index="temperature", columns="model", aggfunc="mean")
    pivot.columns = pivot.columns.str.split("/").str[-1]
    print(pivot.round(4).to_string())


def mode_effect(df: pd.DataFrame) -> None:
    section("4. EFFECT OF MODE (zero-shot vs few-shot)")
    mode_stats = (
        df.groupby("mode_label")["f1"]
        .agg(mean="mean", median="median", std="std", n="count")
        .sort_values("mean", ascending=False)
    )
    print(mode_stats.round(4).to_string())

    modes = df["mode"].unique()
    if len(modes) >= 2:
        subsection("Kruskal-Wallis test across modes")
        groups = [df[df["mode"] == m]["f1"].dropna().values for m in sorted(modes)]
        stat, p = stats.kruskal(*groups)
        print(f"  H={stat:.3f},  p={p:.4f}  {'(significant at 0.05)' if p < 0.05 else '(not significant)'}")

    subsection("Mean F1 per mode, broken down by dataset")
    pivot = df.pivot_table(values="f1", index="mode_label", columns="dataset", aggfunc="mean")
    print(pivot.round(4).to_string())


def markers_effect(df: pd.DataFrame) -> None:
    section("5. EFFECT OF ENTITY MARKERS")
    marker_stats = (
        df.groupby("with_markers")["f1"]
        .agg(mean="mean", median="median", std="std", n="count")
    )
    marker_stats.index = marker_stats.index.map({True: "with_markers", False: "without_markers"})
    print(marker_stats.round(4).to_string())

    if df["with_markers"].nunique() == 2:
        subsection("Mann-Whitney U test (markers vs no markers)")
        a = df[df["with_markers"] == True]["f1"].dropna()
        b = df[df["with_markers"] == False]["f1"].dropna()
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  U={stat:.2f},  p={p:.4f}  {'(significant at 0.05)' if p < 0.05 else '(not significant)'}")


def dataset_differences(df: pd.DataFrame) -> None:
    section("6. DATASET DIFFERENCES")
    dataset_stats = (
        df.groupby("dataset")["f1"]
        .agg(mean="mean", median="median", std="std", n="count")
        .sort_values("mean", ascending=False)
    )
    print(dataset_stats.round(4).to_string())

    datasets = df["dataset"].unique()
    if len(datasets) >= 3:
        subsection("Kruskal-Wallis test across datasets")
        groups = [df[df["dataset"] == d]["f1"].dropna().values for d in datasets]
        stat, p = stats.kruskal(*groups)
        print(f"  H={stat:.3f},  p={p:.4f}  {'(significant at 0.05)' if p < 0.05 else '(not significant)'}")

        if p < 0.05:
            subsection("Pairwise Mann-Whitney U tests (Bonferroni corrected)")
            pairs = [(datasets[i], datasets[j])
                     for i in range(len(datasets)) for j in range(i + 1, len(datasets))]
            n_comparisons = len(pairs)
            for d1, d2 in pairs:
                a = df[df["dataset"] == d1]["f1"].dropna()
                b = df[df["dataset"] == d2]["f1"].dropna()
                stat_u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
                p_adj = min(p_u * n_comparisons, 1.0)
                sig = "(significant)" if p_adj < 0.05 else "(not significant)"
                print(f"  {d1:<20} vs {d2:<20}  U={stat_u:.1f}  p_adj={p_adj:.4f}  {sig}")
    elif len(datasets) == 2:
        subsection("Mann-Whitney U test (dataset comparison)")
        a = df[df["dataset"] == datasets[0]]["f1"].dropna()
        b = df[df["dataset"] == datasets[1]]["f1"].dropna()
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  {datasets[0]} vs {datasets[1]}")
        print(f"  U={stat:.2f},  p={p:.4f}  {'(significant at 0.05)' if p < 0.05 else '(not significant)'}")

    subsection("Mean F1 per dataset, broken down by model")
    pivot = df.pivot_table(values="f1", index="dataset", columns="model", aggfunc="mean")
    pivot.columns = pivot.columns.str.split("/").str[-1]
    print(pivot.round(4).to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"Error: results file not found at {RESULTS_PATH}", file=sys.stderr)
        sys.exit(1)

    df = load_results()
    print(f"Loaded {len(df)} experiment configurations from {RESULTS_PATH}")
    print(f"  Models:       {sorted(df['model'].unique())}")
    print(f"  Datasets:     {sorted(df['dataset'].unique())}")
    print(f"  Temperatures: {sorted(df['temperature'].unique())}")
    print(f"  Modes:        {sorted(df['mode'].unique())} (0=zero-shot, >0=few-shot N)")
    print(f"  F1 range:     {df['f1'].min():.3f} – {df['f1'].max():.3f}")

    best_configuration(df)
    best_model(df)
    temperature_effect(df)
    mode_effect(df)
    markers_effect(df)
    dataset_differences(df)

    print(f"\n{'=' * 60}\n  Done.\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
