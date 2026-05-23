"""
AIML-3 | Week 1 | AI Mental Health UK Project
Task   : Feature exploration & correlation analysis — APMS 2014
Dates  : 3–6 May 2026
Output : reports/correlation_heatmap.png + reports/feature_report.txt
Depends: aiml_w1_02_cleaning.py (apms_2014_cleaned.csv)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / server-safe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT       = Path(__file__).parent.resolve()
DATA_IN    = ROOT / "data" / "processed" / "apms_2014_cleaned.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HEATMAP_PATH = REPORT_DIR / "correlation_heatmap.png"
REPORT_PATH  = REPORT_DIR / "feature_report.txt"

# NHS-compliant colour palette (accessible, matches GOV.UK guidance)
NHS_BLUE   = "#005EB8"
NHS_GREEN  = "#009639"
NHS_DARK   = "#231F20"

# Target (outcome) columns
TARGET_COLS = ["any_cmd", "depression", "anxiety_gad", "phobia", "ocd", "cpd"]

# Feature (predictor) columns
FEATURE_COLS = ["age", "sex", "ethnicity", "imd_quintile", "region", "weight"]


# ─────────────────────────────────────────────
# 1.  Load
# ─────────────────────────────────────────────
def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded %d rows × %d cols", *df.shape)
    return df


# ─────────────────────────────────────────────
# 2.  Correlation matrix (all numeric columns)
# ─────────────────────────────────────────────
def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix across all numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(method="pearson")
    log.info("Computed %d × %d Pearson correlation matrix.", *corr.shape)
    return corr


# ─────────────────────────────────────────────
# 3.  Top feature–target correlations
# ─────────────────────────────────────────────
def top_feature_target_corr(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    """
    For each target column, rank feature columns by absolute Pearson correlation.
    Returns a tidy DataFrame of top-N correlations per target.
    """
    rows = []
    for target in target_cols:
        if target not in df.columns:
            continue
        for feat in feature_cols:
            if feat not in df.columns:
                continue
            r, p = stats.pearsonr(df[feat].dropna(), df[target].dropna())
            rows.append({
                "target"   : target,
                "feature"  : feat,
                "pearson_r": round(r, 4),
                "p_value"  : round(p, 6),
                "abs_r"    : abs(r),
            })

    result_df = (
        pd.DataFrame(rows)
        .sort_values(["target", "abs_r"], ascending=[True, False])
        .groupby("target")
        .head(top_n)
        .reset_index(drop=True)
    )
    return result_df


# ─────────────────────────────────────────────
# 4.  Disorder prevalence by demographic group
# ─────────────────────────────────────────────
def prevalence_by_group(
    df: pd.DataFrame,
    group_col: str,
    disorder_col: str = "any_cmd",
) -> pd.DataFrame:
    """Compute disorder prevalence (%) by a demographic grouping column."""
    if group_col not in df.columns or disorder_col not in df.columns:
        return pd.DataFrame()

    prev = (
        df.groupby(group_col)[disorder_col]
        .agg(["sum", "count"])
        .rename(columns={"sum": "cases", "count": "total"})
    )
    prev["prevalence_pct"] = (prev["cases"] / prev["total"] * 100).round(2)
    return prev.reset_index()


# ─────────────────────────────────────────────
# 5.  Plot: Full correlation heatmap
# ─────────────────────────────────────────────
def plot_correlation_heatmap(corr: pd.DataFrame, save_path: Path) -> None:
    n = len(corr)
    fig_size = max(10, n * 0.7)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    fig.patch.set_facecolor("white")

    mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.4,
        linecolor="#e0e0e0",
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
        ax=ax,
    )

    ax.set_title(
        "APMS 2014 — Feature Correlation Matrix",
        fontsize=14, fontweight="bold", color=NHS_DARK, pad=16,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Correlation heatmap saved → %s", save_path)


# ─────────────────────────────────────────────
# 6.  Plot: Feature–target bar chart
# ─────────────────────────────────────────────
def plot_feature_target_bars(
    corr_df: pd.DataFrame,
    save_path: Path,
) -> None:
    targets = corr_df["target"].unique()
    ncols   = 3
    nrows   = int(np.ceil(len(targets) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for i, target in enumerate(targets):
        ax  = axes[i]
        sub = corr_df[corr_df["target"] == target].sort_values("pearson_r")

        colors = [NHS_BLUE if r >= 0 else "#C00000" for r in sub["pearson_r"]]
        bars   = ax.barh(sub["feature"], sub["pearson_r"], color=colors, edgecolor="white")

        ax.axvline(0, color=NHS_DARK, linewidth=0.8, linestyle="--")
        ax.set_title(target, fontsize=11, fontweight="bold", color=NHS_DARK)
        ax.set_xlabel("Pearson r", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        for bar, r_val in zip(bars, sub["pearson_r"]):
            label_x = r_val + 0.005 if r_val >= 0 else r_val - 0.005
            ha       = "left"        if r_val >= 0 else "right"
            ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                    f"{r_val:.3f}", va="center", ha=ha, fontsize=8)

    # Hide unused subplots
    for j in range(len(targets), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Feature–Target Correlations (APMS 2014)",
                 fontsize=13, fontweight="bold", color=NHS_DARK, y=1.01)
    plt.tight_layout()
    bar_path = save_path.parent / "feature_target_bars.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Feature–target bar chart saved → %s", bar_path)


# ─────────────────────────────────────────────
# 7.  Write text report
# ─────────────────────────────────────────────
def write_report(
    corr_df: pd.DataFrame,
    prevalence_df: pd.DataFrame,
    report_path: Path,
) -> None:
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("APMS 2014 - Feature Exploration Report\n")
        f.write("AI Mental Health UK Project - AIML-3 Week 1\n")
        f.write("=" * 60 + "\n\n")

        f.write("-- Top Feature-Target Correlations --\n")
        f.write(corr_df.to_string(index=False))
        f.write("\n\n")

        f.write("-- any_cmd Prevalence by IMD Quintile --\n")
        if not prevalence_df.empty:
            f.write(prevalence_df.to_string(index=False))
        else:
            f.write("(data not available)\n")
        f.write("\n\n")

        f.write("Notes:\n")
        f.write("  - Pearson r assumes linearity; use point-biserial for binary targets.\n")
        f.write("  - Correlations >|0.3| considered noteworthy for NHS context.\n")
        f.write("  - IMD = Index of Multiple Deprivation (1=least deprived, 5=most).\n")

    log.info("Feature report saved → %s", report_path)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    log.info("─── AIML-3: Feature Exploration & Correlation ───")

    df = load(DATA_IN)

    # Limit to columns we care about for clarity
    all_cols = FEATURE_COLS + TARGET_COLS
    present  = [c for c in all_cols if c in df.columns]
    df_sub   = df[present].copy()

    # Correlation matrix
    corr = compute_correlation_matrix(df_sub)
    plot_correlation_heatmap(corr, HEATMAP_PATH)

    # Feature–target analysis
    feat_present   = [c for c in FEATURE_COLS if c in df_sub.columns]
    target_present = [c for c in TARGET_COLS  if c in df_sub.columns]
    corr_df = top_feature_target_corr(df_sub, feat_present, target_present, top_n=5)
    log.info("\n%s", corr_df.to_string())

    plot_feature_target_bars(corr_df, HEATMAP_PATH)

    # Prevalence breakdown
    prev_df = prevalence_by_group(df_sub, group_col="imd_quintile", disorder_col="any_cmd")
    if not prev_df.empty:
        log.info("\nPrevalence by IMD quintile:\n%s", prev_df.to_string())

    write_report(corr_df, prev_df, REPORT_PATH)
    log.info("─── Feature exploration complete ───")


if __name__ == "__main__":
    main()