"""
AIML-4 | Week 1 | AI Mental Health UK Project
Task   : Baseline ML model — Logistic Regression on APMS 2014
Dates  : 5–9 May 2026
Output : models/baseline_lr.pkl + reports/baseline_model_report.txt
Depends: aiml_w1_02_cleaning.py (apms_2014_cleaned.csv)
"""

import logging
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
MODEL_DIR  = ROOT / "models"
REPORT_DIR = ROOT / "reports"

for d in [MODEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH  = MODEL_DIR  / "baseline_lr.pkl"
REPORT_PATH = REPORT_DIR / "baseline_model_report.txt"
ROC_PATH    = REPORT_DIR / "roc_curve.png"
CM_PATH     = REPORT_DIR / "confusion_matrix.png"

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
TARGET     = "any_cmd"                              # primary binary outcome
FEATURES   = ["age", "sex", "ethnicity", "imd_quintile", "region"]  # adjust as needed

TEST_SIZE  = 0.20
RANDOM_STATE = 42
CV_FOLDS   = 5

LR_PARAMS = {
    "C"           : 1.0,        # inverse regularisation strength
    "max_iter"    : 1000,
    "solver"      : "lbfgs",
    "class_weight": "balanced", # handles class imbalance (CMD ~17% prevalence)
    "random_state": RANDOM_STATE,
}

# ─────────────────────────────────────────────
# 1.  Load & prepare
# ─────────────────────────────────────────────
def load_and_prepare(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded %d rows × %d cols from %s", *df.shape, path.name)

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Drop rows where target is null
    mask = y.notna()
    X, y = X[mask], y[mask]
    y = y.astype(int)

    log.info("Feature matrix: %d rows × %d cols", *X.shape)
    log.info("Target distribution:\n%s", y.value_counts().to_string())
    log.info("CMD prevalence: %.1f%%", y.mean() * 100)

    return X, y


# ─────────────────────────────────────────────
# 2.  Train / test split
# ─────────────────────────────────────────────
def split(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    log.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 3.  Build pipeline & train
# ─────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    """
    StandardScaler + LogisticRegression in an sklearn Pipeline.
    Pipeline prevents data leakage by fitting scaler only on train set.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(**LR_PARAMS)),
    ])


def train(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    log.info("Training Logistic Regression baseline …")
    pipeline.fit(X_train, y_train)
    log.info("Training complete.")
    return pipeline


# ─────────────────────────────────────────────
# 4.  Evaluate
# ─────────────────────────────────────────────
def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred      = pipeline.predict(X_test)
    y_prob      = pipeline.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="binary")
    auc  = roc_auc_score(y_test, y_prob)

    log.info("── Hold-out test metrics ──")
    log.info("  Accuracy : %.4f", acc)
    log.info("  F1 Score : %.4f", f1)
    log.info("  ROC-AUC  : %.4f", auc)
    log.info("\n%s", classification_report(y_test, y_pred, target_names=["No CMD", "CMD"]))

    return {
        "accuracy" : acc,
        "f1"       : f1,
        "roc_auc"  : auc,
        "y_pred"   : y_pred,
        "y_prob"   : y_prob,
        "report"   : classification_report(y_test, y_pred, target_names=["No CMD", "CMD"]),
    }


def cross_validate_model(pipeline: Pipeline, X, y) -> dict:
    """5-fold stratified cross-validation for more robust estimates."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        return_train_score=True,
    )
    log.info("── %d-fold CV metrics ──", CV_FOLDS)
    for metric in ["accuracy", "f1", "roc_auc"]:
        scores = cv_results[f"test_{metric}"]
        log.info("  %-10s: %.4f ± %.4f", metric, scores.mean(), scores.std())

    return cv_results


# ─────────────────────────────────────────────
# 5.  Coefficients inspection
# ─────────────────────────────────────────────
def inspect_coefficients(pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Extract and rank logistic regression coefficients (log-odds)."""
    lr_step = pipeline.named_steps["lr"]
    coefs   = lr_step.coef_[0]

    coef_df = pd.DataFrame({
        "feature"  : feature_names,
        "log_odds" : coefs,
        "odds_ratio": np.exp(coefs),
    }).sort_values("log_odds", key=abs, ascending=False).reset_index(drop=True)

    log.info("── Feature Coefficients (log-odds) ──\n%s", coef_df.to_string())
    return coef_df


# ─────────────────────────────────────────────
# 6.  Plots
# ─────────────────────────────────────────────
def plot_roc(pipeline, X_test, y_test, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, color="#005EB8")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
    ax.set_title("ROC Curve — Logistic Regression Baseline\n(APMS 2014 · any_cmd)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("ROC curve saved → %s", save_path)


def plot_confusion_matrix(y_test, y_pred, save_path: Path) -> None:
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("white")

    disp = ConfusionMatrixDisplay(cm, display_labels=["No CMD", "CMD"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — LR Baseline", fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Confusion matrix saved → %s", save_path)


# ─────────────────────────────────────────────
# 7.  Save model & report
# ─────────────────────────────────────────────
def save_model(pipeline: Pipeline, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    log.info("Model saved → %s", path)


def save_report(
    metrics   : dict,
    cv_results: dict,
    coef_df   : pd.DataFrame,
    path      : Path,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("AI Mental Health UK - Baseline Model Report\n")
        f.write("Task: AIML-4 | Week 1\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")

        f.write("-- Model --\n")
        f.write(f"  Algorithm : Logistic Regression (sklearn)\n")
        f.write(f"  Target    : {TARGET}  (1 = CMD present)\n")
        f.write(f"  Features  : {FEATURES}\n")
        f.write(f"  Test size : {TEST_SIZE:.0%}\n\n")

        f.write("-- Hold-out Test Metrics --\n")
        f.write(f"  Accuracy : {metrics['accuracy']:.4f}\n")
        f.write(f"  F1 Score : {metrics['f1']:.4f}\n")
        f.write(f"  ROC-AUC  : {metrics['roc_auc']:.4f}\n\n")

        f.write("-- Classification Report --\n")
        f.write(metrics["report"] + "\n")

        f.write(f"-- {CV_FOLDS}-Fold Cross-Validation --\n")
        for m in ["accuracy", "f1", "roc_auc"]:
            s = cv_results[f"test_{m}"]
            f.write(f"  {m:<12}: {s.mean():.4f} +/- {s.std():.4f}\n")
        f.write("\n")

        f.write("-- Feature Coefficients (Log-Odds) --\n")
        f.write(coef_df.to_string(index=False) + "\n\n")

        f.write("-- Notes --\n")
        f.write("  - class_weight='balanced' used to handle ~17% CMD prevalence.\n")
        f.write("  - Logistic Regression is interpretable - suitable for NHS clinical context.\n")
        f.write("  - Next step (Week 2): feature engineering & tree-based models.\n")

    log.info("Model report saved → %s", path)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    log.info("─── AIML-4: Baseline Logistic Regression Model ───")

    X, y = load_and_prepare(DATA_IN)
    X_train, X_test, y_train, y_test = split(X, y)

    pipeline = build_pipeline()
    pipeline = train(pipeline, X_train, y_train)

    # Hold-out evaluation
    metrics = evaluate(pipeline, X_test, y_test)

    # Cross-validation
    cv_results = cross_validate_model(pipeline, X, y)

    # Coefficients
    coef_df = inspect_coefficients(pipeline, FEATURES)

    # Plots
    plot_roc(pipeline, X_test, y_test, ROC_PATH)
    plot_confusion_matrix(y_test, metrics["y_pred"], CM_PATH)

    # Save model & report
    save_model(pipeline, MODEL_PATH)
    save_report(metrics, cv_results, coef_df, REPORT_PATH)

    log.info("─── Baseline model complete. ───")
    log.info("    Model  → %s", MODEL_PATH)
    log.info("    Report → %s", REPORT_PATH)
    log.info("    ROC    → %s", ROC_PATH)
    log.info("    CM     → %s", CM_PATH)


if __name__ == "__main__":
    main()