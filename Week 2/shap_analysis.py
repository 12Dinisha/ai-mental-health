"""
AIML-3 — Week 2 SHAP Feature Importance
AI Mental Health UK · May 2026 Sprint

Uses:
- AIML-1 engineered demographic features
- AIML-2 PHQ/GAD engineered score features

Outputs:
    reports/shap_beeswarm.png
    reports/shap_waterfall.png
    reports/shap_bar.png
    reports/importance_ranking.csv
    reports/shap_summary.txt

Run:
    python "week 2/shap_analysis.py"
"""

# ============================================================================
# IMPORTS
# ============================================================================

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)

# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).parent.parent.resolve()

FEATURE_SET_V2 = ROOT / "data" / "processed" / "feature_set_v2.csv"
SCORE_FEATURES = ROOT / "data" / "processed" / "score_features.csv"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "baseline_lr.pkl"

TOP_N_FEATURES = 15

# ============================================================================
# CHECK DEPENDENCIES
# ============================================================================

def check_dependencies():

    required = [
        "shap",
        "sklearn",
        "numpy",
        "pandas",
        "matplotlib",
    ]

    missing = []

    for pkg in required:

        try:
            __import__(pkg)

        except ImportError:
            missing.append(pkg)

    if missing:

        log.error(f"Missing packages → {missing}")

        log.error(
            "Install using:\n"
            "pip install shap scikit-learn pandas numpy matplotlib"
        )

        raise SystemExit(1)

    log.info("All dependencies present.")


# ============================================================================
# REMOVE DUPLICATE COLUMNS
# ============================================================================

def remove_duplicate_columns(df):

    df = df.loc[:, ~df.columns.duplicated()]

    return df


# ============================================================================
# CLEAN DATAFRAME
# ============================================================================

def clean_dataframe(df):

    # remove duplicate rows
    df = df.drop_duplicates()

    # replace inf values
    df = df.replace([np.inf, -np.inf], np.nan)

    # remove empty rows
    df = df.dropna(how="all")

    # remove empty columns
    df = df.dropna(axis=1, how="all")

    # fill remaining nulls
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:

        median_val = df[col].median()

        df[col] = df[col].fillna(median_val)

    # non numeric
    non_numeric = df.select_dtypes(exclude=np.number).columns

    for col in non_numeric:

        df[col] = df[col].fillna("unknown")

    # remove duplicated columns
    df = remove_duplicate_columns(df)

    return df


# ============================================================================
# LOAD FEATURE MATRIX
# ============================================================================

def load_feature_matrix():

    frames = []

    # ------------------------------------------------------------------------
    # AIML-1
    # ------------------------------------------------------------------------

    if FEATURE_SET_V2.exists():

        log.info(f"Loading AIML-1 features → {FEATURE_SET_V2.name}")

        df1 = pd.read_csv(FEATURE_SET_V2)

        df1 = clean_dataframe(df1)

        demographic_features = [

            "age_band_code",
            "imd_proxy",
            "high_deprivation",
            "flag_depression",
            "flag_anxiety",
            "flag_comorbid_anx_dep",
            "comorbidity_count",
        ]

        available = [
            c for c in demographic_features
            if c in df1.columns
        ]

        log.info(f"AIML-1 Features Used → {available}")

        frames.append(df1[available])

    else:

        log.warning("feature_set_v2.csv missing")

    # ------------------------------------------------------------------------
    # AIML-2
    # ------------------------------------------------------------------------

    if SCORE_FEATURES.exists():

        log.info(f"Loading AIML-2 features → {SCORE_FEATURES.name}")

        df2 = pd.read_csv(SCORE_FEATURES)

        df2 = clean_dataframe(df2)

        score_features = [

            "phq9_score",
            "phq9_tier_code",
            "phq9_change_code",

            "gad7_score",
            "gad7_tier_code",
            "gad7_change_code",

            "severity_index",
            "high_risk",
        ]

        available = [
            c for c in score_features
            if c in df2.columns
        ]

        log.info(f"AIML-2 Features Used → {available}")

        frames.append(df2[available])

    else:

        log.warning("score_features.csv missing")

    # ------------------------------------------------------------------------
    # NO FEATURES
    # ------------------------------------------------------------------------

    if not frames:

        log.error("No feature datasets found.")

        raise SystemExit(1)

    # ------------------------------------------------------------------------
    # MERGE
    # ------------------------------------------------------------------------

    X = pd.concat(frames, axis=1)

    X = clean_dataframe(X)

    X = X.reset_index(drop=True)

    log.info(f"Feature Matrix Shape → {X.shape}")

    # ------------------------------------------------------------------------
    # CREATE TARGET
    # ------------------------------------------------------------------------

    y = (

        (
            X.get("phq9_score", 0) >= 10
        )

        |

        (
            X.get("gad7_score", 0) >= 10
        )

        |

        (
            X.get("high_risk", 0) == 1
        )

    ).astype(int)

    # fallback
    if y.nunique() < 2:

        log.warning("Only one class found. Creating synthetic labels.")

        rng = np.random.default_rng(42)

        y = pd.Series(

            rng.choice(
                [0, 1],
                size=len(X),
                p=[0.6, 0.4]
            )

        )

    log.info(f"Target Distribution → {y.value_counts().to_dict()}")

    return X, y


# ============================================================================
# TRAIN MODEL
# ============================================================================

def train_model(X, y):

    from sklearn.linear_model import LogisticRegression

    from sklearn.model_selection import train_test_split

    from sklearn.preprocessing import StandardScaler

    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_auc_score

    import pickle

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )

    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train)

    X_test_sc = scaler.transform(X_test)

    model = LogisticRegression(

        max_iter=2000,

        random_state=42,

        class_weight="balanced"

    )

    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)

    y_prob = model.predict_proba(X_test_sc)[:, 1]

    roc = roc_auc_score(y_test, y_prob)

    log.info(f"ROC-AUC → {roc:.4f}")

    log.info(
        "\n"
        + classification_report(y_test, y_pred)
    )

    # save model
    with open(MODEL_PATH, "wb") as f:

        pickle.dump(model, f)

    log.info(f"Model saved → {MODEL_PATH}")

    X_train_sc = pd.DataFrame(
        X_train_sc,
        columns=X.columns
    )

    X_test_sc = pd.DataFrame(
        X_test_sc,
        columns=X.columns
    )

    return model, X_train_sc, X_test_sc


# ============================================================================
# COMPUTE SHAP
# ============================================================================

def compute_shap(model, X_train, X_test):

    import shap

    from sklearn.cluster import KMeans

    log.info("Computing SHAP values...")

    X_train = clean_dataframe(X_train)

    X_test = clean_dataframe(X_test)

    # ------------------------------------------------------------------------
    # BACKGROUND DATA
    # ------------------------------------------------------------------------

    k = min(30, len(X_train))

    km = KMeans(

        n_clusters=k,

        random_state=42,

        n_init="auto"

    )

    km.fit(X_train)

    background = pd.DataFrame(

        km.cluster_centers_,

        columns=X_train.columns

    )

    # ------------------------------------------------------------------------
    # PREDICT FUNCTION
    # ------------------------------------------------------------------------

    def predict_fn(x):

        x_df = pd.DataFrame(

            x,

            columns=X_train.columns

        )

        return model.predict_proba(x_df)

    # ------------------------------------------------------------------------
    # EXPLAINER
    # ------------------------------------------------------------------------

    explainer = shap.KernelExplainer(

        predict_fn,

        background,

        link="logit"

    )

    sample_size = min(200, len(X_test))

    X_sample = X_test.iloc[:sample_size]

    shap_values = explainer.shap_values(

        X_sample,

        nsamples=100,

        silent=True

    )

    # =====================================================
    # FIX: Handle different output formats
    # =====================================================
    
    # Convert to numpy array
    shap_values = np.asarray(shap_values)

    # Handle list output from KernelExplainer (binary class)
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)

    # Handle 3D array: (samples, features, classes) → (samples, features)
    if shap_values.ndim == 3:
        # Take class 1 (probability of positive class)
        shap_values = shap_values[:, :, 1]
    elif shap_values.ndim == 2 and shap_values.shape[1] > X_sample.shape[1]:
        # Handle case where first dimension is classes
        if shap_values.shape[0] <= 2:
            shap_values = shap_values[1]  # Class 1

    # Ensure 2D array (samples x features)
    if shap_values.ndim > 2:
        shap_values = shap_values[:, :, 1]

    # Convert to 2D if still not
    if shap_values.ndim != 2:
        shap_values = np.reshape(shap_values, (X_sample.shape[0], -1))

    # Trim to correct number of features if needed
    if shap_values.shape[1] > X_sample.shape[1]:
        shap_values = shap_values[:, :X_sample.shape[1]]

    log.info(f"SHAP Shape → {shap_values.shape}")

    return shap_values, explainer, X_sample


# ============================================================================
# PLOTS
# ============================================================================

def plot_beeswarm(shap_values, X_sample):

    import shap

    out = REPORT_DIR / "shap_beeswarm.png"

    plt.figure(figsize=(10, 7))

    shap.summary_plot(

        shap_values,

        X_sample,

        max_display=TOP_N_FEATURES,

        show=False

    )

    plt.title("SHAP Beeswarm Plot")

    plt.tight_layout()

    plt.savefig(out, dpi=150)

    plt.close()

    log.info(f"Saved → {out}")


# ============================================================================
# PLOT WATERFALL (FIXED)
# ============================================================================

def plot_waterfall(shap_values, explainer, X_sample):

    import shap

    out = REPORT_DIR / "shap_waterfall.png"

    # =====================================================
    # FIX: Get values for first sample
    # =====================================================
    values = shap_values[0]

    # Ensure 1D array for waterfall
    if values.ndim > 1:
        values = values.flatten()

    # =====================================================
    # FIX: Handle expected_value properly
    # =====================================================
    base_val = explainer.expected_value

    if base_val is not None:
        # Convert to numpy array
        base_val = np.asarray(base_val)

        # Flatten if needed
        if base_val.ndim > 0:
            # For binary classification: use class 1 probability
            if base_val.shape[0] == 2:
                base_val = float(base_val[1])
            elif base_val.shape[0] == 1:
                base_val = float(base_val[0])
            else:
                base_val = float(base_val.flatten()[0])
        else:
            base_val = float(base_val)
    else:
        # Fallback: use mean of shap values as base
        base_val = 0.0

    # Get feature names
    feature_names = X_sample.columns.tolist()

    # Ensure values matches feature count
    if len(values) != len(feature_names):
        min_len = min(len(values), len(feature_names))
        values = values[:min_len]
        feature_names = feature_names[:min_len]
        log.warning(f"Trimmed values to {min_len} features")

    # Create Explanation object
    explanation = shap.Explanation(
        values=values,
        base_values=base_val,
        data=X_sample.iloc[0].values[:len(values)],
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 6))

    shap.waterfall_plot(
        explanation,
        max_display=TOP_N_FEATURES,
        show=False
    )

    plt.tight_layout()

    plt.savefig(out, dpi=150)

    plt.close()

    log.info(f"Saved → {out}")


def plot_bar(shap_values, X_sample):

    out = REPORT_DIR / "shap_bar.png"

    mean_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({

        "feature": X_sample.columns,

        "mean_abs_shap": mean_shap

    })

    importance_df = importance_df.sort_values(

        "mean_abs_shap",

        ascending=False

    )

    top = importance_df.head(TOP_N_FEATURES)

    plt.figure(figsize=(10, 6))

    plt.barh(

        top["feature"][::-1],

        top["mean_abs_shap"][::-1]

    )

    plt.xlabel("Mean |SHAP|")

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig(out, dpi=150)

    plt.close()

    log.info(f"Saved → {out}")

    csv_out = REPORT_DIR / "importance_ranking.csv"

    importance_df.to_csv(csv_out, index=False)

    log.info(f"Saved → {csv_out}")

    return importance_df


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def write_summary(importance_df, shap_values):

    out = REPORT_DIR / "shap_summary.txt"

    with open(out, "w", encoding="utf-8") as f:

        f.write("=" * 60 + "\n")

        f.write("AIML-3 SHAP SUMMARY\n")

        f.write("=" * 60 + "\n\n")

        f.write("Top Features\n\n")

        f.write(

            importance_df.head(15).to_string(index=False)

        )

        f.write("\n\n")

        f.write(f"SHAP Shape → {np.array(shap_values).shape}\n")

        f.write(f"Max SHAP → {np.abs(shap_values).max():.6f}\n")

        f.write(f"Mean SHAP → {np.abs(shap_values).mean():.6f}\n")

    log.info(f"Saved → {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():

    log.info("AIML-3 SHAP Explainability Pipeline Started")

    check_dependencies()

    # ------------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------------

    X, y = load_feature_matrix()

    # ------------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------------

    model, X_train, X_test = train_model(X, y)

    # ------------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------------

    shap_values, explainer, X_sample = compute_shap(

        model,

        X_train,

        X_test

    )

    # ------------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------------

    plot_beeswarm(

        shap_values,

        X_sample

    )

    plot_waterfall(

        shap_values,

        explainer,

        X_sample

    )

    importance_df = plot_bar(

        shap_values,

        X_sample

    )

    # ------------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------------

    write_summary(

        importance_df,

        shap_values

    )

    log.info("AIML-3 SHAP PIPELINE COMPLETED")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":

    main()