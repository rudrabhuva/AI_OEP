import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — saves plots instead of showing them
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "student_productivity_distraction_dataset_20000.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ann_procrastination_model.pkl")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Helper functions ───────────────────────────────────────────────────────────

def load_and_preprocess(data_path: str = DATA_PATH):
    """Load CSV, clean, split into train/test and select features.

    Returns
    -------
    X_train, X_test, y_train, y_test, selected_features, correlations, removed_features
    """
    target = "productivity_score"

    df = pd.read_csv(data_path)
    df = df.dropna(subset=[target]).copy()
    df = df.drop_duplicates()

    # Drop ID-like columns
    id_cols = [c for c in df.columns if c.lower() == "student_id" or c.lower().endswith("_id")]
    df = df.drop(columns=id_cols, errors="ignore")

    X = df.drop(columns=[target])
    y = df[target].astype(float)

    # Remove constant / all-NaN columns, then rows with any missing value
    X = X.loc[:, X.nunique(dropna=False) > 1]
    X = X.dropna(axis=1, how="all")
    valid = X.notna().all(axis=1)
    X, y = X.loc[valid], y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Feature selection via correlation (training data only — no data leakage)
    numeric_train = X_train.select_dtypes(include=np.number).copy()
    numeric_train[target] = y_train
    correlations = numeric_train.corr()[target].drop(target)
    correlations = correlations.replace([np.inf, -np.inf], np.nan).dropna()

    threshold       = 0.10
    selected_numeric = correlations[correlations.abs() >= threshold].index.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    selected_features = list(dict.fromkeys(selected_numeric + categorical_cols))

    if not selected_features:
        raise ValueError("No features selected. Reduce the correlation threshold.")

    removed_features = [c for c in X.columns if c not in selected_features]

    X_train = X_train[selected_features]
    X_test  = X_test[selected_features]

    return X_train, X_test, y_train, y_test, selected_features, correlations, removed_features


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Build and return an untrained sklearn Pipeline (preprocessor + ANN)."""
    numeric_features     = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("numeric",     StandardScaler(),                      numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("ann", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            learning_rate_init=0.001,
            max_iter=300,
            activation="relu",
            solver="adam",
            early_stopping=True,
            random_state=42,
        )),
    ])


def save_correlation_heatmap(correlations: pd.Series, save_path: str = None) -> str:
    """Save correlation heatmap to *save_path* and return the path."""
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, max(5, len(correlations) * 0.35)))
    sns.heatmap(correlations.to_frame(), annot=True, cmap="coolwarm",
                center=0, fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation with Productivity Score")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def save_actual_vs_predicted(y_test, y_pred, save_path: str = None) -> str:
    """Save actual-vs-predicted scatter plot and return the path."""
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "actual_vs_predicted.png")

    low  = min(float(y_test.min()), float(y_pred.min()))
    high = max(float(y_test.max()), float(y_pred.max()))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, edgecolors="none")
    ax.plot([low, high], [low, high], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Productivity Score")
    ax.set_ylabel("Predicted Productivity Score")
    ax.set_title("ANN: Actual vs Predicted Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def train_and_evaluate(data_path: str = DATA_PATH, model_path: str = MODEL_PATH) -> dict:
    """
    Full pipeline: load → preprocess → train → evaluate → save model.

    Returns a metrics dict (r2, cv_mean, cv_scores, outliers, paths, …).
    """
    print("=" * 55)
    print("  ANN — Student Productivity Score Predictor")
    print("=" * 55)

    # 1. Load & preprocess
    print("\n[1/5] Loading and preprocessing data …")
    X_train, X_test, y_train, y_test, selected_features, correlations, removed_features = \
        load_and_preprocess(data_path)

    print(f"      Rows (train/test) : {len(X_train)} / {len(X_test)}")
    print(f"      Selected features : {selected_features}")
    print(f"      Removed features  : {removed_features}")

    # 2. Correlation heatmap
    print("\n[2/5] Saving correlation heatmap …")
    heatmap_path = save_correlation_heatmap(correlations)
    print(f"      Saved → {heatmap_path}")

    print("\nCorrelation with Target (sorted by |r|):")
    print(correlations.sort_values(key=abs, ascending=False).to_string())

    # 3. Train
    print("\n[3/5] Training ANN …")
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    print("      Training complete.")

    # 4. Evaluate
    print("\n[4/5] Evaluating …")
    y_pred   = model.predict(X_test)
    r2       = float(r2_score(y_test, y_pred))
    r2_pct   = r2 * 100

    cv        = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

    target_z = (y_test - y_test.mean()) / y_test.std(ddof=0)
    outliers = int((target_z.abs() > 3).sum())

    print(f"\n  R2 Score   : {r2:.4f}  ({r2_pct:.2f} %)")
    print(f"  CV Folds   : {np.round(cv_scores, 4)}")
    print(f"  Mean CV R2 : {cv_scores.mean():.4f}")
    print(f"  Outliers   : {outliers}")

    avp_path = save_actual_vs_predicted(y_test, y_pred)
    print(f"\n      Actual vs Predicted plot → {avp_path}")

    # 5. Save model
    print("\n[5/5] Saving model …")
    joblib.dump({"model": model, "features": selected_features}, model_path)
    print(f"      Saved → {model_path}")
    print("\n✓  Done!\n")

    return {
        "r2"              : r2,
        "r2_pct"          : r2_pct,
        "cv_mean"         : float(cv_scores.mean()),
        "cv_scores"       : cv_scores.tolist(),
        "outliers"        : outliers,
        "selected_features": selected_features,
        "removed_features": removed_features,
        "heatmap_path"    : heatmap_path,
        "avp_path"        : avp_path,
        "model_path"      : model_path,
    }


# ── Direct execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_and_evaluate()