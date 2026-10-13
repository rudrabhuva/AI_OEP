"""
ui.py — Streamlit Dashboard
============================
Interactive web UI for the Student Productivity ANN.
• Loads the saved model (ann_procrastination_model.pkl).
• Lets users enter feature values and get live predictions.
• Displays model plots (correlation heatmap, actual-vs-predicted).
• Shows training metadata and feature importance.

Run:  streamlit run ui.py
      (or launch via:  python main.py)
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ann_procrastination_model.pkl")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Productivity Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark glassmorphism theme ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Global background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 50%, #0f0c29 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 1rem 1.4rem;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.25);
}
div[data-testid="stMetricValue"] { color: #a78bfa !important; font-size: 2rem !important; font-weight: 700 !important; }
div[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.85rem !important; }

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    backdrop-filter: blur(14px);
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.3) 0%, rgba(168,85,247,0.3) 100%);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 24px;
    padding: 2.5rem 3rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 40px rgba(99,102,241,0.2);
}
.hero h1 { color: #fff; font-size: 2.4rem; font-weight: 700; margin: 0 0 0.4rem; }
.hero p  { color: #cbd5e1; font-size: 1.05rem; margin: 0; }

/* ── Section headings ── */
.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 1rem;
    letter-spacing: 0.02em;
    border-left: 4px solid #6366f1;
    padding-left: 0.75rem;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2.5rem;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    cursor: pointer;
    transition: all 0.25s ease;
    width: 100%;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4);
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.6);
}

/* ── Result banner ── */
.result-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.2));
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-box .score { font-size: 3rem; font-weight: 700; color: #34d399; }
.result-box .label { color: #6ee7b7; font-size: 1rem; margin-top: 0.25rem; }

/* ── Warning / error box ── */
.warn-box {
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: #fca5a5;
}

/* ── Input labels ── */
label { color: #cbd5e1 !important; font-size: 0.88rem !important; }

/* ── Tabs ── */
button[data-baseweb="tab"] { color: #94a3b8 !important; font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] { color: #a78bfa !important; border-bottom-color: #a78bfa !important; }

/* ── Images ── */
img { border-radius: 12px; }

/* ── Feature tags ── */
.tag {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 8px;
    padding: 0.2rem 0.6rem;
    font-size: 0.78rem;
    color: #a5b4fc;
    margin: 0.15rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model …")
def load_model():
    """Load the saved sklearn pipeline and feature list from disk."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    data = joblib.load(MODEL_PATH)
    return data["model"], data["features"]


def classify_score(score: float) -> tuple[str, str]:
    """Map a numeric productivity score to a qualitative label + emoji."""
    if score >= 80:
        return "🏆 Excellent", "#34d399"
    elif score >= 65:
        return "✅ Good", "#60a5fa"
    elif score >= 50:
        return "⚡ Average", "#fbbf24"
    elif score >= 35:
        return "⚠️ Below Average", "#f97316"
    else:
        return "🔴 Poor", "#f87171"


def get_feature_dtype(model, features):
    """Return dicts: numeric_features and categorical_features with their known categories."""
    try:
        pre = model.named_steps["preprocessor"]
        num_trans  = dict(pre.transformers_)

        num_feats  = list(num_trans.get("numeric",  type("", (), {"feature_names_in_": []})()).feature_names_in_  if hasattr(num_trans.get("numeric",  None), "feature_names_in_") else [])
        cat_trans  = num_trans.get("categorical", None)
        cat_feats  = list(cat_trans.feature_names_in_) if cat_trans and hasattr(cat_trans, "feature_names_in_") else []
        cat_cats   = cat_trans.categories_ if cat_trans and hasattr(cat_trans, "categories_") else []
    except Exception:
        num_feats, cat_feats, cat_cats = features, [], []

    return num_feats, cat_feats, cat_cats


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 ANN Dashboard")
    st.markdown("---")

    model, features = load_model()

    if model is None:
        st.error("⚠️ Model not found.\nRun `python main.py` first.")
        st.stop()

    st.success("✅ Model loaded")
    st.markdown(f"**Features used:** `{len(features)}`")
    st.markdown("---")

    num_feats, cat_feats, cat_cats = get_feature_dtype(model, features)

    st.markdown("### 📋 Feature List")
    for f in features:
        kind = "🔢" if f in num_feats else "🔤"
        st.markdown(f"{kind} `{f}`")

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>Powered by scikit-learn MLPRegressor<br>"
        "Hidden layers: (64, 32) · Activation: ReLU · Solver: Adam</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main content
# ══════════════════════════════════════════════════════════════════════════════

# Hero banner
st.markdown("""
<div class="hero">
  <h1>🧠 Student Productivity Predictor</h1>
  <p>Artificial Neural Network · scikit-learn MLPRegressor · Interactive Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_predict, tab_plots, tab_info = st.tabs(["🎯 Predict", "📊 Model Plots", "ℹ️ About"])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Predict
# ──────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown('<div class="section-title">Enter Student Feature Values</div>', unsafe_allow_html=True)
    st.markdown("Fill in the values below. Numeric sliders will auto-detect sensible ranges from the training data.")

    # Try to get data range hints from the training pipeline
    @st.cache_data(show_spinner=False)
    def get_data_hints():
        try:
            sys.path.insert(0, BASE_DIR)
            from ann_project import load_and_preprocess
            X_train, _, _, _, selected, _, _ = load_and_preprocess()
            hints = {}
            for col in selected:
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    hints[col] = {
                        "min": float(X_train[col].min()),
                        "max": float(X_train[col].max()),
                        "mean": float(X_train[col].mean()),
                    }
                else:
                    hints[col] = {"categories": sorted(X_train[col].dropna().unique().tolist())}
            return hints
        except Exception:
            return {}

    hints = get_data_hints()

    # Build form
    with st.form("prediction_form"):
        cols = st.columns(2)
        user_values = {}

        for i, feat in enumerate(features):
            col = cols[i % 2]
            h = hints.get(feat, {})

            with col:
                if "categories" in h and h["categories"]:
                    user_values[feat] = st.selectbox(
                        label=feat,
                        options=h["categories"],
                        key=f"inp_{feat}",
                    )
                elif "min" in h:
                    step = 1.0 if (h["max"] - h["min"]) > 10 else 0.1
                    user_values[feat] = st.slider(
                        label=feat,
                        min_value=float(h["min"]),
                        max_value=float(h["max"]),
                        value=float(h["mean"]),
                        step=step,
                        key=f"inp_{feat}",
                    )
                else:
                    user_values[feat] = st.number_input(
                        label=feat,
                        value=0.0,
                        key=f"inp_{feat}",
                    )

        submitted = st.form_submit_button("🚀 Predict Productivity Score")

    if submitted:
        try:
            # Build a correctly-typed row using hints (most reliable source of truth)
            typed_values = {}
            for feat in features:
                h = hints.get(feat, {})
                val = user_values[feat]
                if "categories" in h:
                    # Categorical — keep as plain Python str
                    typed_values[feat] = str(val)
                else:
                    # Numeric — cast to float explicitly
                    typed_values[feat] = float(val)

            student_df = pd.DataFrame([typed_values], columns=features)

            # Belt-and-suspenders: ensure every numeric column is float64
            # so sklearn's isnan check never hits an unsupported type
            for feat in features:
                h = hints.get(feat, {})
                if "categories" not in h:
                    student_df[feat] = student_df[feat].astype("float64")

            score = float(model.predict(student_df)[0])
            label, color = classify_score(score)

            st.markdown(f"""
            <div class="result-box">
                <div class="score" style="color:{color}">{score:.1f} / 100</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge-like progress bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Productivity gauge** — {score:.1f}%")
            st.progress(min(int(score), 100))

        except Exception as exc:
            import traceback
            st.markdown(f'<div class="warn-box">❌ Prediction failed: {exc}</div>', unsafe_allow_html=True)
            with st.expander("🔍 Debug traceback"):
                st.code(traceback.format_exc())


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Plots
# ──────────────────────────────────────────────────────────────────────────────
with tab_plots:
    heatmap_path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    avp_path     = os.path.join(PLOTS_DIR, "actual_vs_predicted.png")

    plots_exist = os.path.exists(heatmap_path) or os.path.exists(avp_path)

    if not plots_exist:
        st.info("📌 Plots are generated when the model is trained. Run `python main.py` to create them.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">📈 Correlation Heatmap</div>', unsafe_allow_html=True)
            st.markdown("Pearson correlation of each numeric feature with the target `productivity_score` (training data only — no leakage).")
            if os.path.exists(heatmap_path):
                st.image(heatmap_path, use_container_width=True)
            else:
                st.warning("Heatmap not found — retrain the model.")

        with col2:
            st.markdown('<div class="section-title">🎯 Actual vs Predicted</div>', unsafe_allow_html=True)
            st.markdown("Scatter plot of true vs. predicted scores on the held-out test set. Points near the red dashed line indicate accurate predictions.")
            if os.path.exists(avp_path):
                st.image(avp_path, use_container_width=True)
            else:
                st.warning("Plot not found — retrain the model.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — About
# ──────────────────────────────────────────────────────────────────────────────
with tab_info:
    st.markdown('<div class="section-title">🏗️ Architecture</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📥 Input Layer")
        st.markdown("One node per selected feature. Numeric features are standardised (z-score); categorical features are one-hot encoded.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Hidden Layers")
        st.markdown("**Layer 1:** 64 neurons — ReLU\n\n**Layer 2:** 32 neurons — ReLU\n\nTraining uses **Adam** optimiser (lr = 0.001, max 300 epochs, early stopping).")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📤 Output Layer")
        st.markdown("Single linear neuron — regression output.\n\nPredicts `productivity_score` (continuous, 0–100).")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📂 Project Structure</div>', unsafe_allow_html=True)
    st.code("""
ai oep/
├── ann_project.py          ← ANN training module (importable functions)
├── main.py                 ← Entry-point: trains model → launches Streamlit
├── ui.py                   ← This Streamlit dashboard
├── student_productivity_distraction_dataset_20000.csv
├── ann_procrastination_model.pkl   ← Saved model (auto-generated)
└── plots/
    ├── correlation_heatmap.png     ← (auto-generated)
    └── actual_vs_predicted.png     ← (auto-generated)
""", language="text")

    st.markdown('<div class="section-title">🔄 Workflow</div>', unsafe_allow_html=True)
    st.markdown("""
1. **`python main.py`** — runs the full pipeline and opens the UI
2. `ann_project.py:train_and_evaluate()` loads the CSV, cleans data, selects features via correlation, trains the ANN, evaluates with 5-fold cross-validation, saves plots and the model.
3. Streamlit (`ui.py`) loads the saved `.pkl` and serves interactive predictions.
""")

    st.markdown('<div class="section-title">📦 Dependencies</div>', unsafe_allow_html=True)
    deps = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "joblib", "streamlit"]
    tag_html = "".join(f'<span class="tag">{d}</span>' for d in deps)
    st.markdown(tag_html, unsafe_allow_html=True)
