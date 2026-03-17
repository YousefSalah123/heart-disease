"""
PulseSight — Heart Disease Risk Prediction Dashboard
======================================================================
Production-grade Streamlit application (single-file, multi-section layout).
"""

# ==============================================================================
# STANDARD IMPORTS
# ==============================================================================
import io
import json
import textwrap
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix

# ==============================================================================
# SECTION: config.py — Application constants & filesystem paths
# ==============================================================================

APP_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = APP_DIR.parent

MODEL_PATH: Path = PROJECT_ROOT / "models" / "best_model.pkl"
REPORT_PATH: Path = PROJECT_ROOT / "results" / "best_model_report.json"
DATA_PATH: Path = PROJECT_ROOT / "data" / "selected_features.csv"

APP_TITLE: str = "PulseSight"
APP_SUBTITLE: str = "Heart Disease Risk Assessment Platform"
APP_ICON: str = "🫀"

CHEST_PAIN_OPTIONS: list[str] = [
    "Typical Angina",
    "Atypical Angina",
    "Non-Anginal",
    "Asymptomatic",
]
SLOPE_OPTIONS: list[str] = ["Upsloping", "Flat", "Downsloping"]
THAL_OPTIONS: list[str] = ["Normal", "Fixed Defect", "Reversable Defect"]
REST_ECG_OPTIONS: list[str] = [
    "Normal",
    "ST-T Abnormality",
    "Left Ventricular Hypertrophy",
]

DUMMY_FEATURES: list[str] = [
    "cp_non-anginal",
    "cp_atypical angina",
    "fbs_True",
    "exang_True",
    "slope_flat",
    "slope_upsloping",
    "sex_Male",
    "thal_normal",
    "thal_reversable defect",
]

EXPECTED_NUMERIC_FINAL: list[str] = [
    "chol",
    "chol_per_age",
    "ca",
    "trestbps",
    "heart_rate_reserve",
    "oldpeak",
    "thalch",
    "age",
]

MODEL_COMPARISON_DATA: list[list] = [
    ["baseline",        "DecisionTree",       "default",     0.7772, 0.7905, 0.8137, 0.8019, 0.7727],
    ["tuned",           "DecisionTree",       "grid/random", 0.7826, 0.7500, 0.9118, 0.8230, 0.8510],
    ["baseline",        "LogisticRegression", "default",     0.8370, 0.8333, 0.8824, 0.8571, 0.9081],
    ["tuned",           "LogisticRegression", "grid/random", 0.8424, 0.8614, 0.8529, 0.8571, 0.9108],
    ["baseline",        "RandomForest",       "default",     0.8370, 0.8462, 0.8627, 0.8544, 0.9030],
    ["tuned",           "RandomForest",       "grid/random", 0.8315, 0.8318, 0.8725, 0.8517, 0.9039],
    ["baseline",        "SVC",                "default",     0.8587, 0.8393, 0.9216, 0.8785, 0.9044],
    ["tuned",           "SVC",                "grid/random", 0.8370, 0.8000, 0.9412, 0.8649, 0.9096],
    ["refined",         "dt",                 "smote",       0.8152, 0.7931, 0.9020, 0.8440, 0.8760],
    ["threshold_tuned", "dt",                 "thr_opt",     0.8152, 0.7931, 0.9020, 0.8440, 0.8760],
    ["refined",         "logreg",             "none",        0.8587, 0.8519, 0.9020, 0.8762, 0.9058],
    ["threshold_tuned", "logreg",             "thr_opt",     0.8641, 0.8738, 0.8824, 0.8780, 0.9058],
    ["refined",         "rf",                 "smoteenn",    0.8424, 0.8687, 0.8431, 0.8557, 0.8997],
    ["threshold_tuned", "rf",                 "thr_opt",     0.8587, 0.8958, 0.8431, 0.8687, 0.8997],
    ["ensemble",        "soft_voting",        "mixed",       0.8478, 0.8627, 0.8627, 0.8627, 0.9136],
    ["threshold_tuned", "soft_voting",        "thr_opt",     0.8696, 0.9062, 0.8529, 0.8788, 0.9136],
    ["ensemble",        "stacking",           "mixed",       0.8533, 0.8571, 0.8824, 0.8696, 0.9021],
    ["refined",         "svc",                "none",        0.8152, 0.7931, 0.9020, 0.8440, 0.9052],
    ["threshold_tuned", "svc",                "thr_opt",     0.8804, 0.8846, 0.9020, 0.8932, 0.9052],
    ["refined",         "xgb",                "smoteenn",    0.8533, 0.8713, 0.8627, 0.8670, 0.9000],
    ["threshold_tuned", "xgb",                "thr_opt",     0.8696, 0.9149, 0.8431, 0.8776, 0.9000],
]

# ==============================================================================
# SECTION: models.py — ML compatibility wrappers
# ==============================================================================


class ThresholdSVC(BaseEstimator, ClassifierMixin):
    """SVC wrapper that applies a custom probability threshold at inference time."""

    def __init__(self, threshold: float = 0.5, **svc_params: Any) -> None:
        from sklearn.svm import SVC
        self.threshold = threshold
        self.svc_params = svc_params
        self.model_ = SVC(**svc_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ThresholdSVC":
        self.model_.fit(X, y)
        return self

    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X)[:, 1]
        scores = self.model_.decision_function(X)
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-9:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - mn) / (mx - mn)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._raw_proba(X) >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw_proba(X)
        return np.vstack([1 - raw, raw]).T

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model_, "decision_function"):
            return self.model_.decision_function(X)
        return self.predict_proba(X)[:, 1]

    def get_params(self, deep: bool = True) -> dict:
        return {"threshold": self.threshold, **self.svc_params}

    def set_params(self, **params: Any) -> "ThresholdSVC":
        if "threshold" in params:
            self.threshold = params.pop("threshold")
        self.svc_params.update(params)
        self.model_.set_params(**self.svc_params)
        return self


class ThresholdedEstimator(BaseEstimator, ClassifierMixin):
    """Generic pipeline wrapper with a custom threshold."""

    def __init__(self, pipeline: Any = None, threshold: float = 0.5) -> None:
        self.pipeline = pipeline
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ThresholdedEstimator":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)
        if hasattr(self.pipeline, "decision_function"):
            scores = self.pipeline.decision_function(X)
            probs = expit(scores)
            return np.vstack([1 - probs, probs]).T
        raise AttributeError("Underlying estimator lacks a probability interface.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.pipeline, "decision_function"):
            return self.pipeline.decision_function(X)
        return self.predict_proba(X)[:, 1]


# ==============================================================================
# SECTION: data_loader.py — Cached I/O helpers
# ==============================================================================


@st.cache_resource(show_spinner="Loading ML model...")
def load_pipeline() -> tuple[Any, dict]:
    """Deserialize the best model and its evaluation report from disk."""
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found: `{MODEL_PATH}`. "
            "Please ensure training and export steps have been completed."
        )
        st.stop()

    model_obj = joblib.load(MODEL_PATH)
    report: dict = {}

    if REPORT_PATH.exists():
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as fh:
                report = json.load(fh)
        except Exception as exc:
            st.warning(f"Could not parse model report: {exc}")

    return model_obj, report


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset() -> pd.DataFrame:
    """Load the engineered feature dataset used for visualisation."""
    if DATA_PATH.exists():
        try:
            return pd.read_csv(DATA_PATH)
        except Exception as exc:
            st.warning(f"Failed to load dataset: {exc}")
    return pd.DataFrame()


def ensure_model_loaded() -> None:
    """Load model and dataset into session_state if not already loaded."""
    if not st.session_state.get("model_loaded", False):
        model_obj, report_dict = load_pipeline()
        st.session_state["model"] = model_obj
        st.session_state["report"] = report_dict
        st.session_state["model_loaded"] = True
        raw_df_loaded = load_dataset()
        st.session_state["raw_df"] = raw_df_loaded
        st.session_state["data_loaded"] = not raw_df_loaded.empty


# ==============================================================================
# SECTION: feature_eng.py — Feature construction for inference
# ==============================================================================


def build_feature_frame(form_vals: dict) -> pd.DataFrame:
    """Convert raw form input values into the feature vector expected by the model."""
    age: int = form_vals["age"]
    trestbps: int = form_vals["trestbps"]
    chol: int = form_vals["chol"]
    thalach: int = form_vals["thalach"]
    oldpeak: float = form_vals["oldpeak"]
    ca: int = form_vals["ca"]
    sex: str = form_vals["sex"]
    fbs: str = form_vals["fbs"]
    exang: str = form_vals["exang"]
    slope: str = form_vals["slope"]
    cp: str = form_vals["cp"]
    thal: str = form_vals["thal"]

    chol_per_age: float = chol / (age + 1)
    heart_rate_reserve: float = float(thalach - trestbps)

    data: dict = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalch": thalach,
        "oldpeak": oldpeak,
        "ca": ca,
        "chol_per_age": chol_per_age,
        "heart_rate_reserve": heart_rate_reserve,
    }

    for col in DUMMY_FEATURES:
        data[col] = 0

    if sex == "Male":
        data["sex_Male"] = 1
    if fbs == "Yes":
        data["fbs_True"] = 1
    if exang == "Yes":
        data["exang_True"] = 1

    slope_lower = slope.lower()
    if slope_lower == "flat":
        data["slope_flat"] = 1
    elif slope_lower == "upsloping":
        data["slope_upsloping"] = 1

    cp_lower = cp.lower()
    if cp_lower.startswith("atypical"):
        data["cp_atypical angina"] = 1
    elif cp_lower.startswith("non-anginal"):
        data["cp_non-anginal"] = 1

    thal_lower = thal.lower()
    if thal_lower == "normal":
        data["thal_normal"] = 1
    elif thal_lower == "reversable defect":
        data["thal_reversable defect"] = 1

    df = pd.DataFrame([data])

    model_obj = st.session_state.get("model")
    if model_obj is not None and hasattr(model_obj, "feature_names_in_"):
        for col in model_obj.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[model_obj.feature_names_in_]
    else:
        ordered = [c for c in EXPECTED_NUMERIC_FINAL if c in df.columns]
        rest = [c for c in df.columns if c not in ordered]
        df = df[ordered + rest]

    return df


def run_inference(feat_df: pd.DataFrame) -> tuple[int, float]:
    """Run prediction using the loaded model pipeline."""
    model_obj: Any = st.session_state["model"]
    threshold: float = st.session_state["report"].get("best_threshold", 0.5)

    try:
        prob_arr: np.ndarray = model_obj.predict_proba(feat_df)[:, 1]
    except Exception:
        try:
            scores = model_obj.decision_function(feat_df)
            prob_arr = expit(scores)
        except Exception as exc:
            st.error(f"Model inference failed: {exc}")
            st.stop()

    prob: float = float(prob_arr[0])
    return int(prob >= threshold), prob


# ==============================================================================
# SECTION: ui_components.py — Reusable Streamlit / Plotly widgets
# ==============================================================================


def render_risk_gauge(probability: float, threshold: float) -> None:
    """Render an interactive Plotly gauge chart for the risk probability score."""
    pct = probability * 100
    thr_pct = threshold * 100
    is_high_risk = probability >= threshold
    bar_color = "#E53935" if is_high_risk else "#43A047"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "font": {"size": 32, "color": bar_color}},
            delta={
                "reference": thr_pct,
                "valueformat": ".1f",
                "suffix": "% vs threshold",
                "font": {"size": 12},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#555",
                    "tickfont": {"size": 10},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, thr_pct], "color": "#E8F5E9"},
                    {"range": [thr_pct, 100], "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line": {"color": "#7B1FA2", "width": 4},
                    "thickness": 0.78,
                    "value": thr_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, Inter, sans-serif"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_metric_row(metrics: dict) -> None:
    """Display key model performance metrics as a row of st.metric widgets."""
    cols = st.columns(5)
    fields = [
        ("Accuracy",  "accuracy",  "🎯"),
        ("Precision", "precision", "🔬"),
        ("Recall",    "recall",    "📡"),
        ("F1 Score",  "f1",        "⚖️"),
        ("ROC AUC",   "roc_auc",   "📈"),
    ]
    for col, (label, key, icon) in zip(cols, fields):
        val = metrics.get(key)
        display = f"{val:.3f}" if val is not None else "—"
        col.metric(f"{icon} {label}", display)


def render_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Render an annotated Plotly confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Disease (0)", "Disease (1)"]

    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual"),
        x=labels,
        y=labels,
        color_continuous_scale="Reds",
        text_auto=True,
        title="Confusion Matrix",
        aspect="auto",
    )
    fig.update_coloraxes(showscale=False)
    fig.update_traces(textfont_size=16)
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        title_font_size=14,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_model_comparison_chart(comp_df: pd.DataFrame) -> None:
    """Render a grouped bar chart comparing all model variants across key metrics."""
    long_df = comp_df.melt(
        id_vars=["model", "group", "strategy"],
        value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
        var_name="metric",
        value_name="score",
    )
    long_df["label"] = long_df["model"] + " (" + long_df["group"] + ")"

    fig = px.bar(
        long_df,
        x="label",
        y="score",
        color="metric",
        barmode="group",
        title="All Models — Performance Overview",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        height=480,
    )
    fig.update_layout(
        xaxis_tickangle=-40,
        yaxis=dict(range=[0.6, 1.0], title="Score"),
        legend_title="Metric",
        margin=dict(b=140),
    )
    st.plotly_chart(fig, use_container_width=True)


def example_patient_defaults() -> dict:
    """Return pre-filled example patient values for the prediction form."""
    return dict(
        age=55,
        trestbps=130,
        chol=250,
        thalach=160,
        oldpeak=1.2,
        ca=1,
        sex="Male",
        fbs="Yes",
        exang="Yes",
        slope="Flat",
        cp="Non-Anginal",
        thal="Reversable Defect",
    )


# ==============================================================================
# SECTION: pages/home.py — PulseSight Landing Page
# ==============================================================================


def page_home() -> None:
    """Render the PulseSight landing page."""

    # ── Scoped CSS ─────────────────────────────────────────────────────────────
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

.ps-backdrop {
    min-height: 82vh;
    background: linear-gradient(135deg, #0f0c1a 0%, #1a0e2e 45%, #0e1628 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 3.5rem 1.5rem;
    border-radius: 20px;
    position: relative;
    overflow: hidden;
}
.ps-card {
    position: relative;
    z-index: 10;
    max-width: 720px;
    width: 100%;
    background: rgba(255,255,255,0.038);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 28px;
    padding: 3.5rem 3rem 2.75rem;
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    box-shadow: 0 36px 90px rgba(0,0,0,0.60), 0 0 130px rgba(220,38,38,0.07);
    animation: ps-card-in 0.75s cubic-bezier(0.22,1,0.36,1) forwards;
    opacity: 0;
    transform: translateY(30px);
}
@keyframes ps-card-in { to { opacity:1; transform:translateY(0); } }

.ps-logo-row {
    display: flex; align-items: center; gap: 13px;
    justify-content: center; margin-bottom: 2rem;
}
.ps-logo-icon {
    width: 46px; height: 46px;
    background: linear-gradient(135deg, #dc2626, #9333ea);
    border-radius: 13px;
    display: flex; align-items: center; justify-content: center;
    font-size: 23px;
    box-shadow: 0 4px 22px rgba(220,38,38,0.42);
    animation: ps-logo-pulse 2.8s ease-in-out infinite;
}
@keyframes ps-logo-pulse {
    0%,100% { box-shadow: 0 4px 22px rgba(220,38,38,0.42); }
    50%      { box-shadow: 0 4px 38px rgba(220,38,38,0.75); }
}
.ps-wordmark {
    font-family: 'Syne', sans-serif; font-size:1.55rem; font-weight:800;
    background: linear-gradient(90deg, #f87171, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: 0.4px;
}
.ps-headline {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.9rem, 4.5vw, 2.9rem);
    font-weight: 800; line-height: 1.15;
    text-align: center; margin: 0 0 1.25rem;
    background: linear-gradient(135deg, #ffffff 25%, #f87171 60%, #c084fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.ps-sub {
    text-align: center; font-family: 'DM Sans', sans-serif;
    font-size: 1.04rem; line-height: 1.75;
    color: rgba(255,255,255,0.58);
    margin: 0 auto 2rem; max-width: 530px;
}
.ps-sub strong { color: rgba(255,255,255,0.88); font-weight: 600; }
.ps-tag-row {
    display: flex; gap: 10px; justify-content: center;
    flex-wrap: wrap; margin-bottom: 2.25rem;
}
.ps-tag {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 7px 18px; border-radius: 50px;
    font-family: 'DM Sans', sans-serif; font-size: 0.83rem;
    font-weight: 600; cursor: default; white-space: nowrap;
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}
.ps-tag:hover { transform: translateY(-3px); }
.ps-tag-red   { background:rgba(220,38,38,0.14);  border:1px solid rgba(220,38,38,0.34);  color:#fca5a5; }
.ps-tag-red:hover    { box-shadow: 0 5px 22px rgba(220,38,38,0.32); }
.ps-tag-purple{ background:rgba(147,51,234,0.14); border:1px solid rgba(147,51,234,0.34); color:#d8b4fe; }
.ps-tag-purple:hover { box-shadow: 0 5px 22px rgba(147,51,234,0.30); }
.ps-tag-blue  { background:rgba(59,130,246,0.14); border:1px solid rgba(59,130,246,0.34); color:#93c5fd; }
.ps-tag-blue:hover   { box-shadow: 0 5px 22px rgba(59,130,246,0.26); }
.ps-divider { border:none; border-top:1px solid rgba(255,255,255,0.07); margin:0 0 2rem; }
.ps-stats {
    display: grid; grid-template-columns: repeat(3,1fr);
    gap: 1px; background: rgba(255,255,255,0.07);
    border-radius: 16px; overflow: hidden; margin-bottom: 2.5rem;
}
.ps-stat {
    background: rgba(255,255,255,0.025);
    padding: 1.15rem 0.5rem; text-align: center; transition: background 0.22s;
}
.ps-stat:hover { background: rgba(255,255,255,0.065); }
.ps-stat-num {
    display: block; font-family: 'Syne', sans-serif;
    font-size: 1.55rem; font-weight: 800;
    background: linear-gradient(90deg, #f87171, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 4px;
}
.ps-stat-label {
    font-family: 'DM Sans', sans-serif; font-size: 0.70rem;
    color: rgba(255,255,255,0.38); text-transform: uppercase;
    letter-spacing: 0.9px; font-weight: 500;
}
.ps-credits {
    text-align:center; margin-top:0.9rem;
    font-family:'DM Sans',sans-serif; font-size:0.75rem;
    color:rgba(255,255,255,0.30); letter-spacing:0.3px;
}
.ps-credits span {
    background:linear-gradient(90deg,#f87171,#c084fc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; font-weight:600;
}
.ps-disclaimer {
    text-align:center; margin-top:0.5rem;
    font-family:'DM Sans',sans-serif; font-size:0.70rem;
    color:rgba(255,255,255,0.20); line-height:1.6;
}
.ps-disclaimer strong { color:rgba(255,255,255,0.38); font-weight:600; }
</style>
""", unsafe_allow_html=True)

    # ── Hero HTML ─────────────────────────────────────────────────────────────
    st.markdown("""
<div class="ps-backdrop">
<div class="ps-card">
<div class="ps-logo-row">
<div class="ps-logo-icon">🫀</div>
<span class="ps-wordmark">PulseSight</span>
</div>
<h1 class="ps-headline">See the Risk.<br>Before It Strikes.</h1>
<p class="ps-sub">
Clinical-grade heart disease risk assessment powered by a
<strong>Threshold-Tuned SVC pipeline</strong> —
achieving <strong>87.5% accuracy</strong> and an AUC of
<strong>0.905</strong> on the UCI Heart Disease dataset.
</p>
<div class="ps-tag-row">
<span class="ps-tag ps-tag-red">⚡ 87.5% Accuracy</span>
<span class="ps-tag ps-tag-purple">📊 Threshold-Tuned SVC</span>
<span class="ps-tag ps-tag-blue">🫀 Clinical Decision Support</span>
</div>
<hr class="ps-divider">
<div class="ps-stats">
<div class="ps-stat"><span class="ps-stat-num">0.905</span><span class="ps-stat-label">ROC AUC</span></div>
<div class="ps-stat"><span class="ps-stat-num">17</span><span class="ps-stat-label">Input Features</span></div>
<div class="ps-stat"><span class="ps-stat-num">0.617</span><span class="ps-stat-label">Optimal Threshold</span></div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # ── CTA button ────────────────────────────────────────────────────────────
    st.markdown("<div style='height:1.1rem;'></div>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 1.4, 2])
    with btn_col:
        if st.button(
            "🚀 Start Assessment",
            type="primary",
            use_container_width=True,
            key="cta_home",
        ):
            # Auto-load model so user doesn't need to use the sidebar
            with st.spinner("Loading model…"):
                ensure_model_loaded()
            st.session_state["active_page"] = "Prediction"
            st.rerun()

    # ── Credits + disclaimer ──────────────────────────────────────────────────
    st.markdown("""
<p class="ps-credits">Developed by <span>Yousef Salah</span> &amp; <span>Abdelrahman Mohsen</span></p>
<p class="ps-disclaimer"><strong>Educational &amp; research use only.</strong>
Not a certified medical device. Always consult a qualified healthcare professional.</p>
""", unsafe_allow_html=True)


# ==============================================================================
# SECTION: pages/prediction.py — Prediction form & result dashboard
# ==============================================================================


def page_prediction() -> None:
    """Render the patient input form and display the prediction result."""
    model_obj: Any = st.session_state.get("model")
    report: dict = st.session_state.get("report", {})
    threshold: float = report.get("best_threshold", 0.5)

    if model_obj is None:
        st.warning("Model not loaded. Please click **Load Model** in the sidebar.")
        return

    st.subheader("🔮 Risk Assessment")
    st.caption("Fill in patient parameters below and click **Run Assessment**.")

    use_example: bool = st.toggle("✨ Pre-fill with example patient data", value=False)
    defaults = example_patient_defaults() if use_example else {}

    with st.form("prediction_form", clear_on_submit=False):

        st.markdown("#### 🧬 Demographics & Vitals")
        v1, v2, v3 = st.columns(3)
        age = v1.slider("Age (years)", 20, 100, defaults.get("age", 50))
        sex = v2.radio(
            "Biological Sex",
            ["Male", "Female"],
            index=0 if defaults.get("sex", "Male") == "Male" else 1,
            horizontal=True,
        )
        trestbps = v3.number_input(
            "Resting Blood Pressure (mmHg)", 80, 220, defaults.get("trestbps", 120)
        )

        st.divider()

        st.markdown("#### 🧪 Laboratory Results")
        l1, l2, l3 = st.columns(3)
        chol = l1.number_input(
            "Serum Cholesterol (mg/dl)", 100, 700, defaults.get("chol", 240)
        )
        fbs = l2.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            ["No", "Yes"],
            index=1 if defaults.get("fbs", "No") == "Yes" else 0,
        )
        ca = l3.selectbox(
            "Major Vessels (Fluoroscopy)",
            [0, 1, 2, 3],
            index=defaults.get("ca", 0),
        )

        st.divider()

        st.markdown("#### 🫀 Cardiac Symptoms & Exercise Test")
        s1, s2, s3 = st.columns(3)
        thalach = s1.number_input(
            "Max Heart Rate Achieved (bpm)", 60, 250, defaults.get("thalach", 150)
        )
        exang = s2.radio(
            "Exercise-Induced Angina",
            ["No", "Yes"],
            index=1 if defaults.get("exang", "No") == "Yes" else 0,
            horizontal=True,
        )
        oldpeak = s3.number_input(
            "ST Depression (Oldpeak)",
            0.0, 10.0,
            float(defaults.get("oldpeak", 1.0)),
            step=0.1,
        )

        a1, a2, a3 = st.columns(3)
        cp = a1.selectbox(
            "Chest Pain Type",
            CHEST_PAIN_OPTIONS,
            index=CHEST_PAIN_OPTIONS.index(defaults.get("cp", "Typical Angina")),
        )
        slope = a2.selectbox(
            "ST Slope",
            SLOPE_OPTIONS,
            index=SLOPE_OPTIONS.index(defaults.get("slope", "Upsloping")),
        )
        thal = a3.selectbox(
            "Thalassemia",
            THAL_OPTIONS,
            index=THAL_OPTIONS.index(defaults.get("thal", "Normal")),
        )

        with st.expander("⚙️ Additional Clinical Data (not directly modelled)"):
            st.selectbox("Resting ECG", REST_ECG_OPTIONS, key="restecg_extra")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🧠 Run Assessment", type="primary", use_container_width=True
        )

    if submitted:
        form_vals = dict(
            age=age, trestbps=trestbps, chol=chol, thalach=thalach,
            oldpeak=oldpeak, ca=ca, sex=sex, fbs=fbs, exang=exang,
            slope=slope, cp=cp, thal=thal,
        )
        feat_df = build_feature_frame(form_vals)
        pred_class, prob = run_inference(feat_df)
        st.session_state["last_prediction"] = {
            "class": pred_class,
            "prob": prob,
            "input": form_vals,
            "features": feat_df,
        }

    if st.session_state.get("last_prediction"):
        pred = st.session_state["last_prediction"]
        is_high = pred["class"] == 1
        risk_label = "High Risk" if is_high else "Low Risk"
        risk_color = "#B71C1C" if is_high else "#1B5E20"
        risk_bg = "#FFEBEE" if is_high else "#E8F5E9"
        risk_icon = "⚠️" if is_high else "✅"

        st.markdown("---")
        st.subheader("📋 Assessment Summary")

        r_left, r_right = st.columns([1, 1.4])
        with r_left:
            st.markdown(
                f"""<div style="background:{risk_bg};border:2px solid {risk_color};
border-radius:14px;padding:1.5rem;text-align:center;">
<div style="font-size:2.5rem;">{risk_icon}</div>
<div style="font-size:1.6rem;font-weight:800;color:{risk_color};margin:0.25rem 0;">{risk_label}</div>
<div style="color:#555;font-size:0.9rem;">Predicted outcome based on the provided clinical inputs.</div>
</div>""",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric(
                "Risk Probability",
                f"{pred['prob'] * 100:.1f}%",
                delta=f"{(pred['prob'] - threshold) * 100:+.1f}% vs threshold",
                delta_color="inverse",
            )
            m2.metric("Decision Threshold", f"{threshold:.3f}")

        with r_right:
            render_risk_gauge(pred["prob"], threshold)

        with st.expander("🗂 Submitted Clinical Parameters"):
            echo_df = pd.DataFrame([pred["input"]]).T.rename(columns={0: "Value"})
            st.dataframe(echo_df, use_container_width=True)

        with st.expander("🔧 Encoded Feature Vector (model input)"):
            st.dataframe(pred["features"], use_container_width=True)

        st.markdown("#### ⬇️ Export")
        export_df = pd.DataFrame([{
            **pred["input"],
            "risk_probability": round(pred["prob"], 4),
            "predicted_class": pred["class"],
            "threshold": threshold,
        }])
        csv_buf = io.StringIO()
        export_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download Assessment as CSV",
            data=csv_buf.getvalue(),
            file_name="pulsesight_assessment.csv",
            mime="text/csv",
            key="dl_assessment_csv",
        )

        st.info(
            "⚕️ **Disclaimer:** This tool is for educational and research purposes only "
            "and is **not** a substitute for professional medical advice, diagnosis, "
            "or treatment.",
            icon="ℹ️",
        )


# ==============================================================================
# SECTION: pages/models.py — Model performance & evaluation page
# ==============================================================================


def page_models() -> None:
    """Render the Models & Results page with comparisons, metrics, and evaluation plots."""
    model_obj: Any = st.session_state.get("model")
    report: dict = st.session_state.get("report", {})
    raw_df: pd.DataFrame = st.session_state.get("raw_df", pd.DataFrame())
    threshold: float = report.get("best_threshold", 0.5)

    st.subheader("📈 Models & Results")

    st.markdown("#### 🏅 Best Model — Threshold-Tuned SVC")
    metrics = report.get("metrics", {})

    with st.container(border=True):
        st.markdown(
            "**Threshold-Tuned SVC** — Support Vector Classifier with a calibrated "
            "probability threshold applied post-training via cross-validated optimisation."
        )
        if metrics:
            render_metric_row(metrics)
        else:
            render_metric_row(dict(
                accuracy=0.8804, precision=0.8846,
                recall=0.9020, f1=0.8932, roc_auc=0.9052,
            ))
        st.caption(f"Decision threshold: **{threshold:.4f}**")

    st.markdown("---")

    comp_df = pd.DataFrame(
        MODEL_COMPARISON_DATA,
        columns=["group", "model", "strategy",
                 "accuracy", "precision", "recall", "f1", "roc_auc"],
    )

    tab_compare, tab_eval, tab_raw = st.tabs(
        ["📊 Model Comparison", "🔍 Evaluation Details", "📄 Raw Report"]
    )

    with tab_compare:
        st.markdown("#### All Candidate Models")
        top_acc = comp_df.loc[comp_df["accuracy"].idxmax()]
        top_f1 = comp_df.loc[comp_df["f1"].idxmax()]

        info_l, info_r = st.columns(2)
        info_l.metric(
            "🎯 Best Accuracy",
            f"{top_acc['accuracy']:.4f}",
            f"{top_acc['model']} ({top_acc['group']})",
        )
        info_r.metric(
            "⚖️ Best F1",
            f"{top_f1['f1']:.4f}",
            f"{top_f1['model']} ({top_f1['group']})",
        )

        st.dataframe(
            comp_df.style.highlight_max(
                subset=["accuracy", "precision", "recall", "f1", "roc_auc"],
                color="#C8E6C9",
            ).format(precision=4),
            use_container_width=True,
        )
        render_model_comparison_chart(comp_df)

    # FIX: use if/elif/else instead of return, so tab_raw always renders
    with tab_eval:
        if model_obj is None:
            st.info("Model not loaded — cannot generate live evaluation plots.")
        elif raw_df.empty:
            st.info("Dataset unavailable — cannot generate evaluation plots.")
        else:
            target_col = next(
                (c for c in ["target", "num", "heart_disease"] if c in raw_df.columns),
                raw_df.columns[-1],
            )
            X_all = raw_df.drop(columns=[target_col])
            y_all = raw_df[target_col].values

            try:
                y_prob_all: np.ndarray = model_obj.predict_proba(X_all)[:, 1]
            except Exception:
                try:
                    y_prob_all = expit(model_obj.decision_function(X_all))
                except Exception as exc:
                    st.error(f"Could not obtain model probabilities for evaluation: {exc}")
                    st.stop()

            y_pred_all = (y_prob_all >= threshold).astype(int)

            eval_left, eval_right = st.columns(2)
            with eval_left:
                render_confusion_matrix(y_all, y_pred_all)
            with eval_right:
                st.markdown("#### Classification Report")
                rep_dict = classification_report(
                    y_all, y_pred_all, output_dict=True, zero_division=0
                )
                rep_df = pd.DataFrame(rep_dict).T
                st.dataframe(rep_df.style.format(precision=3), use_container_width=True)

            if metrics:
                metric_plot_df = pd.DataFrame([
                    {"Metric": k.replace("_", " ").title(), "Score": v}
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                ])
                fig_metrics = px.bar(
                    metric_plot_df,
                    x="Metric", y="Score",
                    color="Metric",
                    title="Best Model — Performance Metrics",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    text_auto=".3f",
                )
                fig_metrics.update_layout(
                    yaxis=dict(range=[0, 1]),
                    showlegend=False,
                    height=360,
                )
                st.plotly_chart(fig_metrics, use_container_width=True)

    with tab_raw:
        if report:
            st.json(report)
        else:
            st.info("No report JSON available.")


# ==============================================================================
# SECTION: pages/insights.py — Data Exploration & EDA page
# ==============================================================================


def page_insights() -> None:
    """Render the Data & Insights / EDA page."""
    raw_df: pd.DataFrame = st.session_state.get("raw_df", pd.DataFrame())

    st.subheader("📂 Dataset Insights")

    if raw_df.empty:
        st.warning("Dataset file not found. Visualisations are unavailable.")
        return

    target_col = next(
        (c for c in ["target", "num", "heart_disease"] if c in raw_df.columns),
        raw_df.columns[-1],
    )
    df = raw_df.copy()

    st.markdown("#### 📐 Dataset Overview")
    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Rows", f"{len(df):,}")
    ov2.metric("Features", df.shape[1] - 1)
    miss_pct = (df.isna().sum().sum() / df.size) * 100
    ov3.metric("Missing Values", f"{miss_pct:.1f}%")
    if target_col in df.columns and df[target_col].nunique() <= 5:
        pos_rate = df[target_col].mean() * 100
        ov4.metric("Positive Rate", f"{pos_rate:.1f}%")
    else:
        ov4.metric("Target Classes", df[target_col].nunique())

    st.markdown("---")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    focus_num = [
        c for c in ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
        if c in num_cols
    ]

    tab_dist, tab_cat, tab_corr, tab_eng = st.tabs(
        ["📊 Distributions", "🏷️ Categorical Impact", "🔗 Correlations", "⚙️ Engineered Features"]
    )

    with tab_dist:
        if target_col in df.columns and df[target_col].nunique() <= 10:
            vc = df[target_col].value_counts(dropna=False).reset_index()
            vc.columns = ["class", "count"]
            vc["percent"] = (vc["count"] / vc["count"].sum() * 100).round(1)
            fig_cls = px.bar(
                vc, x="class", y="count", text="percent", color="class",
                title="Target Class Distribution",
                color_discrete_sequence=["#EF9A9A", "#A5D6A7"],
            )
            fig_cls.update_traces(texttemplate="%{text}%")
            st.plotly_chart(fig_cls, use_container_width=True)

        if focus_num:
            st.markdown("#### Key Feature Distributions")
            d1, d2 = st.columns(2)
            for i, col in enumerate(focus_num):
                chart_col = d1 if i % 2 == 0 else d2
                with chart_col:
                    color_arg = target_col if target_col in df.columns else None
                    fig_hist = px.histogram(
                        df, x=col, nbins=28, color=color_arg,
                        marginal="box", opacity=0.82,
                        title=col,
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                    )
                    fig_hist.update_layout(height=320, showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)

    with tab_cat:
        if "sex_Male" in df.columns and target_col in df.columns:
            df["sex_label"] = df["sex_Male"].map({1: "Male", 0: "Female"})
            sex_out = (
                df.groupby("sex_label")[target_col]
                .mean()
                .reset_index()
                .rename(columns={target_col: "disease_rate"})
            )
            fig_sex = px.bar(
                sex_out, x="sex_label", y="disease_rate", color="sex_label",
                title="Disease Rate by Sex",
                color_discrete_sequence=["#42A5F5", "#EF5350"],
            )
            st.plotly_chart(fig_sex, use_container_width=True)

        cp_cols = [c for c in df.columns if c.startswith("cp_")]
        if cp_cols and target_col in df.columns:
            melt = df[cp_cols + [target_col]].copy()
            melt["idx"] = range(len(melt))
            long = melt.melt(id_vars=["idx", target_col], var_name="cp_type", value_name="val")
            long = long[long["val"] == 1]
            cp_rate = long.groupby("cp_type")[target_col].mean().reset_index()
            fig_cp = px.bar(
                cp_rate, x="cp_type", y=target_col,
                title="Disease Rate by Chest Pain Type", color="cp_type",
            )
            st.plotly_chart(fig_cp, use_container_width=True)

        thal_cols = [c for c in df.columns if c.startswith("thal_")]
        if thal_cols and target_col in df.columns:
            melt_t = df[thal_cols + [target_col]].copy()
            melt_t["idx"] = range(len(melt_t))
            long_t = melt_t.melt(
                id_vars=["idx", target_col], var_name="thal_type", value_name="val"
            )
            long_t = long_t[long_t["val"] == 1]
            thal_rate = long_t.groupby("thal_type")[target_col].mean().reset_index()
            fig_thal = px.bar(
                thal_rate, x="thal_type", y=target_col,
                title="Disease Rate by Thalassemia Type", color="thal_type",
            )
            st.plotly_chart(fig_thal, use_container_width=True)

    with tab_corr:
        if len(num_cols) > 2:
            subset = [c for c in num_cols if c != target_col][:12]
            if target_col in num_cols:
                subset.append(target_col)
            corr_matrix = df[subset].corr()
            fig_corr = px.imshow(
                corr_matrix,
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1,
                title="Feature Correlation Matrix",
                aspect="auto",
                text_auto=".2f",
            )
            fig_corr.update_layout(height=520)
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab_eng:
        eng_cols = [c for c in ["chol_per_age", "heart_rate_reserve"] if c in df.columns]
        if not eng_cols:
            st.info(
                "Engineered features (chol_per_age, heart_rate_reserve) "
                "not found in dataset."
            )
        else:
            for eng_col in eng_cols:
                partner = "chol" if eng_col == "chol_per_age" else "thalach"
                if partner in df.columns:
                    fig_sc = px.scatter(
                        df, x=eng_col, y=partner,
                        color=target_col if target_col in df.columns else None,
                        title=f"{partner} vs {eng_col}",
                        opacity=0.65,
                        color_discrete_sequence=px.colors.qualitative.Set1,
                    )
                    st.plotly_chart(fig_sc, use_container_width=True)

        st.caption(
            "chol_per_age = cholesterol / (age + 1)  |  "
            "heart_rate_reserve = max HR - resting BP"
        )


# ==============================================================================
# SECTION: pages/about.py — About page
# ==============================================================================


def page_about() -> None:
    """Render the About / documentation page."""
    st.subheader("ℹ️ About PulseSight")

    with st.container(border=True):
        st.markdown(textwrap.dedent("""
        **Project:** Heart Disease Risk Prediction

        **Model:** Threshold-Tuned Support Vector Classifier (SVC) with an
        integrated preprocessing pipeline including imputation, scaling, and
        one-hot encoding.

        **Performance (approximate):**
        - Accuracy ≈ 87.5–88%
        - F1 Score ≈ 0.893
        - ROC AUC ≈ 0.905

        **Dataset:** UCI Heart Disease repository (Cleveland), with engineered
        features (`chol_per_age`, `heart_rate_reserve`) and a custom decision
        threshold of ≈ 0.617 applied to calibrated probability scores.

        **Team:** Yousef Salah & Abdelrahman Mohsen
        """))

    st.warning(
        "⚕️ **Medical Disclaimer:** This tool is strictly for educational and research "
        "purposes. It is **not** a certified medical device and must **not** be used as "
        "a substitute for professional medical diagnosis, advice, or treatment.",
        icon="⚠️",
    )

    report: dict = st.session_state.get("report", {})
    if report:
        with st.expander("🗂 Full Model Report (JSON)"):
            st.json(report)


# ==============================================================================
# SECTION: app.py — Main entry point, layout, and page router
# ==============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global style overrides ────────────────────────────────────────────────────
st.markdown("""
<style>
header[data-testid="stHeader"] { visibility: hidden; display: none; }
.block-container { padding-top: 5rem; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
div.stButton > button[kind="primary"] {
    background-color: #C62828; border-color: #C62828;
    color: #ffffff; border-radius: 8px; font-weight: 600;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #B71C1C; border-color: #B71C1C;
}
div.stDownloadButton > button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "active_page": "Home",
    "last_prediction": None,
    "model": None,
    "report": {},
    "raw_df": pd.DataFrame(),
    "model_loaded": False,
    "data_loaded": False,
}
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"# {APP_ICON} {APP_TITLE}")
    st.caption(APP_SUBTITLE)
    st.divider()

    st.markdown("### Navigation")
    NAV_PAGES = ["Home", "Prediction", "Models & Results", "Data & Insights", "About"]
    for page_name in NAV_PAGES:
        is_active = st.session_state["active_page"] == page_name
        label = f"**{page_name}**" if is_active else page_name
        if st.button(label, key=f"nav_{page_name}", use_container_width=True):
            st.session_state["active_page"] = page_name
            st.rerun()

    st.divider()

    st.markdown("### ⚙️ Model")
    if not st.session_state["model_loaded"]:
        if st.button("Load Model", type="primary", use_container_width=True):
            ensure_model_loaded()
            st.rerun()
    else:
        threshold_display = st.session_state["report"].get("best_threshold", 0.5)
        st.success("Model ready", icon="✅")
        st.caption(f"Threshold: `{threshold_display:.4f}`")

    if st.session_state.get("last_prediction"):
        st.divider()
        st.markdown("### ⬇️ Export")
        with st.expander("Last Prediction"):
            pred = st.session_state["last_prediction"]
            csv_io = io.StringIO()
            pd.DataFrame(
                [{**pred["input"], "probability": pred["prob"], "class": pred["class"]}]
            ).to_csv(csv_io, index=False)
            st.download_button(
                "Download CSV",
                data=csv_io.getvalue(),
                file_name="pulsesight_last_prediction.csv",
                mime="text/csv",
                key="dl_sidebar_csv",
            )

    st.divider()
    st.caption(
        "For educational purposes only.\nNot a medical device.\n\n"
        "© 2025 Yousef Salah & Abdelrahman Mohsen"
    )

# ── Page router ────────────────────────────────────────────────────────────────
_active = st.session_state["active_page"]

if _active == "Home":
    page_home()

elif _active == "Prediction":
    if not st.session_state["model_loaded"]:
        st.info("👈 Please click **Load Model** in the sidebar before running an assessment.")
    else:
        page_prediction()

elif _active == "Models & Results":
    if not st.session_state["model_loaded"]:
        st.info("👈 Please click **Load Model** in the sidebar to view evaluation results.")
    else:
        page_models()

elif _active == "Data & Insights":
    if not st.session_state["data_loaded"]:
        st.info(
            "👈 Please click **Load Model** in the sidebar to also load the dataset, "
            "then return here."
        )
    else:
        page_insights()

elif _active == "About":
    page_about()