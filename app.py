import io
import os
import re
import tempfile
import warnings
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Student Dropout Predictor with SHAP Explainer",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# Session state initialization
# ============================================================
def init_state():
    defaults = {
        "model": None,
        "preprocessor": None,
        "feature_columns": None,
        "training_file_columns_original": None,
        "target_column": None,
        "student_id_column": None,
        "student_name_column": None,
        "train_metrics": None,
        "predict_df": None,
        "prediction_file": None,
        "latest_explanation": None,
        "latest_plot_bytes": None,
        "global_importance_plot_bytes": None,
        "global_summary_plot_bytes": None,
        "shap_explainer": None,
        "is_trained": False,
        "train_success_message": "",
        "prediction_status": "",
        "explain_status": "",
        "selected_model_name": None,
        "model_comparison_df": None,
        "selection_metric": "F1 Score",
        "global_shap_summary_text": "",
        "shap_variance_low": False,
        "shap_variance_warning": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# Styling
# ============================================================
st.markdown(
    """
    <style>
    .shap-plot-frame {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 0.75rem;
        padding: 0.5rem;
        background: white;
    }

    .training-banner {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.85rem 1rem;
        border-radius: 0.75rem;
        background: #e0f2fe;
        color: #075985;
        border: 1px solid #bae6fd;
        font-weight: 600;
        margin-top: 0.6rem;
        margin-bottom: 0.75rem;
    }

    .training-spinner {
        width: 18px;
        height: 18px;
        border: 3px solid #7dd3fc;
        border-top: 3px solid #0284c7;
        border-radius: 50%;
        animation: training-spin 0.9s linear infinite;
        flex-shrink: 0;
    }

    @keyframes training-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    button[kind="secondary"] {
        white-space: nowrap !important;
        width: auto !important;
        padding-left: 12px !important;
        padding-right: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Utility functions
# ============================================================
def reset_prediction_state():
    st.session_state.predict_df = None
    st.session_state.prediction_file = None
    st.session_state.latest_explanation = None
    st.session_state.latest_plot_bytes = None
    st.session_state.prediction_status = ""
    st.session_state.explain_status = ""



def reset_training_state():
    st.session_state.model = None
    st.session_state.preprocessor = None
    st.session_state.feature_columns = None
    st.session_state.training_file_columns_original = None
    st.session_state.target_column = None
    st.session_state.student_id_column = None
    st.session_state.student_name_column = None
    st.session_state.train_metrics = None
    st.session_state.shap_explainer = None
    st.session_state.global_importance_plot_bytes = None
    st.session_state.global_summary_plot_bytes = None
    st.session_state.is_trained = False
    st.session_state.train_success_message = ""
    st.session_state.selected_model_name = None
    st.session_state.model_comparison_df = None
    st.session_state.global_shap_summary_text = ""
    st.session_state.shap_variance_low = False
    st.session_state.shap_variance_warning = ""
    reset_prediction_state()



def on_training_file_change():
    reset_training_state()



def on_prediction_file_change():
    reset_prediction_state()



def get_model_status_text() -> str:
    if st.session_state.is_trained:
        return "✅ Model status: Trained successfully. This Tab is ready for prediction and SHAP explanations."
    return "⚠️ Model status: Not trained yet. Please complete model training in the first Tab before using this Tab"



def format_probability(prob: float) -> str:
    if pd.isna(prob):
        return "N/A"
    return f"{prob * 100:.2f}%"



def render_training_status(placeholder, message: str):
    placeholder.markdown(
        f"""
        <div class="training-banner">
            <div class="training-spinner"></div>
            <div>{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )




def is_low_variance_prediction_array(pred_probs: np.ndarray, threshold: float = 1e-4) -> bool:
    try:
        arr = np.asarray(pred_probs, dtype=float)
        if arr.size == 0:
            return True
        return float(np.nanstd(arr)) < float(threshold)
    except Exception:
        return False


def normalize_binary_target(series: pd.Series) -> pd.Series:
    s = series.copy()

    if pd.api.types.is_numeric_dtype(s):
        unique_vals = sorted(pd.Series(s.dropna().unique()).tolist())
        if set(unique_vals).issubset({0, 1}):
            return s.astype(int)
        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            return s.map(mapping).astype(int)
        raise ValueError("Target column must be binary with exactly two classes.")

    normalized = s.astype(str).str.strip().str.lower()
    positive_map = {"dropout", "dropped", "drop", "yes", "y", "1", "true", "at risk", "atrisk"}
    negative_map = {"no dropout", "no", "n", "0", "false", "continue", "retained", "stayed"}

    unique_vals = sorted(normalized.dropna().unique().tolist())
    if len(unique_vals) != 2:
        raise ValueError(f"Target column must contain exactly 2 classes. Found: {unique_vals}")

    if "dropout" in unique_vals and "no dropout" in unique_vals:
        return normalized.map({"no dropout": 0, "dropout": 1}).astype(int)

    mapped = []
    for val in normalized:
        if val in positive_map:
            mapped.append(1)
        elif val in negative_map:
            mapped.append(0)
        else:
            mapped = None
            break

    if mapped is None:
        ordered = unique_vals
        mapping = {ordered[0]: 0, ordered[1]: 1}
        return normalized.map(mapping).astype(int)

    return pd.Series(mapped, index=s.index, dtype=int)



def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames
            else OneHotEncoder(handle_unknown="ignore", sparse=False),
        ),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])



def get_transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
                ohe = transformer.named_steps["onehot"]
                try:
                    feature_names.extend(ohe.get_feature_names_out(columns).tolist())
                except Exception:
                    feature_names.extend([f"{col}_encoded" for col in columns])
            else:
                feature_names.extend(list(columns))
        return feature_names



def infer_id_and_name_columns(df: pd.DataFrame) -> Tuple[str, str]:
    lowered = {col.lower().strip(): col for col in df.columns}

    candidate_id = None
    for key in ["student id", "student_id", "id", "studentid"]:
        if key in lowered:
            candidate_id = lowered[key]
            break

    candidate_name = None
    for key in ["student name", "student_name", "name", "studentname"]:
        if key in lowered:
            candidate_name = lowered[key]
            break

    return candidate_id, candidate_name



def guess_target_column(columns: List[str]) -> str:
    lowered_map = {c.lower(): c for c in columns}
    for guess in ["target", "dropout", "status", "label"]:
        if guess in lowered_map:
            return lowered_map[guess]
    return columns[0] if columns else None



def validate_prediction_columns(df: pd.DataFrame) -> Tuple[bool, str, List[str], List[str]]:
    if st.session_state.training_file_columns_original is None:
        return False, "The model training column structure is not available. Please retrain the model.", [], []

    uploaded_cols = [col.strip() for col in df.columns.tolist()]
    training_cols = [col.strip() for col in st.session_state.training_file_columns_original]
    target_col = st.session_state.target_column.strip() if st.session_state.target_column else None

    expected_cols = [col for col in training_cols if col != target_col]
    missing_cols = [col for col in expected_cols if col not in uploaded_cols]
    extra_cols = [col for col in uploaded_cols if col not in expected_cols]

    if missing_cols or extra_cols:
        message = (
            "**Upload file incompatibility:**  \n"
            "The uploaded student file does not match the columns expected by the current model. "
            "Some required columns may be missing, or the file may contain different columns from the training data file. "
            "Please upload a student file with the same columns as the training file, or retrain the model using matching training data file.\n\n"
            "**Following are the incompatibilities:**"
        )
        return False, message, missing_cols, extra_cols

    return True, "", [], []



def generate_metrics_table(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": [round(float(v), 4) if pd.notna(v) else np.nan for v in metrics.values()],
    })



def save_prediction_results(df: pd.DataFrame) -> str:
    export_df = df.copy()
    if "Dropout Probability Value" in export_df.columns:
        export_df = export_df.drop(columns=["Dropout Probability Value"])
    path = os.path.join(tempfile.gettempdir(), "student_dropout_predictions.csv")
    export_df.to_csv(path, index=False)
    return path



def clean_feature_names(feature_names: List[str]) -> List[str]:
    cleaned = []

    for name in feature_names:
        name = str(name)

        name = name.replace("num__", "").replace("cat__", "")

        if "_" in name:
            parts = name.split("_")
            if len(parts) >= 2:
                base = parts[0]
                category = " ".join(parts[1:])
                label = f"{base} ({category})"
            else:
                label = name
        else:
            label = name

        label = label.replace("_", " ").strip()
        cleaned.append(label[:70])

    return cleaned

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
    }



def get_model_by_name(model_name: str):
    if model_name == "XGBoost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed. Please add xgboost to requirements.txt.")
        return XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )

    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42)

    if model_name == "Neural Network":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
        )

    raise ValueError(f"Unsupported model name: {model_name}")



def train_single_model(model_name, X_train_t, X_test_t, y_train, y_test):
    model = get_model_by_name(model_name)
    model.fit(X_train_t, y_train)

    y_prob = model.predict_proba(X_test_t)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    return model, metrics



def choose_best_model(results_df: pd.DataFrame, metric_name: str) -> str:
    best_idx = results_df[metric_name].astype(float).idxmax()
    return results_df.loc[best_idx, "Model"]


# -----------------------------
# Split SHAP approach
# -----------------------------
def get_local_probability_explainer(model, X_background):
    background = X_background[: min(200, len(X_background))]
    return shap.Explainer(model.predict_proba, background)



def get_fast_global_explainer(model, model_name, X_background):
    background = X_background[: min(200, len(X_background))]

    if model_name in ["XGBoost", "Random Forest"]:
        return shap.TreeExplainer(model)

    if model_name == "Logistic Regression":
        try:
            return shap.LinearExplainer(model, background)
        except Exception:
            return shap.Explainer(model, background)

    if model_name == "Neural Network":
        return shap.Explainer(model.predict_proba, background[: min(100, len(background))])

    return shap.Explainer(model, background)



def create_shap_explanation_probability(explainer, X_row_transformed: np.ndarray, feature_names: List[str]):
    shap_values = explainer(X_row_transformed)
    values = shap_values.values
    base_values = shap_values.base_values

    if np.ndim(values) == 3:
        values_1d = values[0, :, 1]
    elif np.ndim(values) == 2:
        values_1d = values[0]
    else:
        values_1d = values

    if np.ndim(base_values) == 2:
        base_value = base_values[0, 1]
    elif np.ndim(base_values) == 1:
        base_value = base_values[1] if len(base_values) > 1 else base_values[0]
    else:
        base_value = base_values

    row_data = X_row_transformed[0] if np.ndim(X_row_transformed) > 1 else X_row_transformed

    return shap.Explanation(
        values=values_1d,
        base_values=base_value,
        data=row_data,
        feature_names=clean_feature_names(feature_names),
    )



def extract_positive_class_shap_values(values_obj) -> np.ndarray:
    values = values_obj.values if hasattr(values_obj, "values") else values_obj

    if isinstance(values, list):
        if len(values) >= 2:
            return np.array(values[1])
        return np.array(values[0])

    values = np.array(values)
    if values.ndim == 3:
        if values.shape[-1] >= 2:
            return values[:, :, 1]
        return values[:, :, 0]
    if values.ndim == 2:
        return values
    raise ValueError("Unexpected SHAP values shape for global plots.")



def build_global_shap_plots_fast(explainer, model_name: str, X_sample: np.ndarray, feature_names: List[str]):
    cleaned_feature_names = clean_feature_names(feature_names)
    shap_values_obj = explainer(X_sample)

    if model_name in ["XGBoost", "Random Forest"]:
        shap_values = extract_positive_class_shap_values(shap_values_obj)
    elif hasattr(shap_values_obj, "values"):
        shap_values = extract_positive_class_shap_values(shap_values_obj)
    else:
        shap_values = np.array(shap_values_obj)

    plt.close("all")
    plt.figure(figsize=(11, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=cleaned_feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    fig_bar = plt.gcf()
    importance_bytes = figure_to_png_bytes(fig_bar)
    plt.close(fig_bar)

    plt.close("all")
    plt.figure(figsize=(11, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=cleaned_feature_names,
        max_display=15,
        show=False,
    )
    fig_summary = plt.gcf()
    summary_bytes = figure_to_png_bytes(fig_summary)
    plt.close(fig_summary)

    return importance_bytes, summary_bytes



def build_shap_figure(explanation, max_display: int = 10):
    plt.close("all")
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    ax = plt.gca()

    labels = [str(t.get_text()) for t in ax.get_yticklabels() if t.get_text()]
    max_len = max((len(x) for x in labels), default=25)

    fig_width = min(max(13, 9 + max_len * 0.08), 18)
    fig_height = min(max(6.5, 0.55 * max_display + 2.2), 10)
    fig.set_size_inches(fig_width, fig_height)

    left_margin = min(max(0.30, max_len * 0.008), 0.50)
    fig.subplots_adjust(left=left_margin, right=0.98, top=0.92, bottom=0.14)

    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    return fig



def figure_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", pad_inches=0.35)
    except Exception:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=160)
    buffer.seek(0)
    return buffer.getvalue()



def get_student_id_choices_from_predictions() -> List[str]:
    student_id_col = st.session_state.student_id_column
    if (
        st.session_state.predict_df is not None
        and student_id_col
        and student_id_col in st.session_state.predict_df.columns
    ):
        return sorted(st.session_state.predict_df[student_id_col].astype(str).dropna().unique().tolist())
    return []



def render_centered_chart_help(title: str, help_text: str, heading_level: int = 0):
    left_col, center_col, right_col = st.columns([3, 2, 1])
    with left_col:
        if heading_level == 3:
            st.markdown(f"### {title}")
        else:
            st.markdown(f"**{title}**")
    with center_col:
        st.write("")
        st.write("")
        with st.popover("ℹ️ How to read this chart"):
            st.markdown(help_text)
    with right_col:
        st.write("")


def generate_plain_language_shap_summary(explanation, prediction_label, prediction_prob, top_n=3):
    feature_names = explanation.feature_names
    shap_values = explanation.values

    items = list(zip(feature_names, shap_values))
    items_sorted = sorted(items, key=lambda x: abs(x[1]), reverse=True)

    risk_increasing = [(name, val) for name, val in items_sorted if val > 0][:top_n]
    risk_reducing = [(name, val) for name, val in items_sorted if val < 0][:top_n]

    def clean_name(name):
        name = str(name)

        if "_" in name:
            parts = name.split("_")
            if len(parts) >= 2:
                base = parts[0]
                category = " ".join(parts[1:])
                return f"{base} ({category})"

        return name.replace("_", " ").strip()

    increasing_text = ", ".join(clean_name(name) for name, _ in risk_increasing) if risk_increasing else "no major factors"
    reducing_text = ", ".join(clean_name(name) for name, _ in risk_reducing) if risk_reducing else "no major factors"

    summary_html = f"""
    This student was predicted as <b>{prediction_label}</b> with a dropout probability of <b>{prediction_prob}</b>.<br><br>
    <b>Factors increasing dropout risk:</b> {increasing_text}<br><br>
    <b>Factors reducing dropout risk:</b> {reducing_text}<br><br>
    """

    if str(prediction_label).lower() == "dropout":
        summary_html += "Overall, the factors increasing dropout risk were stronger than the factors reducing risk which resulted in a <b>Dropout</b> prediction."
    else:
        summary_html += "Overall, the factors reducing dropout risk were stronger than the factors increasing risk which resulted in a <b>No Dropout</b> prediction."

    return summary_html


def generate_shap_recommendations(explanation, prediction_label, prediction_prob_value, raw_row, top_n=3):
    feature_names = explanation.feature_names
    shap_values = explanation.values

    items = list(zip(feature_names, shap_values))
    items_sorted = sorted(items, key=lambda x: abs(x[1]), reverse=True)

    risk_increasing = [(name, val) for name, val in items_sorted if val > 0][:top_n]
    risk_reducing = [(name, val) for name, val in items_sorted if val < 0][:top_n]

    def clean_name(name):
        name = str(name).replace("num__", "").replace("cat__", "").strip()
        if "_" in name:
            parts = name.split("_")
            if len(parts) >= 2:
                base = parts[0].replace("_", " ").strip()
                category = " ".join(parts[1:]).replace("_", " ").strip()
                return f"{base} ({category})"
        return name.replace("_", " ").strip()

    def format_raw_value(value):
        if pd.isna(value):
            return "Missing"
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            value = float(value)
            if value.is_integer():
                return str(int(value))
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return str(value).strip()

    def get_display_label(name):
        cleaned = clean_name(name)

        # First try an exact raw-column match, even if the column name contains parentheses.
        if hasattr(raw_row, "index"):
            lookup = {str(col).strip().lower(): col for col in raw_row.index}
            key = cleaned.strip().lower()
            if key in lookup:
                raw_value = raw_row[lookup[key]]
                return f"<b>{cleaned} - {format_raw_value(raw_value)}</b>"

        # Only if no exact raw-column match exists, treat it like a one-hot style feature.
        if "(" in cleaned and cleaned.endswith(")"):
            base = cleaned[:cleaned.rfind("(")].strip()
            category = cleaned[cleaned.rfind("(") + 1:-1].strip()
            return f"<b>{base} - {category}</b>"

        return f"<b>{cleaned}</b>"

    def get_plain_display_label(name):
        label_html = get_display_label(name)
        return re.sub(r"</?b>", "", label_html)

    def format_feature_list(names):
        cleaned = [clean_name(name) for name in names if str(name).strip()]
        if not cleaned:
            return "no major factors"
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}" 

    recommendations = []

    try:
        prob = float(prediction_prob_value)
    except Exception:
        prob = np.nan

    if pd.notna(prob) and prob >= 0.50:
        top_feature_names = [name for name, _ in risk_increasing]
        top_features_text = format_feature_list(top_feature_names)

        recommendations.append(
            f"The main factors contributing to this student’s dropout risk are {top_features_text}."
        )

        for name, _ in risk_increasing:
            feature_label = get_display_label(name)
            recommendations.append(
                f"{feature_label} is one of the factors contributing to this student’s dropout risk."
            )

    elif pd.notna(prob) and 0.25 <= prob < 0.50:
        top_feature_names = [name for name, _ in risk_increasing]
        top_features_text = format_feature_list(top_feature_names)

        recommendations.append(
            f"The main factors influencing this student’s prediction are {top_features_text}."
        )

        for name, _ in risk_increasing:
            feature_label = get_display_label(name)
            recommendations.append(
                f"{feature_label} is one of the factors influencing this student’s predicted outcome."
            )

    else:
        top_feature_names = [name for name, _ in risk_reducing]
        top_features_text = format_feature_list(top_feature_names)

        recommendations.append(
            f"The main factors supporting this student’s predicted outcome are {top_features_text}."
        )

        for name, _ in risk_reducing:
            feature_label = get_display_label(name)
            recommendations.append(
                f"{feature_label} is one of the factors contributing positively to this student’s predicted outcome."
            )

    if not recommendations:
        recommendations.append(
            "No strong contributing factors were identified among the top SHAP features for this prediction."
        )

    unique_recommendations = []
    seen = set()
    for rec in recommendations:
        if rec not in seen:
            unique_recommendations.append(rec)
            seen.add(rec)

    return unique_recommendations


def render_summary_box(student_id: str, summary_html: str):
    st.markdown(
        f"""
        <div style="
            padding: 0.9rem 1rem;
            border-radius: 0.6rem;
            background: #eff6ff;
            color: #1e3a8a;
            border: 1px solid #bfdbfe;
            margin-top: 0.75rem;
            margin-bottom: 0.75rem;
        ">
            <div style="font-weight: 700; margin-bottom: 0.45rem;">
                Summary explanation for Student ID {student_id}
            </div>
            <div style="line-height: 1.6;">
                {summary_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_box(student_id: str, recommendations):
    rec_html = "".join([f"<li>{rec}</li>" for rec in recommendations])

    st.markdown(
        f"""
        <div style="
            padding: 0.9rem 1rem;
            border-radius: 0.6rem;
            background: #fffbeb;
            color: #92400e;
            border: 1px solid #fde68a;
            margin-top: 0.75rem;
            margin-bottom: 0.75rem;
        ">
            <div style="font-weight: 700; margin-bottom: 0.45rem;">
                Recommendations for Student ID {student_id}
            </div>
            <ul style="margin-top: 0.35rem; padding-left: 1.2rem; line-height: 1.6;">
                {rec_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_global_shap_summary(feature_names: List[str], shap_values: np.ndarray, top_n: int = 5):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_signed = np.mean(shap_values, axis=0)

    feature_importance = list(zip(feature_names, mean_abs, mean_signed))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    top_features = feature_importance[:top_n]
    increasing = [name for name, _, signed in feature_importance if signed > 0][:top_n]
    reducing = [name for name, _, signed in feature_importance if signed < 0][:top_n]

    def clean_name(name):
        name = str(name).replace("num__", "").replace("cat__", "")

        if "_" in name:
            parts = name.split("_")
            if len(parts) >= 2:
                base = parts[0]
                category = " ".join(parts[1:])
                return f"{base} ({category})"

        return name.replace("_", " ").strip()

    def format_feature_list(names):
        cleaned = [clean_name(name) for name in names if str(name).strip()]
        if not cleaned:
            return "no major factors"
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}" 

    top_feature_names = [name for name, _, _ in top_features]
    top_features_text = format_feature_list(top_feature_names)
    increasing_text = format_feature_list(increasing) if increasing else "no clear overall risk-increasing factors"
    reducing_text = format_feature_list(reducing) if reducing else "no clear overall risk-reducing factors"

    if top_feature_names:
        overall_focus_text = format_feature_list(top_feature_names[:4])
        final_sentence = (
            f"Overall, these patterns suggest the institution should pay close attention to "
            f"<b>{overall_focus_text}</b> when identifying students who may need support."
        )
    else:
        final_sentence = (
            "Overall, these patterns suggest the institution should continue monitoring the main drivers of dropout risk when identifying students who may need support."
        )

    summary_html = f"""
The model found that the strongest overall factors related to dropout risk were <b>{top_features_text}</b>.<br><br>
Factors that tended to increase dropout risk overall included <b>{increasing_text}</b>.<br><br>
Factors that tended to reduce dropout risk overall included <b>{reducing_text}</b>.<br><br>
{final_sentence}
"""

    return summary_html

def render_global_summary_box(summary_html: str):
    summary_html = re.sub(r"</?div[^>]*>", "", str(summary_html)).strip()

    st.markdown(
        f"""
        <div style="
            padding: 0.9rem 1rem;
            border-radius: 0.6rem;
            background: #eff6ff;
            color: #1e3a8a;
            border: 1px solid #bfdbfe;
            margin-top: 0.75rem;
            margin-bottom: 0.75rem;
        ">
            <div style="font-weight: 700; margin-bottom: 0.45rem;">
                SHAP Explanation - Overall Summary
            </div>
            <div style="line-height: 1.6;">
                {summary_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def train_institution_model(
    df: pd.DataFrame,
    target_column: str,
    student_id_column: str,
    student_name_column: str,
    test_size: float,
    model_choice: str,
    selection_metric: str,
    status_placeholder,
):
    if XGBClassifier is None and model_choice in ["XGBoost", "Run all 4 and choose the best"]:
        raise RuntimeError("xgboost is not installed. Please add xgboost to requirements.txt.")

    reset_training_state()
    df = df.copy()
    df.columns = df.columns.str.strip()
    original_training_columns = df.columns.tolist()

    if df.empty:
        raise ValueError("The uploaded training file is empty.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the uploaded file.")

    y = normalize_binary_target(df[target_column])

    drop_cols = [target_column]
    if student_id_column and student_id_column in df.columns:
        drop_cols.append(student_id_column)
    if student_name_column and student_name_column in df.columns and student_name_column not in drop_cols:
        drop_cols.append(student_name_column)

    X = df.drop(columns=drop_cols, errors="ignore").copy()
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns found after removing target / ID / name columns.")

    render_training_status(status_placeholder, "Preparing training data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=42,
        stratify=y if y.nunique() == 2 else None,
    )

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    candidate_models = (
        ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"]
        if model_choice == "Run all 4 and choose the best"
        else [model_choice]
    )

    render_training_status(status_placeholder, "Training models...")
    comparison_rows = []
    trained_models = {}

    for idx, name in enumerate(candidate_models, start=1):
        render_training_status(status_placeholder, f"Training models... ({idx}/{len(candidate_models)}) {name}")
        model, metrics = train_single_model(name, X_train_t, X_test_t, y_train, y_test)
        trained_models[name] = model
        comparison_rows.append({
            "Model": name,
            "Accuracy": round(metrics["Accuracy"], 4),
            "Precision": round(metrics["Precision"], 4),
            "Recall": round(metrics["Recall"], 4),
            "F1 Score": round(metrics["F1 Score"], 4),
            "ROC AUC": round(metrics["ROC AUC"], 4) if pd.notna(metrics["ROC AUC"]) else np.nan,
        })

    comparison_df = pd.DataFrame(comparison_rows)

    render_training_status(status_placeholder, "Selecting best model...")
    best_model_name = choose_best_model(comparison_df, selection_metric) if model_choice == "Run all 4 and choose the best" else model_choice
    best_model = trained_models[best_model_name]

    best_metrics_row = comparison_df[comparison_df["Model"] == best_model_name].iloc[0].to_dict()
    best_metrics = {
        "Accuracy": best_metrics_row["Accuracy"],
        "Precision": best_metrics_row["Precision"],
        "Recall": best_metrics_row["Recall"],
        "F1 Score": best_metrics_row["F1 Score"],
        "ROC AUC": best_metrics_row["ROC AUC"],
    }

    feature_names = get_transformed_feature_names(preprocessor)

    global_sample_size = min(300, X_train_t.shape[0])
    sample_idx = np.random.RandomState(42).choice(X_train_t.shape[0], size=global_sample_size, replace=False)
    X_shap_sample = X_train_t[sample_idx]
    shap_sample_pred_probs = best_model.predict_proba(X_shap_sample)[:, 1]

    shap_variance_low = is_low_variance_prediction_array(shap_sample_pred_probs)
    shap_variance_warning = ""
    local_explainer = None
    importance_bytes = None
    summary_bytes = None
    global_shap_summary_text = ""

    if shap_variance_low:
        shap_variance_warning = (
            "⚠️ The model predictions show very low variation for this dataset. "
            "This means the model is producing nearly the same prediction for most students, "
            "so SHAP explanations are not meaningful. Global and individual SHAP plots have been skipped."
        )
    else:
        render_training_status(status_placeholder, "Building local SHAP explainer...")
        X_local_background = X_train_t[: min(200, len(X_train_t))]
        local_explainer = get_local_probability_explainer(best_model, X_local_background)

        render_training_status(status_placeholder, "Generating global SHAP plots...")
        X_global_background = X_train_t[: min(200, len(X_train_t))]
        global_explainer = get_fast_global_explainer(best_model, best_model_name, X_global_background)

        importance_bytes, summary_bytes = build_global_shap_plots_fast(
            global_explainer,
            best_model_name,
            X_shap_sample,
            feature_names,
        )

        global_shap_values_obj = global_explainer(X_shap_sample)
        if hasattr(global_shap_values_obj, "values"):
            global_shap_values = extract_positive_class_shap_values(global_shap_values_obj)
        else:
            global_shap_values = np.array(global_shap_values_obj)

        global_shap_summary_text = generate_global_shap_summary(
            feature_names,
            global_shap_values,
            top_n=5,
        )

    render_training_status(status_placeholder, "Finalizing trained model...")
    st.session_state.model = best_model
    st.session_state.preprocessor = preprocessor
    st.session_state.feature_columns = X.columns.tolist()
    st.session_state.training_file_columns_original = original_training_columns
    st.session_state.target_column = target_column
    st.session_state.student_id_column = student_id_column if student_id_column in df.columns else None
    st.session_state.student_name_column = student_name_column if student_name_column in df.columns else None
    st.session_state.train_metrics = best_metrics
    st.session_state.shap_explainer = local_explainer
    st.session_state.global_importance_plot_bytes = importance_bytes
    st.session_state.global_summary_plot_bytes = summary_bytes
    st.session_state.is_trained = True
    st.session_state.selected_model_name = best_model_name
    st.session_state.model_comparison_df = comparison_df
    st.session_state.selection_metric = selection_metric
    st.session_state.global_shap_summary_text = global_shap_summary_text
    st.session_state.shap_variance_low = shap_variance_low
    st.session_state.shap_variance_warning = shap_variance_warning

    if model_choice == "Run all 4 and choose the best":
        st.session_state.train_success_message = (
            f"✅ Best model selected: {best_model_name} based on {selection_metric}. "
            f"Model trained successfully on {len(df)} records with {X.shape[1]} feature columns. "
            "You can now use the next Tab 'Predict + Explain' for predictions and SHAP explanations."
        )
    else:
        st.session_state.train_success_message = (
            f"✅ {best_model_name} trained successfully on {len(df)} records with {X.shape[1]} feature columns. "
            "You can now use the next Tab 'Predict + Explain' for predictions and SHAP explanations."
        )



def generate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if not st.session_state.is_trained or st.session_state.model is None or st.session_state.preprocessor is None:
        raise RuntimeError("Please train the institution model first in Tab 1.")

    df = df.copy()
    df.columns = df.columns.str.strip()

    is_valid, validation_message, _, _ = validate_prediction_columns(df)
    if not is_valid:
        raise ValueError(validation_message)

    X_new = df[st.session_state.feature_columns].copy()
    X_new_transformed = st.session_state.preprocessor.transform(X_new)
    probs = st.session_state.model.predict_proba(X_new_transformed)[:, 1]
    preds = (probs >= 0.5).astype(int)

    result_df = df.copy()
    result_df["Dropout Probability"] = [format_probability(p) for p in probs]
    result_df["Dropout Probability Value"] = probs
    result_df["Prediction"] = np.where(preds == 1, "Dropout", "No Dropout")

    st.session_state.predict_df = result_df
    st.session_state.prediction_file = save_prediction_results(result_df)
    st.session_state.prediction_status = f"✅ Predictions generated successfully for {len(result_df)} records."
    return result_df



def format_explain_status(student_id: str, pred_label: str, pred_prob: float) -> str:
    return (
        f"✅ SHAP explanation generated for Student ID {student_id}\n"
        f"<b>Prediction:</b> {pred_label}\n"
        f"<b>Dropout Probability:</b> {format_probability(pred_prob)}"
    )



def explain_student(chosen_id: str):
    if not st.session_state.is_trained:
        raise RuntimeError("Please train the institution model first in Tab 1.")
    if st.session_state.shap_variance_low or st.session_state.shap_explainer is None:
        raise RuntimeError(
            "SHAP explanations are not available because the model predictions show very low variation for this dataset."
        )
    if st.session_state.predict_df is None:
        raise RuntimeError("Please upload a prediction file and generate predictions first.")

    student_id_col = st.session_state.student_id_column
    if not student_id_col or student_id_col not in st.session_state.predict_df.columns:
        raise RuntimeError("No student ID column was configured during training or found in prediction results.")

    matched = st.session_state.predict_df[
        st.session_state.predict_df[student_id_col].astype(str) == str(chosen_id)
    ]
    if matched.empty:
        raise ValueError(f"Student ID '{chosen_id}' was not found in the uploaded prediction results.")

    row = matched.iloc[[0]].copy()
    X_row = row[st.session_state.feature_columns]
    X_row_transformed = st.session_state.preprocessor.transform(X_row)

    explanation = create_shap_explanation_probability(
        st.session_state.shap_explainer,
        X_row_transformed,
        get_transformed_feature_names(st.session_state.preprocessor),
    )

    fig = build_shap_figure(explanation)
    plot_bytes = figure_to_png_bytes(fig)
    plt.close(fig)

    st.session_state.latest_explanation = explanation
    st.session_state.latest_plot_bytes = plot_bytes

    pred_label = row["Prediction"].iloc[0] if "Prediction" in row.columns else "Unknown"
    pred_prob = row["Dropout Probability Value"].iloc[0] if "Dropout Probability Value" in row.columns else np.nan
    st.session_state.explain_status = format_explain_status(chosen_id, pred_label, pred_prob)


# ============================================================
# UI
# ============================================================
st.title("🎓 Student Dropout Predictor with SHAP Explainer")
st.write(
    "Upload a CSV file containing student records to train an institution-specific model, "
    "then generate dropout predictions and SHAP-based explanations."
)

train_tab, predict_tab = st.tabs(["🏫 Train Institution Model", "📊 Predict + Explain"])

with train_tab:
    st.subheader("Train Institution Model")
    col1, col2 = st.columns([1, 3])

    with col1:
        training_file = st.file_uploader(
            "📄 Upload Labeled Training CSV",
            type=["csv"],
            key="training_file_uploader",
            on_change=on_training_file_change,
        )

        target_column = None
        student_id_column = None
        student_name_column = None
        training_df_preview = None

        if training_file is not None:
            try:
                training_df_preview = pd.read_csv(training_file)
                training_df_preview.columns = training_df_preview.columns.str.strip()
                columns = training_df_preview.columns.tolist()
                guessed_id, guessed_name = infer_id_and_name_columns(training_df_preview)
                guessed_target = guess_target_column(columns)

                target_column = st.selectbox(
                    "Target Column",
                    options=columns,
                    index=columns.index(guessed_target) if guessed_target in columns else 0,
                )

                id_options = ["None"] + columns
                name_options = ["None"] + columns

                student_id_column = st.selectbox(
                    "Student ID Column",
                    options=id_options,
                    index=id_options.index(guessed_id) if guessed_id in id_options else 0,
                )
                student_name_column = st.selectbox(
                    "Student Name Column",
                    options=name_options,
                    index=name_options.index(guessed_name) if guessed_name in name_options else 0,
                )

                test_size = st.slider(
                    "Test Split Proportion",
                    min_value=0.10,
                    max_value=0.40,
                    value=0.20,
                    step=0.05,
                )

                model_choice = st.selectbox(
                    "Model Selection",
                    [
                        "XGBoost",
                        "Random Forest",
                        "Logistic Regression",
                        "Neural Network",
                        "Run all 4 and choose the best",
                    ],
                )

                selection_metric = st.selectbox(
                    "Best Model Selection Metric",
                    ["F1 Score", "ROC AUC", "Accuracy", "Recall", "Precision"],
                    index=0,
                )

                train_button = st.button("🚀 Train Model", use_container_width=True)
                training_status_placeholder = st.empty()

                if train_button:
                    st.session_state.train_metrics = None
                    st.session_state.global_importance_plot_bytes = None
                    st.session_state.global_summary_plot_bytes = None
                    st.session_state.train_success_message = ""
                    st.session_state.selected_model_name = None
                    st.session_state.model_comparison_df = None

                    try:
                        train_institution_model(
                            training_df_preview,
                            target_column,
                            None if student_id_column == "None" else student_id_column,
                            None if student_name_column == "None" else student_name_column,
                            test_size,
                            model_choice,
                            selection_metric,
                            training_status_placeholder,
                        )
                    except Exception as e:
                        st.error(f"Model training failed: {e}")
                    finally:
                        training_status_placeholder.empty()

            except Exception as e:
                st.error(f"Training file could not be read: {e}")

        if st.session_state.train_success_message:
            st.success(st.session_state.train_success_message)
            if st.session_state.global_shap_summary_text:
                render_global_summary_box(st.session_state.global_shap_summary_text)

    with col2:
        st.markdown("### Model Performance Metrics")
        if st.session_state.train_metrics is not None:
            metrics_df = generate_metrics_table(st.session_state.train_metrics)
            display_df = metrics_df, width=380, hide_index=True.reset_index(drop=True).copy()
            display_df.index = display_df.index + 1
            display_df.index.name = "No."
            st.dataframe(display_df)

            if st.session_state.selected_model_name is not None:
                st.markdown(f"**Selected Model:** {st.session_state.selected_model_name}")

            if st.session_state.model_comparison_df is not None and len(st.session_state.model_comparison_df) > 1:
                st.markdown("### Model Comparison")
                st.dataframe(st.session_state.model_comparison_df, width="stretch", hide_index=True)

            st.markdown("### Global SHAP Summary")
            if st.session_state.shap_variance_warning:
                st.warning(st.session_state.shap_variance_warning)
            plot_col1, plot_col2 = st.columns(2)

            with plot_col1:
                render_centered_chart_help(
                    "Feature Importance Plot",
                    """
This chart shows the **most important factors** affecting dropout risk overall.

- Longer bars mean a factor has a **stronger influence**
- Features at the top are the **most important**

This chart shows **importance only**, not direction.
""",
                )
                if st.session_state.global_importance_plot_bytes is not None:
                    st.image(st.session_state.global_importance_plot_bytes, width="stretch")
                else:
                    st.info("Feature importance plot will appear here after model training.")

            with plot_col2:
                render_centered_chart_help(
                    "Summary Plot",
                    """
This chart shows how different factors influence dropout risk **across all students**.

- Each dot represents **one student**
- Features are ordered by **importance (top = most important)**

🔴 **Red dots** → higher feature values  
🔵 **Blue dots** → lower feature values  

➡️ Dots to the **right** increase dropout risk  
⬅️ Dots to the **left** reduce dropout risk  

The wider the spread, the stronger the feature's overall impact.
""",
                )
                if st.session_state.global_summary_plot_bytes is not None:
                    st.image(st.session_state.global_summary_plot_bytes, width="stretch")
                else:
                    st.info("Summary plot will appear here after model training.")
        else:
            st.info("Model performance metrics and global SHAP plots will appear here after model training.")

with predict_tab:
    st.info(get_model_status_text())

    if not st.session_state.is_trained:
        st.warning("You need to complete model training in the first Tab 'Train Institution Model' before using this section")
    else:
        if st.session_state.selected_model_name:
            st.caption(f"Active model: {st.session_state.selected_model_name}")

        st.subheader("Upload & Predict")
        pred_col1, pred_col2 = st.columns([1, 3])

        with pred_col1:
            prediction_file = st.file_uploader(
                "📄 Upload new Student CSV File to get predictions",
                type=["csv"],
                key="prediction_file_uploader",
                on_change=on_prediction_file_change,
            )

            prediction_df_preview = None
            prediction_file_is_valid = False
            prediction_validation_message = ""
            missing_cols = []
            extra_cols = []

            if prediction_file is not None:
                try:
                    prediction_df_preview = pd.read_csv(prediction_file)
                    prediction_df_preview.columns = prediction_df_preview.columns.str.strip()
                    prediction_file_is_valid, prediction_validation_message, missing_cols, extra_cols = (
                        validate_prediction_columns(prediction_df_preview)
                    )

                    if prediction_file_is_valid and not st.session_state.prediction_status.startswith("✅ Predictions generated"):
                        st.session_state.prediction_status = "Upload successful. File is compatible with the model."
                    elif not prediction_file_is_valid:
                        st.session_state.prediction_status = f"❌ {prediction_validation_message}"
                except Exception as e:
                    prediction_validation_message = f"Unable to read the uploaded file: {e}"
                    st.session_state.prediction_status = f"❌ {prediction_validation_message}"
                    prediction_file_is_valid = False

            prediction_status_placeholder = st.empty()
            if st.session_state.prediction_status:
                if st.session_state.prediction_status.startswith("❌"):
                    prediction_status_placeholder.error(st.session_state.prediction_status)
                else:
                    prediction_status_placeholder.success(st.session_state.prediction_status)

            if prediction_file is not None and not prediction_file_is_valid:
                if missing_cols:
                    st.markdown(f"**Missing required columns:** {', '.join(missing_cols)}")
                if extra_cols:
                    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
                    st.markdown(f"**Different / unexpected columns in uploaded file:** {', '.join(extra_cols)}")

            submit_prediction = st.button(
                "Submit File for Predictions",
                use_container_width=True,
                disabled=(prediction_file is None or not prediction_file_is_valid),
            )

            if submit_prediction:
                try:
                    if prediction_file is None or not prediction_file_is_valid or prediction_df_preview is None:
                        raise ValueError("Please upload a valid compatible file before generating predictions.")
                    generate_predictions(prediction_df_preview)
                except Exception as e:
                    st.session_state.prediction_status = f"❌ {e}"

                if st.session_state.prediction_status.startswith("❌"):
                    prediction_status_placeholder.error(st.session_state.prediction_status)
                else:
                    prediction_status_placeholder.success(st.session_state.prediction_status)

            if st.session_state.predict_df is not None and st.session_state.prediction_file is not None:
                with open(st.session_state.prediction_file, "rb") as f:
                    st.download_button(
                        "📥 Download Prediction Results",
                        data=f.read(),
                        file_name="student_dropout_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            clear_prediction = st.button("Clear Prediction Results", use_container_width=True)
            if clear_prediction:
                reset_prediction_state()

        with pred_col2:
            st.markdown("### Prediction Results")
            if st.session_state.predict_df is not None:
                preview_cols = []
                if st.session_state.student_id_column and st.session_state.student_id_column in st.session_state.predict_df.columns:
                    preview_cols.append(st.session_state.student_id_column)
                if st.session_state.student_name_column and st.session_state.student_name_column in st.session_state.predict_df.columns:
                    preview_cols.append(st.session_state.student_name_column)
                preview_cols += ["Dropout Probability", "Prediction"]

                preview_df = st.session_state.predict_df[preview_cols].copy() if preview_cols else st.session_state.predict_df.copy()
                st.dataframe(preview_df, width="stretch", height=280)
            else:
                st.info("Prediction results will appear here after you upload a file and submit it for predictions.")

        st.markdown("---")
        st.subheader("SHAP Explanation for a Specific Student")
        shap_col1, shap_col2 = st.columns([1, 3])

        with shap_col1:
            student_choices = get_student_id_choices_from_predictions()
            selected_student_id = None

            if student_choices:
                selected_student_id = st.selectbox(
                    "Select Student ID",
                    options=student_choices,
                    index=0,
                    key="selected_student_id",
                )
            else:
                selected_student_id = st.text_input("Student ID", placeholder="e.g., A10001")

            explain_clicked = st.button("🔎 Explain Prediction", use_container_width=True, disabled=st.session_state.shap_variance_low)
            clear_shap_clicked = st.button("Clear SHAP Section", use_container_width=True)

            if clear_shap_clicked:
                st.session_state.latest_explanation = None
                st.session_state.latest_plot_bytes = None
                st.session_state.explain_status = ""

            if st.session_state.shap_variance_warning:
                st.warning(st.session_state.shap_variance_warning)

            if explain_clicked:
                try:
                    if not selected_student_id:
                        raise ValueError("Please enter or select a Student ID.")
                    explain_student(selected_student_id)
                except Exception as e:
                    st.session_state.explain_status = f"❌ {e}"

                       # Download button FIRST
            if st.session_state.latest_plot_bytes:
                st.download_button(
                    "📥 Download Explanation Plot",
                    data=st.session_state.latest_plot_bytes,
                    file_name="student_shap_waterfall.png",
                    mime="image/png",
                    use_container_width=True,
                )
            
            # THEN the status message
            if st.session_state.explain_status:
                if st.session_state.explain_status.startswith("✅"):
                    formatted_html = st.session_state.explain_status.replace("\n", "<br>")
                    st.markdown(
                        f"""
                        <div style="padding:0.75rem 1rem; border-radius:0.5rem; background:#d1fae5; color:#065f46;">
                            {formatted_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(st.session_state.explain_status)

            if (
                st.session_state.latest_explanation is not None
                and st.session_state.predict_df is not None
                and selected_student_id
                and st.session_state.explain_status.startswith("✅")
            ):
                student_id_col = st.session_state.student_id_column
                row = st.session_state.predict_df[
                    st.session_state.predict_df[student_id_col].astype(str) == str(selected_student_id)
                ].iloc[0]

                pred_label = row["Prediction"]
                pred_prob = format_probability(row["Dropout Probability Value"])

                summary_html = generate_plain_language_shap_summary(
                    st.session_state.latest_explanation,
                    pred_label,
                    pred_prob,
                )
                render_summary_box(selected_student_id, summary_html)

                recommendations = generate_shap_recommendations(
                    st.session_state.latest_explanation,
                    row["Prediction"] if "Prediction" in row.index else "Unknown",
                    row["Dropout Probability Value"] if "Dropout Probability Value" in row.index else np.nan,
                    row,
                )
                render_recommendation_box(selected_student_id, recommendations)

        with shap_col2:
            render_centered_chart_help(
                "Individual SHAP Waterfall Plot",
                """
This chart explains **why this specific student received their prediction**.

- The starting point is the **average dropout risk** indicated by **E[f(x)]**
- Each feature pushes the dropout risk **higher or lower**

🔴 **Red bars** → increase dropout risk  
🔵 **Blue bars** → decrease dropout risk  

➡️ Bars pushing to the **right** increase risk  
⬅️ Bars pushing to the **left** reduce risk  

- Longer bars mean a **stronger influence** on the prediction  
- The final dropout risk indicated by **f(x)** is based on the combined effect of all displayed factors
""",
                heading_level=3,
            )

            if st.session_state.latest_plot_bytes is not None:
                st.markdown('<div class="shap-plot-frame">', unsafe_allow_html=True)
                plot_container = st.container(height=760, border=False)
                with plot_container:
                    st.image(st.session_state.latest_plot_bytes, width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("The SHAP waterfall plot will appear here after you generate an explanation.")
