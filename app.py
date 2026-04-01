import io
import os
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Utility functions
# ============================================================
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
    reset_prediction_state()


def reset_prediction_state():
    st.session_state.predict_df = None
    st.session_state.prediction_file = None
    st.session_state.latest_explanation = None
    st.session_state.latest_plot_bytes = None
    st.session_state.prediction_status = ""
    st.session_state.explain_status = ""


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
    positive_map = {
        "dropout", "dropped", "drop", "yes", "y", "1", "true", "at risk", "atrisk"
    }
    negative_map = {
        "no dropout", "no", "n", "0", "false", "continue", "retained", "stayed"
    }

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
        label = str(name)
        label = label.replace("num__", "")
        label = label.replace("cat__", "")
        label = label.replace("_", " ")
        label = label[:70]
        cleaned.append(label)
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
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

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
        return sorted(
            st.session_state.predict_df[student_id_col].astype(str).dropna().unique().tolist()
        )
    return []


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
    if model_choice == "Run all 4 and choose the best":
        best_model_name = choose_best_model(comparison_df, selection_metric)
    else:
        best_model_name = model_choice

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

    render_training_status(status_placeholder, "Building local SHAP explainer...")
    X_local_background = X_train_t[: min(200, len(X_train_t))]
    local_explainer = get_local_probability_explainer(best_model, X_local_background)

    render_training_status(status_placeholder, "Generating global SHAP plots...")
    X_global_background = X_train_t[: min(200, len(X_train_t))]
    global_explainer = get_fast_global_explainer(best_model, best_model_name, X_global_background)

    global_sample_size = min(300, X_train_t.shape[0])
    sample_idx = np.random.RandomState(42).choice(
        X_train_t.shape[0],
        size=global_sample_size,
        replace=False,
    )
    X_shap_sample = X_train_t[sample_idx]

    importance_bytes, summary_bytes = build_global_shap_plots_fast(
        global_explainer,
        best_model_name,
        X_shap_sample,
        feature_names,
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
    st.session_state.prediction_status = f"✅ Predictions generated for {len(result_df)} students."

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

                train_button = st.button("🚀 Train Model", width="stretch")
                training_status_placeholder = st.empty()

                if train_button:
                    st.session_state.train_metrics = None
                    st.session_state.global_importance_plot_bytes = None
                    st.session_state.global_summary_plot_bytes = None
                    st.session_state.train_success_message = ""
                    st.session_state.selected_model_name = None
                    st.session_state.model_comparison_df = None

                    render_training_status(training_status_placeholder, "Preparing training data...")

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
                        st.rerun()

            except Exception as e:
                st.error(f"Training file could not be read: {e}")

        if st.session_state.train_success_message:
            st.success(st.session_state.train_success_message)

    with col2:
        st.markdown("### Model Performance Metrics")
        if st.session_state.train_metrics is not None:
            metrics_df = generate_metrics_table(st.session_state.train_metrics)
            st.dataframe(metrics_df, width=380, hide_index=True)

            if st.session_state.selected_model_name is not None:
                st.markdown(f"**Selected Model:** {st.session_state.selected_model_name}")

            if (
                st.session_state.model_comparison_df is not None
                and len(st.session_state.model_comparison_df) > 1
            ):
                st.markdown("### Model Comparison")
                st.dataframe(st.session_state.model_comparison_df, width="stretch", hide_index=True)

            st.markdown("### Global SHAP Summary")

            plot_col1, plot_col2 = st.columns(2)

            with plot_col1:
                title_col, help_col = st.columns([6, 1])
                with title_col:
                    st.markdown("**Feature Importance Plot**")
                with help_col:
                    with st.popover("ℹ️"):
                        st.markdown("""
This chart shows the **most important factors** affecting dropout risk overall.

- Longer bars mean a factor has a **stronger influence**
- Features at the top are the **most important**

This chart shows **importance only**, not whether a factor increases or decreases risk.
""")
                if st.session_state.global_importance_plot_bytes is not None:
                    st.image(st.session_state.global_importance_plot_bytes, width="stretch")
                else:
                    st.info("Feature importance plot will appear here after model training.")

            with plot_col2:
                title_col, help_col = st.columns([6, 1])
                with title_col:
                    st.markdown("**Summary Plot**")
                with help_col:
                    with st.popover("ℹ️"):
                        st.markdown("""
This chart shows how different factors influence dropout risk **across all students**.

- Each dot represents **one student**
- Red dots usually indicate **higher feature values**
- Blue dots usually indicate **lower feature values**

Dots to the **right** tend to increase dropout risk.  
Dots to the **left** tend to decrease dropout risk.
""")
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
        pred_col1, pred_col2 = st.columns([1, 2])

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
                width="stretch",
                disabled=(prediction_file is None or not prediction_file_is_valid),
            )

            if submit_prediction:
                if prediction_file is None or not prediction_file_is_valid:
                    st.session_state.prediction_status = "❌ Please upload a valid compatible file before generating predictions."
                    prediction_status_placeholder.error(st.session_state.prediction_status)
                else:
                    try:
                        prediction_file.seek(0)
                        prediction_df_for_prediction = pd.read_csv(prediction_file)
                        prediction_df_for_prediction.columns = prediction_df_for_prediction.columns.str.strip()
                        result_df = generate_predictions(prediction_df_for_prediction)
                        prediction_status_placeholder.success(
                            f"✅ Predictions generated successfully for {len(result_df)} records."
                        )
                    except Exception as e:
                        st.session_state.prediction_status = f"❌ {e}"
                        prediction_status_placeholder.error(st.session_state.prediction_status)

            if st.session_state.prediction_file and os.path.exists(st.session_state.prediction_file):
                with open(st.session_state.prediction_file, "rb") as f:
                    st.download_button(
                        "📥 Download Prediction Results",
                        data=f.read(),
                        file_name="student_dropout_predictions.csv",
                        mime="text/csv",
                        width="stretch",
                    )
            else:
                st.button(
                    "📥 Download Prediction Results",
                    width="stretch",
                    disabled=True,
                )

            clear_prediction = st.button(
                "Clear Prediction Results",
                width="stretch",
            )

            if clear_prediction:
                reset_prediction_state()
                st.rerun()

        with pred_col2:
            st.markdown("### Prediction Results")
            if st.session_state.predict_df is not None:
                preview_cols = []
                if (
                    st.session_state.student_id_column
                    and st.session_state.student_id_column in st.session_state.predict_df.columns
                ):
                    preview_cols.append(st.session_state.student_id_column)
                if (
                    st.session_state.student_name_column
                    and st.session_state.student_name_column in st.session_state.predict_df.columns
                ):
                    preview_cols.append(st.session_state.student_name_column)
                preview_cols += ["Dropout Probability", "Prediction"]

                preview_df = (
                    st.session_state.predict_df[preview_cols].copy()
                    if preview_cols else st.session_state.predict_df.copy()
                )
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
                typed_student_id = st.text_input("Student ID", placeholder="e.g., A10001")
                selected_student_id = typed_student_id

            explain_clicked = st.button("🔎 Explain Prediction", width="stretch")
            clear_shap_clicked = st.button("Clear SHAP Section", width="stretch")

            if clear_shap_clicked:
                st.session_state.latest_explanation = None
                st.session_state.latest_plot_bytes = None
                st.session_state.explain_status = ""
                st.rerun()

            if explain_clicked:
                try:
                    if not selected_student_id:
                        raise ValueError("Please enter or select a Student ID.")
                    explain_student(selected_student_id)
                except Exception as e:
                    st.session_state.explain_status = f"❌ {e}"

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

            if st.session_state.latest_plot_bytes:
                st.download_button(
                    "📥 Download Explanation Plot",
                    data=st.session_state.latest_plot_bytes,
                    file_name="student_shap_waterfall.png",
                    mime="image/png",
                    width="stretch",
                )

        with shap_col2:
            title_col, help_col = st.columns([6, 1])
            with title_col:
                st.markdown("### Individual SHAP Waterfall Plot")
            with help_col:
                with st.popover("ℹ️"):
                    st.markdown("""
This plot explains **why this specific student** was predicted as Dropout or No Dropout.

- Bars pushing to the **right** increase dropout risk
- Bars pushing to the **left** decrease dropout risk
- Larger bars mean a **stronger effect**
- The final prediction is based on the combined effect of all displayed factors
""")

            if st.session_state.latest_plot_bytes is not None:
                st.markdown('<div class="shap-plot-frame">', unsafe_allow_html=True)
                plot_container = st.container(height=760, border=False)
                with plot_container:
                    st.image(st.session_state.latest_plot_bytes, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("The SHAP waterfall plot will appear here after you generate an explanation.")
