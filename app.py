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
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
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
        "target_column": None,
        "student_id_column": None,
        "student_name_column": None,
        "train_metrics": None,
        "predict_df": None,
        "prediction_file": None,
        "latest_plot": None,
        "shap_explainer": None,
        "is_trained": False,
        "train_success_message": "",
        "prediction_status": "",
        "explain_status": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# Utility functions
# ============================================================
def reset_training_state():
    st.session_state.model = None
    st.session_state.preprocessor = None
    st.session_state.feature_columns = None
    st.session_state.target_column = None
    st.session_state.student_id_column = None
    st.session_state.student_name_column = None
    st.session_state.train_metrics = None
    st.session_state.shap_explainer = None
    st.session_state.is_trained = False
    st.session_state.train_success_message = ""
    reset_prediction_state()


def reset_prediction_state():
    st.session_state.predict_df = None
    st.session_state.prediction_file = None
    st.session_state.latest_plot = None
    st.session_state.prediction_status = ""
    st.session_state.explain_status = ""


def get_model_status_text() -> str:
    if st.session_state.is_trained:
        return "✅ Model status: Trained successfully. Tab 2 is ready for prediction and SHAP explanations."
    return "⚠️ Model status: Not trained yet. Please complete Tab 1 before using Tab 2."


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


def validate_prediction_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, "Uploaded file is missing required columns: " + ", ".join(missing_cols)
    return True, ""


def generate_metrics_table(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": [round(float(v), 4) if pd.notna(v) else np.nan for v in metrics.values()],
    })


def save_prediction_results(df: pd.DataFrame) -> str:
    path = os.path.join(tempfile.gettempdir(), "student_dropout_predictions.csv")
    df.to_csv(path, index=False)
    return path

def save_shap_plot(explanation, max_display: int = 12) -> str:
    plt.close("all")
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    path = os.path.join(tempfile.gettempdir(), "student_shap_waterfall.png")

    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def create_shap_explanation(explainer, X_row_transformed: np.ndarray, feature_names: List[str]):
    shap_values = explainer(X_row_transformed)

    values = shap_values.values
    base_values = shap_values.base_values

    if np.ndim(values) == 3:
        if values.shape[-1] == 2:
            values_1d = values[0, :, 1]
        else:
            values_1d = values[0, :, 0]
    elif np.ndim(values) == 2:
        values_1d = values[0]
    else:
        values_1d = values

    if np.ndim(base_values) == 2:
        if base_values.shape[-1] == 2:
            base_value = base_values[0, 1]
        else:
            base_value = base_values[0, 0]
    elif np.ndim(base_values) == 1:
        if len(base_values) == 2:
            base_value = base_values[1]
        else:
            base_value = base_values[0]
    else:
        base_value = base_values

    row_data = X_row_transformed[0] if np.ndim(X_row_transformed) > 1 else X_row_transformed

    # FIXED indentation here
    safe_feature_names = [str(name)[:60] for name in feature_names]

    return shap.Explanation(
        values=values_1d,
        base_values=base_value,
        data=row_data,
        feature_names=safe_feature_names,
    )


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


def train_institution_model(df: pd.DataFrame, target_column: str, student_id_column: str, student_name_column: str, test_size: float):
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed. Please add streamlit and xgboost to requirements.txt.")

    reset_training_state()
    df = df.copy()
    df.columns = df.columns.str.strip()

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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=42,
        stratify=y if y.nunique() == 2 else None,
    )

    preprocessor = build_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train_transformed, y_train)

    y_prob = model.predict_proba(X_test_transformed)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else np.nan,
    }

    st.session_state.model = model
    st.session_state.preprocessor = preprocessor
    st.session_state.feature_columns = X.columns.tolist()
    st.session_state.target_column = target_column
    st.session_state.student_id_column = student_id_column if student_id_column in df.columns else None
    st.session_state.student_name_column = student_name_column if student_name_column in df.columns else None
    st.session_state.train_metrics = metrics
    st.session_state.shap_explainer = shap.TreeExplainer(model)
    st.session_state.is_trained = True
    st.session_state.train_success_message = (
        f"✅ Model trained successfully on {len(df)} records with {X.shape[1]} feature columns. "
        "You can now use Tab 2 for prediction and SHAP explanations."
    )


def generate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if not st.session_state.is_trained or st.session_state.model is None or st.session_state.preprocessor is None:
        raise RuntimeError("Please train the institution model first in Tab 1.")

    df = df.copy()
    df.columns = df.columns.str.strip()

    is_valid, validation_message = validate_prediction_columns(df, st.session_state.feature_columns)
    if not is_valid:
        raise ValueError(validation_message)

    X_new = df[st.session_state.feature_columns].copy()
    X_new_transformed = st.session_state.preprocessor.transform(X_new)
    probs = st.session_state.model.predict_proba(X_new_transformed)[:, 1]
    preds = (probs >= 0.5).astype(int)

    result_df = df.copy()
    result_df["Dropout Probability"] = np.round(probs, 4)
    result_df["Prediction"] = np.where(preds == 1, "Dropout", "No Dropout")

    st.session_state.predict_df = result_df
    st.session_state.prediction_file = save_prediction_results(result_df)
    st.session_state.prediction_status = f"✅ Predictions generated for {len(result_df)} students."

    return result_df


def explain_student(chosen_id: str) -> str:
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

    explanation = create_shap_explanation(
        st.session_state.shap_explainer,
        X_row_transformed,
        get_transformed_feature_names(st.session_state.preprocessor),
    )

    plot_path = save_shap_plot(explanation)
    st.session_state.latest_plot = plot_path

    pred_label = row["Prediction"].iloc[0] if "Prediction" in row.columns else "Unknown"
    pred_prob = row["Dropout Probability"].iloc[0] if "Dropout Probability" in row.columns else np.nan

    st.session_state.explain_status = (
        f"✅ SHAP explanation generated for Student ID {chosen_id}. "
        f"Prediction: {pred_label} | Dropout Probability: {pred_prob:.4f}"
    )
    return plot_path


# ============================================================
# UI
# ============================================================
st.title("🎓 Student Dropout Predictor with SHAP Explainer")
st.write(
    "Upload a CSV file containing student records to train an institution-specific XGBoost model, "
    "then generate dropout predictions and SHAP-based explanations."
)

train_tab, predict_tab = st.tabs(["🏫 Train Institution Model", "📊 Predict + Explain"])

with train_tab:
    st.subheader("Train Institution Model")
    col1, col2 = st.columns([1, 1])

    with col1:
        training_file = st.file_uploader(
            "📄 Upload Labeled Training CSV",
            type=["csv"],
            key="training_file_uploader",
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

                if st.button("🚀 Train Model", width="stretch"):
                    train_institution_model(
                        training_df_preview,
                        target_column,
                        None if student_id_column == "None" else student_id_column,
                        None if student_name_column == "None" else student_name_column,
                        test_size,
                    )

            except Exception as e:
                st.error(f"Training file could not be read: {e}")

        if st.session_state.train_success_message:
            st.success(st.session_state.train_success_message)

    with col2:
        st.markdown("### Validation Metrics")
        if st.session_state.train_metrics is not None:
            st.dataframe(generate_metrics_table(st.session_state.train_metrics), width="stretch")
        else:
            st.info("Validation metrics will appear here after model training.")

with predict_tab:
    st.info(get_model_status_text())

    if not st.session_state.is_trained:
        st.warning("Please train the model in Tab 1 before using this section.")
    else:
        st.subheader("Upload & Predict")
        pred_col1, pred_col2 = st.columns([1, 2])

        with pred_col1:
            prediction_file = st.file_uploader(
                "📄 Upload Student CSV File",
                type=["csv"],
                key="prediction_file_uploader",
            )

            button_col1, button_col2 = st.columns(2)
            with button_col1:
                submit_prediction = st.button("Submit File", width="stretch")
            with button_col2:
                clear_prediction = st.button("Clear", width="stretch")

            if clear_prediction:
                reset_prediction_state()
                st.rerun()

            if prediction_file is not None and submit_prediction:
                try:
                    prediction_df = pd.read_csv(prediction_file)
                    prediction_df.columns = prediction_df.columns.str.strip()
                    generate_predictions(prediction_df)
                except Exception as e:
                    st.session_state.prediction_status = f"❌ {e}"

            if st.session_state.prediction_status:
                if st.session_state.prediction_status.startswith("✅"):
                    st.success(st.session_state.prediction_status)
                else:
                    st.error(st.session_state.prediction_status)

            if st.session_state.prediction_file and os.path.exists(st.session_state.prediction_file):
                with open(st.session_state.prediction_file, "rb") as f:
                    st.download_button(
                        "📥 Download Prediction Results",
                        data=f,
                        file_name="student_dropout_predictions.csv",
                        mime="text/csv",
                        width="stretch",
                    )

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
                st.info("Prediction results will appear here after you upload a file and click Submit File.")

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
                )
            else:
                typed_student_id = st.text_input("Student ID", placeholder="e.g., A10001")
                selected_student_id = typed_student_id

            explain_clicked = st.button("🔎 Explain Prediction", width="stretch")
            clear_shap_clicked = st.button("Clear SHAP Section", width="stretch")

            if clear_shap_clicked:
                st.session_state.latest_plot = None
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
                    st.success(st.session_state.explain_status)
                else:
                    st.error(st.session_state.explain_status)

            if st.session_state.latest_plot and os.path.exists(st.session_state.latest_plot):
                with open(st.session_state.latest_plot, "rb") as f:
                    st.download_button(
                        "📥 Download Explanation Plot",
                        data=f,
                        file_name="student_shap_waterfall.png",
                        mime="image/png",
                        width="stretch",
                    )

        with shap_col2:
            if st.session_state.latest_plot and os.path.exists(st.session_state.latest_plot):
                st.image(st.session_state.latest_plot, caption="SHAP Waterfall Plot", width="stretch")
            else:
                st.info("The SHAP waterfall plot will appear here after you generate an explanation.")
