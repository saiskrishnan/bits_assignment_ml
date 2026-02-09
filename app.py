import os
import glob
import pickle
import joblib
import streamlit as st
import pandas as pd
import tempfile
import subprocess

# New imports for evaluation & plotting
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

st.set_page_config(page_title="Model Loader + Preprocessor + Label Encoder", layout="wide")
st.title("Load multiple ML models and run predictions with optional preprocessor/label encoder")

# --- Configuration / input ---
models_source = st.text_input(
    "Models directory (local folder) or GitHub repo URL (https://github.com/owner/repo)",
    value="https://github.com/saiskrishnan/bits_assignment_ml/tree/main/model",
)
models_dir = models_source

# If a GitHub URL is provided, allow cloning it into a temp directory
if isinstance(models_source, str) and models_source.startswith("http") and "github.com" in models_source:
    if st.button("Clone GitHub repo"):
        tmpdir = tempfile.mkdtemp(prefix="models_repo_")
        try:
            subprocess.check_call(["git", "clone", "--depth", "1", models_source, tmpdir])
            st.success(f"Cloned repo to {tmpdir}")
            models_dir = tmpdir
        except Exception as e:
            st.error(f"Failed to clone repo: {e}")
            models_dir = models_source

allowed_exts = (".joblib", ".pkl", ".sav", ".model")

# --- Helpers ---
def find_files(directory, exts=allowed_exts):
    if not os.path.isdir(directory):
        return []
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    return sorted(files)

def find_csvs(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(glob.glob(os.path.join(directory, "*.csv")))

@st.cache(allow_output_mutation=True)
def load_artifact(path):
    """Try joblib.load then fall back to pickle.load. Return exception on failure."""
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            return e

# --- Discover artifacts ---
all_files = find_files(models_dir)
if not all_files:
    st.warning("No model/artifact files found in the specified directory. Place model/artifact files there or clone a repo.")
    st.stop()

# Prefer explicit filenames for preprocessor/label_encoder if present
preproc_path = None
labelenc_path = None
for p in all_files:
    name = os.path.basename(p).lower()
    if name in ("preprocessor.pkl", "preprocessor.joblib", "preprocessor.sav"):
        preproc_path = p
    if name in ("label_encoder.pkl", "labelencoder.pkl", "label_encoder.joblib"):
        labelenc_path = p

# Sidebar summary
st.sidebar.header("Detected files")
st.sidebar.write(os.path.basename(preproc_path) if preproc_path else "No preprocessor detected")
st.sidebar.write(os.path.basename(labelenc_path) if labelenc_path else "No label encoder detected")

# Build model list: exclude obvious artifacts
artifact_names = {os.path.basename(preproc_path) if preproc_path else None,
                  os.path.basename(labelenc_path) if labelenc_path else None}
model_files = [p for p in all_files if os.path.basename(p) not in artifact_names]

if not model_files:
    st.warning("No model files found (only artifacts detected). Place model files (*.joblib, *.pkl, *.sav) in the folder.")
    st.stop()

# Map display names -> paths
display_names = [os.path.basename(p) for p in model_files]
selection = st.selectbox("Select a model", options=display_names)
selected_path = model_files[display_names.index(selection)]
st.write("Model path:", selected_path)

# Load artifacts (cached)
preprocessor = None
label_encoder = None

if preproc_path:
    preprocessor_or_err = load_artifact(preproc_path)
    if isinstance(preprocessor_or_err, Exception):
        st.error(f"Failed to load preprocessor: {preprocessor_or_err}")
    else:
        preprocessor = preprocessor_or_err
        st.success(f"Loaded preprocessor: {os.path.basename(preproc_path)}")

if labelenc_path:
    labelenc_or_err = load_artifact(labelenc_path)
    if isinstance(labelenc_or_err, Exception):
        st.error(f"Failed to load label encoder: {labelenc_or_err}")
    else:
        label_encoder = labelenc_or_err
        st.success(f"Loaded label encoder: {os.path.basename(labelenc_path)}")

# Load selected model
model_or_err = load_artifact(selected_path)
if isinstance(model_or_err, Exception):
    st.error(f"Failed to load model: {model_or_err}")
    st.stop()

model = model_or_err
st.write("Model type:", type(model))

# show params if available
if hasattr(model, "get_params"):
    try:
        params = model.get_params()
        st.expander("Model parameters (get_params)")(st.json(params))
    except Exception:
        pass

# --- Dataset selection UI ---
st.header("Dataset for prediction & evaluation")

data_source = st.radio("Choose dataset source", options=["Upload CSV", "Use CSV in repo/folder"], index=0)

input_df = None
selected_csv_path = None

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file with features (rows = samples). If available include target column 'NObeyesdad' for evaluation.", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            input_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(input_df.head())
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            st.stop()
else:
    # list csv files in the selected models_dir (repo clone or local folder)
    csv_files = find_csvs(models_dir)
    if not csv_files:
        st.info("No CSV files detected in the selected directory/repo. You can upload a CSV instead.")
    else:
        csv_display = [os.path.basename(p) for p in csv_files]
        csv_choice = st.selectbox("Select CSV from repo/folder", options=csv_display)
        selected_csv_path = csv_files[csv_display.index(csv_choice)]
        if st.button("Load selected CSV"):
            try:
                input_df = pd.read_csv(selected_csv_path)
                st.success(f"Loaded {selected_csv_path}")
                st.dataframe(input_df.head())
            except Exception as e:
                st.error(f"Failed to load CSV: {e}")
                st.stop()

# allow user to specify target column name (default NObeyesdad)
target_column = st.text_input("Target column name (for evaluation)", value="NObeyesdad")

# --- Prediction & Evaluation UI ---
if input_df is None:
    st.info("Provide a dataset (upload or choose from folder) to enable predictions and evaluation.")
else:
    st.header("Make predictions & view evaluation")
    st.write(f"Rows: {len(input_df)}, Columns: {len(input_df.columns)}")
    if st.button("Run prediction and evaluation"):
        try:
            X = input_df.copy()

            # Prepare features for model: drop target column if present
            if target_column in X.columns:
                X_features = X.drop(columns=[target_column])
            else:
                X_features = X

            # Preprocess if available
            X_for_model = X_features
            if preprocessor is not None:
                if hasattr(preprocessor, "transform"):
                    try:
                        X_for_model = preprocessor.transform(X_features)
                    except Exception as e:
                        st.error(f"Preprocessor.transform failed: {e}")
                        st.stop()
                else:
                    st.warning("Loaded preprocessor does not have a transform method. Passing raw features to model.")

            # Predict
            preds = model.predict(X_for_model)

            # If label encoder present, try inverse transform
            decoded_preds = None
            if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
                try:
                    decoded_preds = label_encoder.inverse_transform(preds)
                except Exception:
                    decoded_preds = None

            out = pd.DataFrame({"prediction": preds})
            if decoded_preds is not None:
                out["prediction_label"] = decoded_preds

            # probabilities if available
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_for_model)
                    proba_df = pd.DataFrame(proba, columns=[f"proba_{i}" for i in range(proba.shape[1])])
                    out = pd.concat([out.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
                except Exception:
                    pass

            result = pd.concat([X.reset_index(drop=True), out.reset_index(drop=True)], axis=1)

            # Display and offer download (remove any prediction columns from original if present)
            pred_cols_to_remove = [c for c in result.columns if str(c).lower().startswith("pred_") and c not in ("prediction", "prediction_label")]
            if pred_cols_to_remove:
                result = result.drop(columns=pred_cols_to_remove, errors="ignore")

            st.success("Prediction complete")
            st.dataframe(result.head(200))

            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

            # --- Evaluation (if target available) ---
            if target_column in input_df.columns:
                # create y_true and y_pred labels for comparison
                y_true = input_df[target_column].values

                # Choose comparable y_pred labels
                if decoded_preds is not None:
                    y_pred_labels = decoded_preds
                else:
                    # if y_true are strings and preds numeric and label_encoder present, attempt inverse transform
                    if label_encoder is not None and input_df[target_column].dtype == object:
                        try:
                            y_pred_labels = label_encoder.inverse_transform(preds)
                        except Exception:
                            y_pred_labels = preds
                    else:
                        y_pred_labels = preds

                # If y_true are encoded numbers while y_pred are labels, try to map y_true via label_encoder
                y_true_labels = y_true
                try:
                    # if y_true numeric but label_encoder exists and preds are decoded strings -> convert y_true to strings
                    if label_encoder is not None and np.issubdtype(y_true.dtype, np.number) and (decoded_preds is not None):
                        y_true_labels = label_encoder.inverse_transform(y_true.astype(int))
                except Exception:
                    y_true_labels = y_true

                # Compute metrics (robust to label types)
                acc = accuracy_score(y_true_labels, y_pred_labels)
                prec = precision_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
                rec = recall_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
                f1 = f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
                try:
                    mcc = matthews_corrcoef(y_true_labels, y_pred_labels)
                except Exception:
                    mcc = None

                st.subheader("Evaluation metrics (on provided target column)")
                metrics_df = pd.DataFrame(
                    [
                        {
                            "accuracy": acc,
                            "precision_macro": prec,
                            "recall_macro": rec,
                            "f1_macro": f1,
                            "mcc": mcc,
                        }
                    ]
                )
                st.table(metrics_df.T.rename(columns={0: "value"}))

                # Classification report
                st.subheader("Classification report")
                try:
                    crep = classification_report(y_true_labels, y_pred_labels, output_dict=True, zero_division=0)
                    crep_df = pd.DataFrame(crep).T
                    # pretty rounding
                    for c in ("precision", "recall", "f1-score", "support"):
                        if c in crep_df.columns:
                            crep_df[c] = crep_df[c].apply(lambda v: round(float(v), 4) if pd.notna(v) else v)
                    st.dataframe(crep_df)
                except Exception as e:
                    st.text("Could not compute classification report: " + str(e))

                # Confusion matrix plot
                st.subheader("Confusion matrix")
                try:
                    labels = np.unique(np.concatenate([y_true_labels, y_pred_labels]).astype(str))
                    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    ax.set_title("Confusion matrix")
                    st.pyplot(fig)
                except Exception as e:
                    st.text("Could not compute/plot confusion matrix: " + str(e))
            else:
                st.info(f"Target column '{target_column}' not found in dataset — evaluation skipped. Include '{target_column}' for evaluation.")
        except Exception as e:
            st.error(f"Prediction/evaluation failed: {e}")