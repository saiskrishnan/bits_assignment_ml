import os
import glob
import json
import pickle
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize

# streamlit_app.py
# Streamlit app to select and load a trained model + preprocessing artifacts,
# run predictions on an uploaded CSV and download results.
#
# Place this file in the same directory that contains the saved models folder
# (./models or ./model). Run with: streamlit run streamlit_app.py
st.set_page_config(page_title="Obesity Models Predictor", layout="wide")

st.title("Obesity-level Prediction — Select & Run Model")

# Discover saved models and metadata
def discover_models():
    # map expected display names to joblib filenames (prefer joblib, fall back to pkl)
    # auto-detect model directory: prefer 'models', then 'model', then current dir
    for candidate in ("models", "model", "."):
        if os.path.isdir(candidate):
            base_dir = candidate
            break
    else:
        # fallback to "models" if nothing exists (keeps compatibility)
        base_dir = "models"
    filename_map = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
        "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
    }

    models = {}
    for name, fname in filename_map.items():
        # prefer joblib path as listed, but allow .pkl fallback
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            models[name] = path
            continue
        # try .pkl fallback
        alt = os.path.join(base_dir, os.path.splitext(fname)[0] + ".pkl")
        if os.path.exists(alt):
            models[name] = alt
            continue
        # try either extension in the same folder if different naming was used
        for ext in ("joblib", "pkl"):
            alt2 = os.path.join(base_dir, f"{os.path.splitext(fname)[0]}.{ext}")
            if os.path.exists(alt2):
                models[name] = alt2
                break

    # If none found in the explicit map, fallback to a simple directory scan
    if not models:
        for d in ("models", "model", "."):
            if not os.path.isdir(d):
                continue
            for ext in ("joblib", "pkl"):
                for path in glob.glob(os.path.join(d, f"*.{ext}")):
                    fname = os.path.splitext(os.path.basename(path))[0]
                    display_name = fname.replace("_", " ").title()
                    models[display_name] = path
        # return whatever discovered by fallback
        return models

    return models

models_map = discover_models()
if not models_map:
    st.error("No saved models found. Place .pkl/.joblib files under ./models or ./model, or include models_metadata.json.")
    st.stop()

model_names = sorted(models_map.keys())
if not model_names:
    st.sidebar.error("No models available.")
    selected_models = []
else:
    st.sidebar.header("Model selection")
    selected = st.sidebar.radio("Choose a model to run and display confusion matrix", model_names, index=0)
    st.sidebar.markdown(f"- `{selected}`")
    selected_models = [selected]

# render the reserved input widgets so they appear after the metrics container/sidebar block
def render_input_widgets():
    global uploaded, single_text
    input_subheader_slot.subheader("Input data")
    uploaded = uploader_slot.file_uploader(
        "Upload CSV (rows of feature columns). If no preprocessor found, upload already preprocessed feature columns.",
        type=["csv"],
    )
    single_text = text_slot.text_area(
        "Or paste a single JSON/dict row (feature_name: value). Leave empty if uploading CSV.",
        height=120,
    )

# Attempt to load preprocessing artifacts (label encoder, preprocessor)
def load_preprocessor_labelencoder():
    preprocessor = None
    label_encoder = None
    # prefer joblib .joblib or pickle .pkl in both folders
    candidates = [
        ("model", "preprocessor.joblib", "label_encoder.joblib"),
        ("models", "preprocessor.pkl", "label_encoder.pkl"),
        ("model", "preprocessor.pkl", "label_encoder.pkl"),
        ("models", "preprocessor.joblib", "label_encoder.joblib"),
    ]
    for d, pre_f, lab_f in candidates:
        pre_path = os.path.join(d, pre_f)
        lab_path = os.path.join(d, lab_f)
        if os.path.exists(pre_path):
            try:
                if pre_path.endswith(".joblib"):
                    preprocessor = joblib.load(pre_path)
                else:
                    with open(pre_path, "rb") as f:
                        preprocessor = pickle.load(f)
            except Exception:
                preprocessor = None
        if os.path.exists(lab_path):
            try:
                if lab_path.endswith(".joblib"):
                    label_encoder = joblib.load(lab_path)
                else:
                    with open(lab_path, "rb") as f:
                        label_encoder = pickle.load(f)
            except Exception:
                label_encoder = None
        if preprocessor is not None or label_encoder is not None:
            # return whatever found (could be one of them)
            return preprocessor, label_encoder
    return None, None

preprocessor, label_encoder = load_preprocessor_labelencoder()
if preprocessor is not None:
    # don't display a success message when preprocessor is found
    pass
else:
    st.info("Preprocessor not found (predictions will expect already preprocessed features).")

if label_encoder is not None:
    # don't display a success message when label encoder is found
    pass
else:
    st.info("Label encoder not found (predictions will output encoded labels if classifier uses integer labels).")

# Load selected model helper
@st.cache_resource
def load_model(path: str):
    if path.endswith(".joblib"):
        return joblib.load(path)
    # try joblib anyway (some files may be joblib but named .pkl)
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

# Show basic metadata if present in models_metadata.json
def read_metadata_for(path):
    for meta_dir in ("models", "model"):
        meta_path = os.path.join(meta_dir, "models_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                for e in entries:
                    if os.path.normpath(e.get("file", "")) == os.path.normpath(path) or os.path.basename(e.get("file","")) == os.path.basename(path):
                        return e
            except Exception:
                pass
    return None

# Reserve slots so the input controls can be rendered later (at the end of the sidebar)
input_subheader_slot = st.sidebar.empty()
uploader_slot = st.sidebar.empty()
text_slot = st.sidebar.empty()

# Default sample test set (used when user does not upload a file or paste JSON)
sample_url = "https://raw.githubusercontent.com/saiskrishnan/bits_assignment_ml/main/test_set.csv"

# Initialize variables; they will be assigned when the widgets are rendered at the end
uploaded = None
single_text = None
# Helper to render the input widgets into the reserved slots (call this at the end of the script)



# Use the sample test_set.csv by default when no upload and no pasted JSON provided
if uploaded is None and not single_text:
    uploaded = sample_url
    st.caption(f"Using default sample CSV: {sample_url}")

df_input = None
if uploaded is not None:
    try:
        df_input = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
elif single_text:
    try:
        obj = json.loads(single_text)
        if isinstance(obj, dict):
            df_input = pd.DataFrame([obj])
        elif isinstance(obj, list):
            df_input = pd.DataFrame(obj)
        else:
            st.error("JSON must be an object or array of objects.")
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")

if df_input is None:
    st.info("Upload a CSV or paste JSON to run predictions.")
else:
    st.write("Input preview:")
    st.dataframe(df_input.head())

    # Target column is fixed to "NObeyesdad"
    gt_col = "NObeyesdad"
    st.sidebar.markdown(f"Target column : `{gt_col}`")

    # Preprocess if preprocessor available
    X = df_input.copy()
    try:
        if preprocessor is not None:
            # preprocessor.transform expects the original feature columns used at training
            X_proc = preprocessor.transform(X)
            # if returned sparse matrix-like, convert to array
            if hasattr(X_proc, "toarray"):
                X_proc = X_proc.toarray()
            X_for_model = X_proc
        else:
            # assume uploaded data is already processed and numeric
            X_for_model = X.values
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # If no models selected, nothing to do
    if not selected_models:
        st.info("Choose at least one model to run predictions and display confusion matrix.")
    else:
        # iterate selected models and display confusion matrix (requires ground-truth)
        for model_name in selected_models:
            model_path = models_map[model_name]
            with st.spinner(f"Loading model {model_name}..."):
                try:
                    model = load_model(model_path)
                except Exception as e:
                    st.error(f"Failed to load model {model_name}: {e}")
                    continue

            st.write("Model:", model_name, "->", os.path.basename(model_path))
            # run prediction
            try:
                y_pred_encoded = model.predict(X_for_model)
            except Exception as e:
                st.error(f"Model prediction failed for {model_name}: {e}")
                continue

            # decode labels if possible
            try:
                if label_encoder is not None and hasattr(label_encoder, "inverse_transform") and np.issubdtype(np.array(y_pred_encoded).dtype, np.integer):
                    y_pred = label_encoder.inverse_transform(np.array(y_pred_encoded, dtype=int))
                else:
                    # if model outputs indexes and has classes_, map them
                    if np.issubdtype(np.array(y_pred_encoded).dtype, np.integer) and hasattr(model, "classes_"):
                        classes_arr = np.array(model.classes_)
                        y_pred = classes_arr[np.array(y_pred_encoded, dtype=int)]
                    else:
                        y_pred = np.array(y_pred_encoded)
            except Exception:
                y_pred = np.array(y_pred_encoded)


            # If ground-truth provided and present, compute and show confusion matrix + metrics
            if gt_col and gt_col in df_input.columns:
                y_true = df_input[gt_col].values
                # decide labels ordering
                if label_encoder is not None and hasattr(label_encoder, "classes_"):
                    labels = list(label_encoder.classes_)
                elif hasattr(model, "classes_"):
                    labels = list(np.array(model.classes_, dtype=object))
                else:
                    labels = list(np.unique(np.concatenate([y_true, y_pred]).astype(object)))
                # ensure labels include all present in y_true and y_pred
                labels = list(dict.fromkeys(list(labels) + list(np.unique(np.concatenate([y_true, y_pred]).astype(object)))))

                try:
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                except Exception as e:
                    st.error(f"Failed to compute confusion matrix for {model_name}: {e}")
                    continue

                # compute additional metrics
                try:

                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
                    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
                    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
                    mcc = matthews_corrcoef(y_true, y_pred)

                    # try to get proper probability scores for AUC, fall back to binarized predicted labels
                    auc = None
                    try:
                        if hasattr(model, "predict_proba"):
                            y_score = model.predict_proba(X_for_model)
                        elif hasattr(model, "decision_function"):
                            y_score = model.decision_function(X_for_model)
                            # decision_function for multiclass may return shape (n_samples, n_classes)
                        else:
                            # fallback to one-hot encoded predicted labels (not ideal but acceptable)
                            y_score = label_binarize(y_pred, classes=labels)

                        y_true_bin = label_binarize(y_true, classes=labels)
                        # roc_auc_score requires at least two classes in y_true_bin
                        if y_true_bin.shape[1] >= 2:
                            auc = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")
                    except Exception:
                        auc = None
                except Exception as e:
                    st.error(f"Failed to compute metrics for {model_name}: {e}")
                    accuracy = precision = recall = f1 = mcc = auc = None

                # show confusion matrix plot
                fig, ax = plt.subplots(figsize=(7, 6))
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.set_title(f"Confusion matrix — {model_name}")
                tick_marks = np.arange(len(labels))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_yticklabels(labels)
                thresh = cm.max() / 2. if cm.max() != 0 else 0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                st.pyplot(fig)

                # display raw matrix and simple metrics
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                st.write("Confusion matrix (counts):")
                st.dataframe(cm_df)

                # display the computed metrics in a fixed sidebar column so they update
                metrics = {
                    "accuracy": accuracy,
                    "precision (macro)": precision,
                    "recall (macro)": recall,
                    "f1 (macro)": f1,
                    "mcc": mcc,
                    "auc (ovr, macro)": auc if auc is not None else None
                }

                # format numeric metrics
                def _fmt(v):
                    if v is None:
                        return "N/A"
                    try:
                        if isinstance(v, (float, np.floating)):
                            return round(float(v), 4)
                        return v
                    except Exception:
                        return v

                metrics_fmt = {k: _fmt(v) for k, v in metrics.items()}

                # write metrics to a fixed sidebar panel so they persist and update when model changes
                # use a different variable name so downstream logic doesn't treat it as the download target
                metrics_container = st.sidebar.container()
                metrics_container.header("Model metrics")
                metrics_container.subheader(model_name)
                metrics_df = pd.DataFrame.from_dict(metrics_fmt, orient="index", columns=["value"])
                metrics_container.dataframe(metrics_df)

                # also show a compact summary under the confusion matrix for convenience
                st.write("Metrics (summary):")
                st.table(metrics_df)
            else:
                st.info(f"No ground-truth column provided or column '{gt_col}' not found. Provide ground-truth to view confusion matrix and metrics.")
                # show prediction class counts
                vals = pd.Series(y_pred).value_counts().rename_axis("class").reset_index(name="count")
                st.write("Prediction class distribution:")
                st.dataframe(vals)

            # allow download of the input CSV (placed in the fixed sidebar panel when available)
            try:
                input_csv = df_input.to_csv(index=False).encode("utf-8")
                if isinstance(uploaded, str) and uploaded == sample_url:
                    input_fname = "test_set.csv"
                elif hasattr(uploaded, "name"):
                    input_fname = uploaded.name
                else:
                    input_fname = "input.csv"

                # prefer adding the button to the existing metrics_panel (fixed sidebar) if present,
                # otherwise fall back to st.sidebar
                target_sidebar = metrics_container if "metrics_container" in locals() else st.sidebar
                target_sidebar.download_button(
                    f"Download input CSV ({input_fname})",
                    data=input_csv,
                    file_name=input_fname,
                    mime="text/csv",
                )

                # call the function to actually render the widgets
                render_input_widgets()
            except Exception:
                pass