import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data_loader import load_breast_cancer_data
from src.utils import (
    ICON_PATH,
    STYLE_PATH,
    LogisticRegressionScratch,
    preprocess_breast_cancer_data,
    add_bias

)

from PIL import Image

icon = Image.open(ICON_PATH)

#===========================================
# Page setup
#===========================================

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon=icon,
    layout="wide"
)

def load_css(path: str) -> None:
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    scratch = joblib.load("models/scratch_tuned.joblib")
    sklearn_ = joblib.load("models/sklearn_tuned.joblib")

    return scratch, sklearn_

def load_data():
    df = load_breast_cancer_data(source="csv")
    return preprocess_breast_cancer_data(df)

# Load model, style, data
load_css(STYLE_PATH)
model_scratch, model_sklearn = load_models()
X_train, X_val, X_test, y_train, y_val, y_test, feature_names, preprocessor = load_data()

# Select threshold
X_test_b = add_bias(X_test)
y_proba_test = model_scratch.predict_proba(X_test_b)
fpr_c, tpr_c, thresholds_c = roc_curve(y_test, y_proba_test)

#===========================================
# Sidebar
#===========================================

st.sidebar.title("Breast Cancer App")
page = st.sidebar.radio(
    "Navigation",
    ["Predict", "Compare models", "Metrics"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.caption("Logistic Regression · Lave Anfield")

#===========================================
# Page Predict
#===========================================

if page == "Predict":
    st.title("Breast Cancer Prediction")
    st.caption("Enter tumor measurements below, then click **Run prediction**.")

    st.markdown("### MODEL")
    model_choice = st.radio(
        "Click the button to choose model",
        ["CUSTOM", "SCIKIT-LEARN"],
        horizontal=True
    )
    
    is_scratch = model_choice == "CUSTOM"
    model = model_scratch if is_scratch else model_sklearn


    st.markdown("#### Tumor Features")

    sample_raw = preprocessor.scaler.inverse_transform(X_test[20].reshape(1, -1))[0]

    cols = st.columns(3)
    inputs = []
    for i, name in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(
                name,
                value=float(round(sample_raw[i], 4)),
                format="%.4f",
                key=name
            )
            inputs.append(val)
    
    st.markdown("")

    if st.button("Run prediction", type="primary", use_container_width=True):
        x_df = pd.DataFrame([inputs], columns=feature_names)
        x_scale = preprocessor.transform(x_df)

        if is_scratch:
            x_scale = add_bias(x_scale)
            prob = float(model.predict_proba(x_scale)[0])
        else:
            prob = float(model.predict_proba(x_scale)[0, 1])
        
        label = "Malignant" if prob >= 0.5 else "Benign"
        confidence = prob if label == "Malignant" else 1 - prob

        col_a, col_b = st.columns(2)
        col_a.metric("Prediction", label)
        col_b.metric("Confidence", f"{confidence:.2%}")
        st.progress(confidence)

        if label == "Benign":
            st.success("The model predicts this tumor is **BENIGN**.")
        else:
            st.error("The model predicts this tumor is **MALIGNANT**.")


#===========================================
# Page Compare models
#===========================================

elif page == "Compare models":
    st.title("Model Comparison")
    st.caption("Side-by-side evaluation on the held-out test set.")

    proba_scratch = np.array([model_scratch.predict_proba(x.reshape(1, -1))[0] for x in X_test_b])
    proba_sklearn = model_sklearn.predict_proba(X_test)[:, 1]

    best_threshold_0 = joblib.load("threshold/lr_thr.joblib")
    best_threshold_1 = joblib.load("threshold/sk_thr.joblib")

    pred_scratch = (proba_scratch >= best_threshold_0).astype(int)
    pred_sklearn = (proba_sklearn >= best_threshold_1).astype(int)

    col1, col2 = st.columns(2)
    for col, name, pred, proba in [
        (col1, "CUSTOM", pred_scratch, proba_scratch),
        (col2, "SCIKIT-LEARN", pred_sklearn, proba_sklearn),
    ]:
        with col:
            st.subheader(name)
            st.metric("Accuracy", f"{accuracy_score(y_test, pred):.4f}")
            st.metric("F1 score", f"{f1_score(y_test, pred):.4f}")
            st.metric("ROC-AUC", f"{roc_auc_score(y_test, proba):.4f}")

    st.markdown("---")
    st.markdown("#### Predicted probability - CUSTOM vs SKLEARN")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(proba_scratch, proba_sklearn, alpha=0.5, s=20, c=y_test, cmap="coolwarm")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Custom model probability")
    ax.set_ylabel("Scikit-learn model probability")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

#===========================================
# Page Metrics
#===========================================

elif page == "Metrics":
    st.title("Model Metrics")
    st.caption("Evaluation on the held-out test set.")

    st.markdown("### MODEL")
    model_choice = st.radio(
        "Click the button to choose model",
        ["CUSTOM", "SCIKIT-LEARN"],
        horizontal=True
    )

    is_scratch = model_choice == "CUSTOM"
    model = model_scratch if is_scratch else model_sklearn
    if is_scratch:
        best_threshold = joblib.load("threshold/lr_thr.joblib")
    else:
        best_threshold = joblib.load("threshold/sk_thr.joblib")

    if is_scratch:
        proba = np.array([model_scratch.predict_proba(x.reshape(1, -1))[0] for x in X_test_b])
    else:
        proba = model_sklearn.predict_proba(X_test)[:, 1]

    pred = (proba >= best_threshold).astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score(y_test, pred):.4f}")
    c2.metric("Precision", f"{precision_score(y_test, pred):.4f}")
    c3.metric("Recall", f"{recall_score(y_test, pred):.4f}")
    c4.metric("F1 score", f"{f1_score(y_test, pred):.4f}")

    st.markdown("---")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("#### Confusion Matrix")

        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(5, 4))

        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign", "Malignant"])
        ax.set_yticklabels(["Benign", "Malignant"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black"
                )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col_roc:
        st.markdown("#### ROC Curve")
        auc = roc_auc_score(y_test, proba)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(fpr_c, tpr_c, linewidth=1.8, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")

        idx = np.argmin(np.abs(thresholds_c - best_threshold))
        ax.scatter(
            fpr_c[idx], tpr_c[idx], color='gold', s=100, zorder=5,
            label=f"Threshold = {best_threshold}\n(FPR={fpr_c[idx]:.2f}, TPR={tpr_c[idx]:.2f})"
        )
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
        ax.legend(fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    