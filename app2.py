import streamlit as st
import numpy as np
import pandas as pd
import io
import pickle
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

# -----------------------
# Theme toggles / CSS
# -----------------------
dark_mode = st.sidebar.checkbox("Enable dark mode", value=False)

if not dark_mode:
    # Blue + white
    st.markdown("""
        <style>
            .main { background-color: #f5faff; }
            section[data-testid="stSidebar"] { background-color: #e7f0ff; }
            h1, h2, h3, h4 { color: #003d99; }
            button[kind="primary"] { background-color: #0066cc !important; color: white !important; border-radius: 8px !important; }
            .stButton>button { border-radius:8px; }
        </style>
    """, unsafe_allow_html=True)
else:
    # Simple dark theme
    st.markdown("""
        <style>
            .main { background-color: #0f1720; color: #e6eef8; }
            section[data-testid="stSidebar"] { background-color: #071224; color: #cfe8ff; }
            h1, h2, h3, h4 { color: #9cc8ff; }
            .stButton>button { border-radius:8px; color: #000; }
            .css-1d391kg { color: #e6eef8; }
        </style>
    """, unsafe_allow_html=True)

# -----------------------
# Load dataset and train model
# -----------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
target_names = data.target_names  # ['malignant' 'benign']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Pipeline
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=120, random_state=42))
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# -----------------------
# Page layout
# -----------------------
st.title("Breast Cancer Detection System")
st.markdown("<hr style='border:1px solid #cce0ff;'>", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose input type", ["Sample", "Manual", "CSV Upload"])
manual_detail = st.sidebar.radio("Manual mode", ["Quick (5 features)", "Advanced (30 features)"])

# Quick info card in sidebar
with st.sidebar.expander("Model & Data summary", expanded=True):
    st.write("Algorithm: RandomForest (with StandardScaler pipeline)")
    st.write(f"Dataset: sklearn.datasets.load_breast_cancer (569 samples, 30 features)")
    st.write(f"Test Accuracy: **{acc:.3f}**")
    if st.checkbox("Show classification report", value=False):
        st.text(classification_report(y_test, y_pred, target_names=target_names))

# -----------------------
# Prediction helper (styled)
# -----------------------
def predict_and_display(df):
    prob = model.predict_proba(df)[0]
    pred = model.predict(df)[0]

    malignant_pct = prob[0] * 100
    benign_pct = prob[1] * 100

    st.write("### Prediction Result")
    st.markdown(f"""
        <div style="padding:15px; border-radius:10px; background:white;
                    border-left:6px solid {'#ff4d4d' if pred==0 else '#00b36b'};">
            <h3 style='margin:0;'>Result: {target_names[pred].upper()}</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <p style="font-size:18px; margin-top:10px;">
            <b style="color:#ff1a1a;">Malignant: {malignant_pct:.2f}%</b><br>
            <b style="color:#009933;">Benign: {benign_pct:.2f}%</b>
        </p>
    """, unsafe_allow_html=True)

    st.write("### Features used")
    st.dataframe(df.transpose())

# -----------------------
# Add charts: class distribution, confusion matrix, feature importance
# -----------------------
def plot_class_distribution():
    counts = y.value_counts().sort_index()  # index 0=malignant,1=benign
    labels = ['malignant', 'benign']
    fig, ax = plt.subplots()
    ax.pie(counts.values, labels=labels, autopct='%1.1f%%', startangle=90,
           wedgeprops=dict(edgecolor='w'))
    ax.set_title("Class distribution (full dataset)")
    st.pyplot(fig)

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title("Confusion matrix (test set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['malignant','benign']); ax.set_yticklabels(['malignant','benign'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

def plot_feature_importances(model_pipeline, top_n=10):
    # extract feature importances from RandomForest (last step)
    rf = model_pipeline.named_steps[list(model_pipeline.named_steps.keys())[-1]]
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:top_n]
    fig, ax = plt.subplots(figsize=(6,4))
    feat_imp[::-1].plot.barh(ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances (RandomForest)")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# Show charts in an expandable area
with st.expander("Model dashboard (charts & explainability)", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        plot_class_distribution()
        st.write("Dataset counts (malignant=0, benign=1):")
        st.write(y.value_counts())
    with col2:
        plot_confusion_matrix(cm)
        plot_feature_importances(model, top_n=12)

# -----------------------
# Input modes
# -----------------------
# SAMPLE MODE
if mode == "Sample":
    st.subheader("Pick a sample from the built-in dataset")
    idx = st.number_input("Enter sample index", min_value=0, max_value=len(X)-1, value=0)
    sample = X.iloc[[idx]]
    st.dataframe(sample)
    if st.button("Predict Sample"):
        predict_and_display(sample)

# MANUAL MODE
elif mode == "Manual":
    if manual_detail.startswith("Quick"):
        st.subheader("Manual Entry (Quick Mode)")
        quick_features = [
            "mean radius", "mean texture", "mean perimeter",
            "mean area", "mean smoothness"
        ]
        defaults = X[quick_features].median()
        inp = {}
        cols = st.columns(2)
        for i, feat in enumerate(quick_features):
            with cols[i % 2]:
                inp[feat] = st.number_input(feat, value=float(defaults[feat]), step=0.1)
        df = pd.DataFrame([inp])
        for col in X.columns:
            if col not in df.columns:
                df[col] = X[col].median()
        df = df[X.columns]
        if st.button("Predict"):
            predict_and_display(df)
    else:
        st.subheader("Advanced Manual Entry (30 features)")
        with st.form("adv_form"):
            vals = {}
            defaults = X.median()
            cols = st.columns(2)
            for i, feat in enumerate(X.columns):
                with cols[i % 2]:
                    vals[feat] = st.number_input(feat, value=float(defaults[feat]))
            submitted = st.form_submit_button("Predict")
            if submitted:
                df = pd.DataFrame([vals])
                predict_and_display(df)

# CSV UPLOAD MODE
else:
    st.header("Upload CSV for Batch Prediction")
    uploaded = st.file_uploader("Upload CSV with 30 features (no header check).", type="csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("### Preview of uploaded file")
            st.dataframe(df.head())

            if df.shape[1] != X.shape[1]:
                st.error(f"CSV must have {X.shape[1]} columns. Yours has {df.shape[1]}.")
            else:
                preds = model.predict(df)
                probs = model.predict_proba(df)

                result = df.copy()
                result["prediction"] = [target_names[p].upper() for p in preds]
                # percentages (rounded)
                result["malignant_pct"] = (probs[:, 0] * 100).round(2)
                result["benign_pct"] = (probs[:, 1] * 100).round(2)

                st.success("Prediction complete.")
                st.dataframe(result)

                csv_out = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results CSV",
                    data=csv_out,
                    file_name="breast_cancer_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

# -----------------------
# Model download (pickle)
# -----------------------
st.markdown("---")
st.subheader("Model export")
st.write("Download the trained model (pickle). Useful to load later without retraining.")
buf = io.BytesIO()
pickle.dump(model, buf)
buf.seek(0)
st.download_button("Download trained model (.pkl)", data=buf, file_name="breast_cancer_rf_pipeline.pkl", mime="application/octet-stream")

st.caption("Educational use only. Not for medical diagnosis or treatment.")
