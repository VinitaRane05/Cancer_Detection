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

# ================================================================
# WHITE MAIN BACKGROUND + ROYAL BLUE SIDEBAR (FINAL THEME)
# ================================================================
st.markdown("""
    <style>

        /* MAIN AREA */
        .main {
            background-color: #ffffff !important;
            color: #0a0a0a !important;
        }

        /* All text readable */
        h1, h2, h3, h4, p, label, span, div {
            color: #0a0a0a !important;
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background-color: #0047ab !important; /* Royal blue */
            color: white !important;
        }

        /* Sidebar text */
        section[data-testid="stSidebar"] * {
            color: white !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #0047ab !important;
            color: white !important;
            border-radius: 6px;
            border: none;
        }

        /* Inputs readable */
        .stTextInput input, .stNumberInput input, textarea {
            background-color: #f7f7f7 !important;
            color: #000 !important;
        }

        /* DataFrame white */
        .stDataFrame, .stTable, .dataframe {
            background-color: white !important;
            color: #0a0a0a !important;
        }

        /* Expanders */
        .streamlit-expanderHeader, .streamlit-expanderContent {
            background-color: #f5f5f5 !important;
            color: #0a0a0a !important;
        }

    </style>
""", unsafe_allow_html=True)

# ================================================================
# LOAD DATA + TRAIN MODEL
# ================================================================
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
target_names = data.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=120, random_state=42))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ================================================================
# PAGE UI
# ================================================================
st.title("Breast Cancer Detection System")
st.markdown("<hr style='border:1px solid #cce0ff;'>", unsafe_allow_html=True)

st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose input type", ["Sample", "Manual", "CSV Upload"])
manual_detail = st.sidebar.radio("Manual mode", ["Quick (5 features)", "Advanced (30 features)"])

with st.sidebar.expander("Model & Data summary", expanded=True):
    st.write("Algorithm: RandomForest (StandardScaler pipeline)")
    st.write(f"Dataset: 569 samples, 30 features")
    st.write(f"Test Accuracy: **{acc:.3f}**")
    if st.checkbox("Show classification report", value=False):
        st.text(classification_report(y_test, y_pred, target_names=target_names))

# ================================================================
# PREDICTION RESULT CARD
# ================================================================
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
        <p style="font-size:18px;">
            <b style="color:#ff1a1a;">Malignant: {malignant_pct:.2f}%</b><br>
            <b style="color:#009933;">Benign: {benign_pct:.2f}%</b>
        </p>
    """, unsafe_allow_html=True)

    st.write("### Features used")
    st.dataframe(df.transpose())

# ================================================================
# CHART FUNCTIONS
# ================================================================
def plot_class_distribution():
    counts = y.value_counts().sort_index()
    labels = ['malignant', 'benign']
    fig, ax = plt.subplots()
    ax.pie(counts.values, labels=labels, autopct='%1.1f%%', startangle=90,
           wedgeprops=dict(edgecolor='w'))
    ax.set_title("Class distribution")
    st.pyplot(fig)

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['malignant','benign'])
    ax.set_yticklabels(['malignant','benign'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha='center', va='center', color='black')
    fig.colorbar(im)
    st.pyplot(fig)

def plot_feature_importances(model_pipeline, top_n=12):
    rf = model_pipeline.named_steps['randomforestclassifier']
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:top_n]
    fig, ax = plt.subplots()
    imp[::-1].plot.barh(ax=ax)
    ax.set_title("Top Features")
    st.pyplot(fig)

with st.expander("Model dashboard (charts & explainability)", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        plot_class_distribution()
    with col2:
        plot_confusion_matrix(cm)
    plot_feature_importances(model)

# ================================================================
# INPUT MODES
# ================================================================
if mode == "Sample":
    st.subheader("Pick a sample from the dataset")
    idx = st.number_input("Sample index", min_value=0, max_value=len(X)-1, value=0)
    sample = X.iloc[[idx]]
    st.dataframe(sample)
    if st.button("Predict Sample"):
        predict_and_display(sample)

elif mode == "Manual":
    if manual_detail.startswith("Quick"):
        st.subheader("Quick Manual Input")
        quick_features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]
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
        st.subheader("Advanced Manual Input (30 features)")
        with st.form("adv_form"):
            vals = {}
            cols = st.columns(2)
            defaults = X.median()
            for i, feat in enumerate(X.columns):
                with cols[i % 2]:
                    vals[feat] = st.number_input(feat, value=float(defaults[feat]))
            if st.form_submit_button("Predict"):
                df = pd.DataFrame([vals])
                predict_and_display(df)

else:
    st.header("CSV Upload for Batch Predictions")
    file = st.file_uploader("Upload CSV with 30 features", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())
        if df.shape[1] != X.shape[1]:
            st.error(f"CSV must have {X.shape[1]} columns.")
        else:
            preds = model.predict(df)
            probs = model.predict_proba(df)
            df["prediction"] = [target_names[p].upper() for p in preds]
            df["malignant_pct"] = (probs[:,0] * 100).round(2)
            df["benign_pct"] = (probs[:,1] * 100).round(2)
            st.success("Prediction complete.")
            st.dataframe(df)
            st.download_button("Download CSV", df.to_csv(index=False), "predictions.csv")

# ================================================================
# MODEL DOWNLOAD
# ================================================================
st.markdown("---")
st.subheader("Model Export")
buf = io.BytesIO()
pickle.dump(model, buf)
buf.seek(0)
st.download_button("Download trained model (.pkl)", buf, "breast_cancer_model.pkl")

st.caption("Educational use only. Not for medical diagnosis or treatment.")
