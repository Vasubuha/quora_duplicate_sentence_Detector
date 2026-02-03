import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Duplicate Sentence Detector",
    page_icon="🔍",
    layout="centered"
)

# -------------------------------------------------
# Load models (cached)
# -------------------------------------------------
@st.cache_resource
def load_models():
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    lr = joblib.load(r"D:\projects\quora_question_pair\lr_similarity.joblib")
    return sbert, lr

sbert, lr = load_models()

# -------------------------------------------------
# Prediction logic
# -------------------------------------------------
def predict_duplicate(q1: str, q2: str, threshold: float):
    emb1 = sbert.encode(q1)
    emb2 = sbert.encode(q2)

    sim = cosine_similarity(
        emb1.reshape(1, -1),
        emb2.reshape(1, -1)
    )[0][0]

    prob = lr.predict_proba([[sim]])[0][1]
    label = int(prob >= threshold)

    return sim, prob, label

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🔍 Duplicate Sentence Detector")
st.write(
    "Detect whether two questions are **semantically duplicates** "
    "using **Sentence-BERT embeddings**."
)

q1 = st.text_area("Question 1", height=100)
q2 = st.text_area("Question 2", height=100)

threshold = st.slider(
    "Decision Threshold",
    min_value=0.5,
    max_value=0.8,
    value=0.6,
    step=0.05
)

if st.button("Check Duplicate"):
    if not q1.strip() or not q2.strip():
        st.warning("Please enter both questions.")
    else:
        with st.spinner("Analyzing..."):
            sim, prob, label = predict_duplicate(q1, q2, threshold)

        st.subheader("Result")

        if label == 1:
            st.success("✅ These questions are DUPLICATES")
        else:
            st.error("❌ These questions are NOT duplicates")

        st.markdown("### Scores")
        st.write(f"**SBERT Similarity:** `{sim:.3f}`")
        st.write(f"**Duplicate Probability:** `{prob:.3f}`")
