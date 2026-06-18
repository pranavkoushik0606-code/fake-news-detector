import streamlit as st

st.title("🚀 Model Performance")

st.metric("Accuracy", "98.62%")
st.metric("Precision", "99%")
st.metric("Recall", "99%")
st.metric("F1 Score", "99%")

st.success(
    "Model trained using TF-IDF and Logistic Regression"
)