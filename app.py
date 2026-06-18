import streamlit as st
import pickle
from utils.database import create_table, save_prediction, get_predictions
# Create database table
create_table()

# Load Model and Vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Page Config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Sidebar
st.sidebar.title("📰 Navigation")

st.sidebar.info("""
Built by Pranav Koushik

### Tech Stack
- Python
- NLP
- Scikit-Learn
- Streamlit
- SQLite
""")

# Main Title
st.title("📰 AI Fake News Detector")

st.markdown("""
Detect whether a news article is **REAL** or **FAKE**
using Machine Learning and Natural Language Processing.
""")

# Input
news = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste any news article here..."
)

# Predict Button
if st.button("🔍 Predict"):

    if news.strip() == "":
        st.warning("⚠ Please enter some news text.")
    else:

        # Transform text
        vectorized_text = vectorizer.transform([news])

        # Prediction
        prediction = model.predict(vectorized_text)

        # Probability
        probability = model.predict_proba(vectorized_text)

        confidence = max(probability[0]) * 100

        # Display Result
        st.subheader("Prediction Result")

        if prediction[0] == 1:
            result = "REAL NEWS"
            st.success("✅ REAL NEWS")
        else:
            result = "FAKE NEWS"
            st.error("❌ FAKE NEWS")

        # Save Prediction
        save_prediction(news, result)

        # Confidence Score
        st.subheader("Confidence Score")

        st.write(f"**{confidence:.2f}%**")

        st.progress(int(confidence))

# History Section
st.markdown("---")

st.subheader("📜 Recent Predictions")

history = get_predictions()

if len(history) > 0:

    for row in history[:10]:

        st.write(f"**Prediction:** {row[2]}")
        st.write(f"**Date:** {row[3]}")
        st.write(f"**News:** {row[1][:150]}...")
        st.write("---")

else:
    st.info("No predictions available yet.")