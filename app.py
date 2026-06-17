import streamlit as st
import pickle

# Load saved model
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Page Title
st.title("📰 Fake News Detector")
st.write("Enter a news article and check whether it is Real or Fake.")

# Input box
news = st.text_area("Enter News Article")

# Prediction button
if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        news_vector = vectorizer.transform([news])

        prediction = model.predict(news_vector)

        if prediction[0] == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")