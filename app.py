import streamlit as st
import pickle
import pandas as pd
from utils.database import create_table, save_prediction, get_predictions

# ----------------------------

# PAGE CONFIG

# ----------------------------

st.set_page_config(
page_title="AI Fake News Detector",
page_icon="📰",
layout="wide",
initial_sidebar_state="expanded"
)

# ----------------------------

# DATABASE

# ----------------------------

create_table()

# ----------------------------

# LOAD MODEL

# ----------------------------

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# ----------------------------

# SIDEBAR

# ----------------------------

st.sidebar.title("📰 Navigation")

st.sidebar.info("""
Built by Pranav Koushik

### Tech Stack

* Python
* NLP
* Scikit-Learn
* Streamlit
* SQLite
  """)

# ----------------------------

# TITLE

# ----------------------------

st.title("📰 AI Fake News Detector")

st.markdown("""
Detect whether a news article is **REAL** or **FAKE**
using Machine Learning and Natural Language Processing.
""")

# ----------------------------

# INPUT

# ----------------------------

news = st.text_area(
"Enter News Article",
height=250,
placeholder="Paste any news article here..."
)

# ----------------------------

# PREDICTION

# ----------------------------

if st.button("🔍 Predict"):

```
if news.strip() == "":
    st.warning("⚠ Please enter some news text.")

else:

    with st.spinner("🔎 Analyzing article..."):

        vectorized_text = vectorizer.transform([news])

        prediction = model.predict(vectorized_text)

        probability = model.predict_proba(vectorized_text)

        confidence = max(probability[0]) * 100

    st.divider()

    st.subheader("🎯 Prediction Result")

    if prediction[0] == 1:

        result = "REAL NEWS"

        st.success("✅ REAL NEWS")

    else:

        result = "FAKE NEWS"

        st.error("❌ FAKE NEWS")

    # Save Prediction
    save_prediction(news, result)

    # Confidence Score

    st.subheader("📊 Confidence Score")

    st.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )

    st.progress(int(confidence))
```

# ----------------------------

# HISTORY

# ----------------------------

st.divider()

st.subheader("📜 Recent Predictions")

history = get_predictions()

if len(history) > 0:

```
for row in history[:10]:

    st.write(f"**Prediction:** {row[2]}")
    st.write(f"**Date:** {row[3]}")
    st.write(f"**News:** {row[1][:150]}...")
    st.write("---")
```

else:

```
st.info("No predictions available yet.")
```

# ----------------------------

# DOWNLOAD CSV

# ----------------------------

if len(history) > 0:

```
df = pd.DataFrame(
    history,
    columns=[
        "ID",
        "News",
        "Prediction",
        "Timestamp"
    ]
)

csv = df.to_csv(index=False)

st.download_button(
    label="📥 Download Prediction History",
    data=csv,
    file_name="prediction_history.csv",
    mime="text/csv"
)
```

# ----------------------------

# FOOTER

# ----------------------------

st.markdown("---")

st.markdown(
""" <center> <h4>Built with ❤️ by Pranav Koushik</h4>
AI Fake News Detection using NLP & Machine Learning </center>
""",
unsafe_allow_html=True
)
