"""Main Streamlit application for the AI Fake News Detector."""

from pathlib import Path
import re

import nltk
import pandas as pd
import pickle
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from utils.database import create_table, get_predictions, save_prediction


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"
LOGO_PATH = BASE_DIR / "assets" / "fake-news-logo.svg"


st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --app-bg: #0b1120;
        --panel: #111a2e;
        --panel-soft: #172033;
        --border: #263552;
        --text: #e8eefc;
        --muted: #9aa9c2;
        --accent: #6d7cff;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 8%, rgba(79, 70, 229, .16), transparent 28rem),
            radial-gradient(circle at 15% 85%, rgba(8, 145, 178, .10), transparent 30rem),
            var(--app-bg);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10182a 0%, #0d1424 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] span {
        color: var(--text) !important;
    }

    [data-testid="stSidebarNav"] a {
        color: var(--text) !important;
        border-radius: 9px;
    }

    [data-testid="stSidebarNav"] a:hover {
        background: #202b48;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, .75);
    }

    h1, h2, h3, p, label {
        color: var(--text);
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    [data-testid="stMetric"],
    [data-testid="stForm"],
    [data-testid="stDataFrame"] {
        background: rgba(17, 26, 46, .82);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 14px 34px rgba(0, 0, 0, .16);
    }

    .stTextInput input, .stTextArea textarea {
        color: var(--text);
        background: var(--panel-soft);
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
    }

    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #7787a3;
        opacity: 1;
    }

    .stButton > button, .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        color: white;
        background: linear-gradient(90deg, var(--accent), #4f46e5);
        border: 0;
        border-radius: 10px;
        font-weight: 700;
        transition: transform .15s ease, box-shadow .15s ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(79, 70, 229, .32);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: .5rem;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--muted);
        background: var(--panel);
        border-radius: 10px 10px 0 0;
        padding: .65rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        color: white !important;
        background: #202b48 !important;
    }

    hr {
        border-color: var(--border) !important;
    }

    .app-subtitle {
        color: var(--muted);
        font-size: 1.05rem;
        margin-top: -.6rem;
        margin-bottom: 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_files():
    """Load and cache the trained classifier and TF-IDF vectorizer."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model files were not found. Run `python train.py` first."
        )

    with MODEL_PATH.open("rb") as model_file:
        trained_model = pickle.load(model_file)

    with VECTORIZER_PATH.open("rb") as vectorizer_file:
        trained_vectorizer = pickle.load(vectorizer_file)

    return trained_model, trained_vectorizer


@st.cache_resource
def load_stop_words():
    """Load the same English stop-word list used while training."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
        except Exception:
            # This keeps the app usable when the NLTK server is unavailable.
            return set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """Apply the preprocessing used in train.py."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [word for word in words if word not in load_stop_words()]
    return " ".join(words)


def predict_news(news_text: str):
    """Return the display label, confidence, and class probabilities."""
    cleaned_news = clean_text(news_text)
    if not cleaned_news:
        raise ValueError("Please enter meaningful English news text.")

    features = vectorizer.transform([cleaned_news])
    predicted_class = int(model.predict(features)[0])
    label = "REAL NEWS" if predicted_class == 1 else "FAKE NEWS"

    probabilities = {}
    confidence = None

    if hasattr(model, "predict_proba"):
        raw_probabilities = model.predict_proba(features)[0]
        probabilities = {
            int(class_name): float(probability)
            for class_name, probability in zip(model.classes_, raw_probabilities)
        }
        confidence = probabilities.get(predicted_class)

    return label, confidence, probabilities


def history_dataframe() -> pd.DataFrame:
    """Return saved predictions as a DataFrame."""
    records = get_predictions()
    return pd.DataFrame(
        records,
        columns=["ID", "News", "Prediction", "Timestamp"],
    )


try:
    create_table()
    model, vectorizer = load_model_files()
except Exception as error:
    st.error(f"Application setup failed: {error}")
    st.info("Make sure the model files exist, then restart the app.")
    st.stop()


with st.sidebar:
    st.image(str(LOGO_PATH), width=112)
    st.title("Fake News Detector")
    st.caption("Machine-learning assisted news classification")
    st.divider()
    st.markdown(
        """
        **How to use**

        1. Paste the complete news article.
        2. Select **Analyze News**.
        3. Review the prediction and confidence.

        **Model**

        - TF-IDF vectorization
        - Logistic Regression
        - Trained on Fake.csv and True.csv
        """
    )
    st.divider()
    st.warning(
        "This tool predicts writing patterns learned from its dataset. "
        "Always verify important claims with reliable sources."
    )


title_col, logo_col = st.columns([8, 1])
with title_col:
    st.title("AI Fake News Detector")
    st.markdown(
        '<p class="app-subtitle">Paste a news article to estimate whether its '
        "writing resembles real or fake news in the training dataset.</p>",
        unsafe_allow_html=True,
    )
with logo_col:
    st.image(str(LOGO_PATH), width=88)

predict_tab, history_tab = st.tabs(["🔍 Check News", "🕘 Recent History"])


with predict_tab:
    with st.form("news_prediction_form"):
        news_title = st.text_input(
            "News title (optional)",
            placeholder="Enter the article headline",
        )
        news_body = st.text_area(
            "News article",
            height=280,
            placeholder="Paste the full news article here...",
            help="Longer, complete articles generally give the model more context.",
        )
        submitted = st.form_submit_button(
            "Analyze News",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        combined_text = " ".join(
            part.strip() for part in (news_title, news_body) if part.strip()
        )

        if not combined_text:
            st.warning("Please enter a headline or news article.")
        elif len(combined_text.split()) < 10:
            st.warning(
                "Please enter at least 10 words so the model has enough context."
            )
        else:
            try:
                prediction, confidence, class_probabilities = predict_news(
                    combined_text
                )
                save_prediction(combined_text, prediction)

                st.divider()
                if prediction == "REAL NEWS":
                    st.success("✅ Prediction: REAL NEWS")
                else:
                    st.error("⚠️ Prediction: FAKE NEWS")

                if confidence is not None:
                    confidence_percent = confidence * 100
                    st.metric("Model confidence", f"{confidence_percent:.2f}%")
                    st.progress(float(confidence))

                    real_probability = class_probabilities.get(1, 0.0) * 100
                    fake_probability = class_probabilities.get(0, 0.0) * 100

                    col1, col2 = st.columns(2)
                    col1.metric("Real probability", f"{real_probability:.2f}%")
                    col2.metric("Fake probability", f"{fake_probability:.2f}%")

                st.caption(
                    "A high confidence score is not proof that a story is true "
                    "or false; it only reflects the model's classification."
                )
            except Exception as error:
                st.error(f"Could not analyze the article: {error}")


with history_tab:
    try:
        history = history_dataframe()

        if history.empty:
            st.info("No predictions have been saved yet.")
        else:
            real_count = int((history["Prediction"] == "REAL NEWS").sum())
            fake_count = int((history["Prediction"] == "FAKE NEWS").sum())

            col1, col2, col3 = st.columns(3)
            col1.metric("Total checks", len(history))
            col2.metric("Real predictions", real_count)
            col3.metric("Fake predictions", fake_count)

            display_history = history.copy()
            display_history["News"] = display_history["News"].str.slice(0, 160)
            st.dataframe(
                display_history,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download history as CSV",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
            )
    except Exception as error:
        st.error(f"Could not load prediction history: {error}")


st.divider()
st.caption("Built with Python, Streamlit, scikit-learn, NLTK, and SQLite.")
