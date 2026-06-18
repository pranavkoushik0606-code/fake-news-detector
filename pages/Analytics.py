import streamlit as st
from utils.database import get_predictions
import pandas as pd
import plotly.express as px

# Page Title
st.title("📊 Analytics Dashboard")

# Get Data
history = get_predictions()

if len(history) > 0:

    # Convert to DataFrame
    df = pd.DataFrame(
        history,
        columns=[
            "ID",
            "News",
            "Prediction",
            "Timestamp"
        ]
    )

    # Statistics
    total = len(df)

    real_count = len(
        df[df["Prediction"] == "REAL NEWS"]
    )

    fake_count = len(
        df[df["Prediction"] == "FAKE NEWS"]
    )

    # =========================
    # METRICS
    # =========================

    st.subheader("📈 Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Predictions",
            total
        )

    with col2:
        st.metric(
            "Real News",
            real_count
        )

    with col3:
        st.metric(
            "Fake News",
            fake_count
        )

    st.divider()

    # =========================
    # PIE CHART
    # =========================

    st.subheader("🥧 Prediction Distribution")

    chart_df = pd.DataFrame({
        "Category": ["Real News", "Fake News"],
        "Count": [real_count, fake_count]
    })

    fig = px.pie(
        chart_df,
        names="Category",
        values="Count",
        hole=0.5,
        title="Prediction Distribution"
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    fig.update_layout(
        title_x=0.3,
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # =========================
    # BAR CHART
    # =========================

    st.subheader("📊 Prediction Count")

    bar_fig = px.bar(
        chart_df,
        x="Category",
        y="Count",
        color="Category",
        text="Count",
        title="Real vs Fake News Count"
    )

    bar_fig.update_layout(
        height=500
    )

    st.plotly_chart(
        bar_fig,
        use_container_width=True
    )

    st.divider()

    # =========================
    # LATEST PREDICTION
    # =========================

    st.subheader("🕒 Latest Prediction")

    latest = df.iloc[0]

    st.info(
        f"""
Prediction: {latest['Prediction']}

Time: {latest['Timestamp']}
        """
    )

    st.divider()

    # =========================
    # SEARCH FEATURE
    # =========================

    st.subheader("🔍 Search Prediction Records")

    search = st.text_input(
        "Search News Content"
    )

    filtered_df = df.copy()

    if search:
        filtered_df = filtered_df[
            filtered_df["News"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    # =========================
    # TABLE
    # =========================

    st.subheader("📋 Prediction Records")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    # =========================
    # DOWNLOAD CSV
    # =========================

    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    st.divider()

    # =========================
    # RECENT ACTIVITY
    # =========================

    st.subheader("📜 Recent Activity")

    recent = df.head(5)

    for _, row in recent.iterrows():

        st.write(
            f"**{row['Prediction']}** | {row['Timestamp']}"
        )

        st.caption(
            row["News"][:100] + "..."
        )

else:

    st.warning(
        "No prediction data available yet."
    )