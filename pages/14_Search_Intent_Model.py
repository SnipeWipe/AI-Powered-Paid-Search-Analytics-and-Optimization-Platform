import streamlit as st
import pandas as pd
from transformers import pipeline

st.title("Search Intent Classification")

# -----------------------------
# Load Model Once
# -----------------------------
@st.cache_resource
def load_model():

    return pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli"
    )

classifier = load_model()

# -----------------------------
# Intent Labels
# -----------------------------
labels = [
    "Informational",
    "Commercial",
    "Transactional",
    "Navigational"
]

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(
    "datasets/search_queries.csv"
)

# -----------------------------
# Create Metrics
# -----------------------------
df["Conversion_Rate"] = (
    df["Conversions"] /
    df["Clicks"].replace(0, 1)
) * 100

# -----------------------------
# Top Queries Only
# -----------------------------
df = (
    df.sort_values(
        "Conversions",
        ascending=False
    )
    .head(300)
)

# -----------------------------
# Batch Classification
# -----------------------------
@st.cache_data
def classify_queries(queries):

    results = classifier(
        queries,
        candidate_labels=labels
    )

    return [
        result["labels"][0]
        for result in results
    ]

# -----------------------------
# Predict Intent
# -----------------------------
with st.spinner(
    "Classifying Search Queries..."
):

    df["Intent"] = classify_queries(
        df["Query"].tolist()
    )

# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Queries",
        len(df)
    )

with col2:
    st.metric(
        "Informational",
        (df["Intent"] == "Informational").sum()
    )

with col3:
    st.metric(
        "Commercial",
        (df["Intent"] == "Commercial").sum()
    )

with col4:
    st.metric(
        "Transactional",
        (df["Intent"] == "Transactional").sum()
    )

with col5:
    st.metric(
        "Revenue",
        f"${df['Revenue'].sum():,.0f}"
    )
    
# -----------------------------
# Intent Distribution
# -----------------------------
st.subheader(
    "Intent Distribution"
)

intent_summary = (
    df["Intent"]
    .value_counts()
    .reset_index()
)

intent_summary.columns = [
    "Intent",
    "Count"
]

st.bar_chart(
    intent_summary.set_index(
        "Intent"
    )
)

# -----------------------------
# Intent Performance Summary
# -----------------------------
st.subheader(
    "Intent Performance Summary"
)

intent_perf = (
    df.groupby("Intent")
    .agg({
        "Clicks":"sum",
        "Conversions":"sum",
        "Revenue":"sum",
        "Conversion_Rate":"mean"
    })
    .round(2)
)

st.dataframe(
    intent_perf,
    use_container_width=True
)

# -----------------------------
# Intent Predictions
# -----------------------------
st.subheader(
    "Intent Predictions"
)

st.dataframe(
    df[
        [
            "Query",
            "Intent",
            "Clicks",
            "Conversions",
            "Revenue",
            "Conversion_Rate",
            "Avg_Position",
            "Quality_Score"
        ]
    ],
    use_container_width=True,
    height=600
)