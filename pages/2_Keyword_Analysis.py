import streamlit as st
import pandas as pd

df = pd.read_csv(
    "datasets/keywords.csv"
)

df["CTR"] = (
    df["Clicks"]
    /
    df["Impressions"]
)

st.title(
    "Keyword Analysis"
)

st.dataframe(df.head())

st.bar_chart(
    df.sort_values(
        "CTR",
        ascending=False
    )
    .head(20)
    .set_index("Keyword")["CTR"]
)