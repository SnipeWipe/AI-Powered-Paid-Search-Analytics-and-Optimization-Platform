import streamlit as st
import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv(
    "datasets/campaigns.csv"
)

group_a = df.sample(500)["Conversions"]

group_b = df.sample(500)["Conversions"]

t,p = ttest_ind(
    group_a,
    group_b
)

st.title("Experimentation Framework")

st.write(
    f"P-value: {p}"
)

if p < 0.05:
    st.success(
        "Significant Difference"
    )
else:
    st.warning(
        "No Significant Difference"
    )