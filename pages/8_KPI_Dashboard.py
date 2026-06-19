import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Executive Marketing Dashboard",
    layout="wide"
)

st.title("Executive Marketing Dashboard")

df = pd.read_csv("datasets/campaigns.csv")

# ==========================================
# FILTERS
# ==========================================

col_f1, col_f2 = st.columns(2)

with col_f1:
    campaign_filter = st.multiselect(
        "Campaign Type",
        options=df["Campaign_Type"].unique(),
        default=df["Campaign_Type"].unique()
    )

with col_f2:
    device_filter = st.multiselect(
        "Device",
        options=df["Device"].unique(),
        default=df["Device"].unique()
    )

filtered = df[
    df["Campaign_Type"].isin(campaign_filter) &
    df["Device"].isin(device_filter)
]

# ==========================================
# DELTA CALCULATION
# Simulate a "previous period" by splitting
# the dataset in half by Campaign_ID
# ==========================================

mid = df["Campaign_ID"].median()

current = filtered[filtered["Campaign_ID"] > mid]
previous = filtered[filtered["Campaign_ID"] <= mid]

def safe_delta(curr_val, prev_val):
    if prev_val == 0:
        return 0.0
    return round(((curr_val - prev_val) / prev_val) * 100, 2)

curr_ctr  = current["CTR"].mean() * 100
prev_ctr  = previous["CTR"].mean() * 100

curr_cvr  = current["CVR"].mean() * 100
prev_cvr  = previous["CVR"].mean() * 100

curr_cpa  = current["CPA"].mean()
prev_cpa  = previous["CPA"].mean()

curr_roas = current["ROAS"].mean()
prev_roas = previous["ROAS"].mean()

curr_spend   = current["Spend"].sum()
prev_spend   = previous["Spend"].sum()

curr_revenue = current["Revenue"].sum()
prev_revenue = previous["Revenue"].sum()

# ==========================================
# KPI METRICS WITH DELTAS
# ==========================================

st.subheader("Performance vs Previous Period")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "CTR",
    f"{curr_ctr:.2f}%",
    delta=f"{safe_delta(curr_ctr, prev_ctr)}%"
)

col2.metric(
    "CVR",
    f"{curr_cvr:.2f}%",
    delta=f"{safe_delta(curr_cvr, prev_cvr)}%"
)

col3.metric(
    "CPA",
    f"${curr_cpa:,.2f}",
    # Lower CPA is better — invert delta color
    delta=f"{safe_delta(curr_cpa, prev_cpa)}%",
    delta_color="inverse"
)

col4.metric(
    "ROAS",
    f"{curr_roas:.2f}x",
    delta=f"{safe_delta(curr_roas, prev_roas)}%"
)

col5, col6 = st.columns(2)

col5.metric(
    "Total Spend",
    f"${curr_spend:,.0f}",
    delta=f"{safe_delta(curr_spend, prev_spend)}%",
    delta_color="inverse"
)

col6.metric(
    "Total Revenue",
    f"${curr_revenue:,.0f}",
    delta=f"{safe_delta(curr_revenue, prev_revenue)}%"
)

# ==========================================
# CAMPAIGN TYPE BREAKDOWN
# ==========================================

st.subheader("Performance by Campaign Type")

breakdown = (
    filtered
    .groupby("Campaign_Type")
    .agg(
        Spend=("Spend", "sum"),
        Revenue=("Revenue", "sum"),
        Conversions=("Conversions", "sum"),
        CTR=("CTR", "mean"),
        CVR=("CVR", "mean"),
        CPA=("CPA", "mean"),
        ROAS=("ROAS", "mean")
    )
    .reset_index()
)

breakdown["CTR"]  = (breakdown["CTR"] * 100).round(2)
breakdown["CVR"]  = (breakdown["CVR"] * 100).round(2)
breakdown["CPA"]  = breakdown["CPA"].round(2)
breakdown["ROAS"] = breakdown["ROAS"].round(2)

st.dataframe(
    breakdown,
    use_container_width=True
)

# ==========================================
# DEVICE BREAKDOWN
# ==========================================

st.subheader("Performance by Device")

device_breakdown = (
    filtered
    .groupby("Device")
    .agg(
        Spend=("Spend", "sum"),
        Revenue=("Revenue", "sum"),
        ROAS=("ROAS", "mean"),
        CPA=("CPA", "mean")
    )
    .reset_index()
)

device_breakdown["ROAS"] = device_breakdown["ROAS"].round(2)
device_breakdown["CPA"]  = device_breakdown["CPA"].round(2)

st.dataframe(
    device_breakdown,
    use_container_width=True
)

st.bar_chart(
    device_breakdown.set_index("Device")["ROAS"]
)

