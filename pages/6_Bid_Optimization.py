import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Bid Optimization Engine",
    layout="wide"
)

st.title("Bid Optimization Engine")

df = pd.read_csv("datasets/campaigns.csv")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================

st.sidebar.header("Optimization Settings")

cpa_percentile = st.sidebar.slider(
    "High CPA Threshold (percentile)",
    min_value=50,
    max_value=95,
    value=75,
    step=5,
    help="Campaigns above this CPA percentile are flagged for bid decrease"
)

roas_percentile = st.sidebar.slider(
    "Low ROAS Threshold (percentile)",
    min_value=5,
    max_value=50,
    value=25,
    step=5,
    help="Campaigns below this ROAS percentile are flagged for bid decrease"
)

target_cpa = st.sidebar.number_input(
    "Target CPA ($)",
    min_value=10,
    value=300,
    step=10
)

# ==========================================
# THRESHOLDS
# ==========================================

high_cpa_cutoff  = df["CPA"].quantile(cpa_percentile / 100)
low_roas_cutoff  = df["ROAS"].quantile(roas_percentile / 100)
high_roas_cutoff = df["ROAS"].quantile(0.75)
low_cpa_cutoff   = df["CPA"].quantile(0.25)

# ==========================================
# RECOMMENDATION LOGIC
# Decrease: high CPA AND low ROAS (both conditions met)
# Increase: low CPA AND high ROAS (both conditions met)
# Hold: everything else
# ==========================================

def recommend(row):
    high_cpa  = row["CPA"]  >= high_cpa_cutoff
    low_roas  = row["ROAS"] <= low_roas_cutoff
    low_cpa   = row["CPA"]  <= low_cpa_cutoff
    high_roas = row["ROAS"] >= high_roas_cutoff

    if high_cpa and low_roas:
        return "🔴 Decrease Bid"
    elif low_cpa and high_roas:
        return "🟢 Increase Bid"
    else:
        return "🟡 Hold"

df["Recommendation"] = df.apply(recommend, axis=1)

# ==========================================
# POTENTIAL SAVINGS / UPSIDE
# ==========================================

def potential_impact(row):
    if row["Recommendation"] == "🔴 Decrease Bid":
        # Estimated savings if CPA is brought down to target
        excess_cpa    = max(row["CPA"] - target_cpa, 0)
        est_savings   = excess_cpa * row["Conversions"]
        return round(est_savings, 2)
    elif row["Recommendation"] == "🟢 Increase Bid":
        # Estimated revenue upside if spend increases 20%
        est_upside = row["Revenue"] * 0.20
        return round(est_upside, 2)
    else:
        return 0.0

df["Potential_Impact_$"] = df.apply(potential_impact, axis=1)

# ==========================================
# SUMMARY METRICS
# ==========================================

total_decrease = (df["Recommendation"] == "🔴 Decrease Bid").sum()
total_increase = (df["Recommendation"] == "🟢 Increase Bid").sum()
total_hold     = (df["Recommendation"] == "🟡 Hold").sum()

total_savings  = df.loc[
    df["Recommendation"] == "🔴 Decrease Bid",
    "Potential_Impact_$"
].sum()

total_upside   = df.loc[
    df["Recommendation"] == "🟢 Increase Bid",
    "Potential_Impact_$"
].sum()

col1, col2, col3 = st.columns(3)
col1.metric("🔴 Decrease Bid",   f"{total_decrease} campaigns")
col2.metric("🟢 Increase Bid",   f"{total_increase} campaigns")
col3.metric("🟡 Hold",           f"{total_hold} campaigns")

col4, col5 = st.columns(2)
col4.metric("Estimated Savings (Decrease)",  f"${total_savings:,.0f}")
col5.metric("Estimated Revenue Upside (Increase)", f"${total_upside:,.0f}")

st.caption(
    f"Thresholds — High CPA: ${high_cpa_cutoff:,.0f} "
    f"({cpa_percentile}th pct) | "
    f"Low ROAS: {low_roas_cutoff:.2f}x "
    f"({roas_percentile}th pct) | "
    f"Target CPA: ${target_cpa}"
)

# ==========================================
# RESULTS TABLE
# ==========================================

st.subheader("Campaign Bid Recommendations")

display_cols = [
    "Campaign_ID",
    "Campaign_Type",
    "Device",
    "CPA",
    "ROAS",
    "Spend",
    "Revenue",
    "Conversions",
    "Recommendation",
    "Potential_Impact_$"
]

rec_filter = st.selectbox(
    "Filter by Recommendation",
    ["All", "🔴 Decrease Bid", "🟢 Increase Bid", "🟡 Hold"]
)

table_df = df if rec_filter == "All" else df[df["Recommendation"] == rec_filter]

st.dataframe(
    table_df[display_cols].sort_values(
        "Potential_Impact_$",
        ascending=False
    ),
    use_container_width=True,
    height=500
)

# ==========================================
# DISTRIBUTION CHART
# ==========================================

st.subheader("CPA Distribution by Recommendation")

dist_data = (
    df.groupby("Recommendation")["CPA"]
    .mean()
    .reset_index()
    .set_index("Recommendation")
)

st.bar_chart(dist_data)

