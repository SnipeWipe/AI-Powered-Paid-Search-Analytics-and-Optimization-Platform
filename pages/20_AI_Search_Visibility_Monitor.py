import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI Search Visibility Monitor",
    layout="wide"
)

st.title(
    "AI Search Visibility Monitor"
)

# ==========================================
# Load Data
# ==========================================

df = pd.read_csv(
    "datasets/ai_visibility.csv"
)

# ==========================================
# Filters
# ==========================================

platform_filter = st.sidebar.multiselect(
    "Platform",
    options=df["Platform"].unique(),
    default=df["Platform"].unique()
)

brand_filter = st.sidebar.multiselect(
    "Brand",
    options=df["Brand_Mentioned"].unique(),
    default=df["Brand_Mentioned"].unique()
)

filtered_df = df[
    (
        df["Platform"]
        .isin(platform_filter)
    )
    &
    (
        df["Brand_Mentioned"]
        .isin(brand_filter)
    )
]

# ==========================================
# Visibility Score
# ==========================================

filtered_df = filtered_df.copy()

filtered_df["Visibility_Score"] = (
    1 / filtered_df["Rank"]
)

visibility_summary = (
    filtered_df
    .groupby(
        "Brand_Mentioned"
    )
    ["Visibility_Score"]
    .mean()
    .reset_index()
)

visibility_summary = (
    visibility_summary
    .sort_values(
        "Visibility_Score",
        ascending=False
    )
)

# ==========================================
# KPI Section
# ==========================================

st.subheader(
    "Visibility Leaderboard"
)

leader = (
    visibility_summary
    .iloc[0]
)

st.metric(
    "Top Visible Brand",
    leader[
        "Brand_Mentioned"
    ]
)

st.metric(
    "Visibility Score",
    round(
        leader[
            "Visibility_Score"
        ],
        3
    )
)

# ==========================================
# Leaderboard
# ==========================================

st.dataframe(
    visibility_summary,
    use_container_width=True
)

# ==========================================
# Bar Chart
# ==========================================

fig = px.bar(

    visibility_summary,

    x="Brand_Mentioned",

    y="Visibility_Score",

    title=
    "AI Search Visibility Score"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# Platform Breakdown
# ==========================================

platform_view = (

    filtered_df

    .groupby(
        [
            "Platform",
            "Brand_Mentioned"
        ]
    )

    ["Visibility_Score"]

    .mean()

    .reset_index()
)

fig2 = px.bar(

    platform_view,

    x="Platform",

    y="Visibility_Score",

    color="Brand_Mentioned",

    barmode="group",

    title=
    "Visibility by Platform"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ==========================================
# Prompt Analysis
# ==========================================

st.subheader(
    "Prompt Intelligence"
)

prompt_view = (

    filtered_df

    .groupby(
        "Prompt"
    )

    ["Visibility_Score"]

    .mean()

    .reset_index()

    .sort_values(
        "Visibility_Score",
        ascending=False
    )
)

st.dataframe(
    prompt_view,
    use_container_width=True
)

# ==========================================
# Brand Market Share
# ==========================================

market_share = (

    filtered_df

    .groupby(
        "Brand_Mentioned"
    )

    .size()

    .reset_index(
        name="Mentions"
    )
)

fig3 = px.pie(

    market_share,

    names=
    "Brand_Mentioned",

    values=
    "Mentions",

    title=
    "AI Search Share of Voice"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ==========================================
# Business Insights
# ==========================================

st.subheader(
    "AI Search Insights"
)

best_prompt = (
    prompt_view
    .iloc[0]
)

worst_prompt = (
    prompt_view
    .iloc[-1]
)

st.success(
    f"""
    Highest visibility prompt:

    {best_prompt['Prompt']}

    Score:
    {best_prompt['Visibility_Score']:.3f}
    """
)

st.warning(
    f"""
    Lowest visibility prompt:

    {worst_prompt['Prompt']}

    Score:
    {worst_prompt['Visibility_Score']:.3f}
    """
)

# Visibility Trend Simulation

trend_df = filtered_df.copy()

trend_df["Month"] = pd.date_range(
    start="2025-01-01",
    periods=len(trend_df),
    freq="D"
)

trend_df = (
    trend_df
    .groupby(
        pd.Grouper(
            key="Month",
            freq="ME"
        )
    )
    ["Visibility_Score"]
    .mean()
    .reset_index()
)

fig4 = px.line(

    trend_df,

    x="Month",

    y="Visibility_Score",

    title=
    "Visibility Trend"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)