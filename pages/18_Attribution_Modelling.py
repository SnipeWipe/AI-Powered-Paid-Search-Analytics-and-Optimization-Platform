import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Multi-Touch Attribution",
    layout="wide"
)

st.title("Multi-Touch Attribution Modeling")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("datasets/customer_journeys.csv")

st.subheader("Customer Journeys (Sample)")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# ==========================================
# ATTRIBUTION ENGINE
# Returns channel_credit dict for any model
# ==========================================

def compute_attribution(dataframe, model_type):
    channel_credit = {}

    for _, row in dataframe.iterrows():
        journey = row["Journey"].split(" > ")
        revenue = row["Conversion_Value"]

        if model_type == "First Touch":
            channel = journey[0]
            channel_credit[channel] = channel_credit.get(channel, 0) + revenue

        elif model_type == "Last Touch":
            channel = journey[-1]
            channel_credit[channel] = channel_credit.get(channel, 0) + revenue

        elif model_type == "Linear":
            credit = revenue / len(journey)
            for channel in journey:
                channel_credit[channel] = channel_credit.get(channel, 0) + credit

        elif model_type == "Position Based":
            if len(journey) == 1:
                ch = journey[0]
                channel_credit[ch] = channel_credit.get(ch, 0) + revenue
            else:
                first_credit  = revenue * 0.4
                last_credit   = revenue * 0.4
                middle_credit = revenue * 0.2
                middle_n      = len(journey) - 2

                first_ch = journey[0]
                last_ch  = journey[-1]

                channel_credit[first_ch] = channel_credit.get(first_ch, 0) + first_credit
                channel_credit[last_ch]  = channel_credit.get(last_ch,  0) + last_credit

                if middle_n > 0:
                    per_middle = middle_credit / middle_n
                    for ch in journey[1:-1]:
                        channel_credit[ch] = channel_credit.get(ch, 0) + per_middle

    results = pd.DataFrame({
        "Channel":            list(channel_credit.keys()),
        "Attributed_Revenue": list(channel_credit.values())
    })

    results = results.sort_values("Attributed_Revenue", ascending=False)
    total   = results["Attributed_Revenue"].sum()
    results["Attribution_%"] = (results["Attributed_Revenue"] / total * 100).round(2)

    return results

# ==========================================
# VIEW MODE
# ==========================================

view_mode = st.radio(
    "View Mode",
    ["Single Model", "Compare All Models"],
    horizontal=True
)

all_models = ["First Touch", "Last Touch", "Linear", "Position Based"]

# ==========================================
# SINGLE MODEL VIEW
# ==========================================

if view_mode == "Single Model":

    model_type = st.selectbox(
        "Select Attribution Model",
        all_models
    )

    results = compute_attribution(df, model_type)

    st.subheader(f"{model_type} Attribution Results")

    st.dataframe(results, use_container_width=True)

    st.bar_chart(
        results.set_index("Channel")["Attributed_Revenue"]
    )

    best_channel = results.iloc[0]

    st.success(
        f"Top Channel: **{best_channel['Channel']}** — "
        f"{best_channel['Attribution_%']:.2f}% share"
    )

    st.metric(
        "Total Attributed Revenue",
        f"${results['Attributed_Revenue'].sum():,.0f}"
    )

# ==========================================
# COMPARE ALL MODELS SIDE BY SIDE
# ==========================================

else:

    st.subheader("All Models — Side-by-Side Comparison")

    # Build one combined dataframe
    combined = {}
    for model in all_models:
        res = compute_attribution(df, model)
        combined[model] = res.set_index("Channel")["Attribution_%"]

    compare_df = pd.DataFrame(combined).fillna(0)
    compare_df.columns = [m.replace(" ", "_") for m in all_models]

    st.dataframe(
        compare_df.style.format("{:.2f}%"),
        use_container_width=True
    )

    # ---- Grouped bar chart via Plotly ----

    channels = compare_df.index.tolist()
    colors   = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

    fig = go.Figure()

    for i, model in enumerate(all_models):
        col_name = model.replace(" ", "_")
        fig.add_trace(go.Bar(
            name=model,
            x=channels,
            y=compare_df[col_name],
            marker_color=colors[i]
        ))

    fig.update_layout(
        barmode="group",
        title="Attribution Share (%) by Model",
        xaxis_title="Channel",
        yaxis_title="Attribution %",
        legend_title="Model",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---- Revenue comparison ----

    st.subheader("Attributed Revenue by Model")

    rev_combined = {}
    for model in all_models:
        res = compute_attribution(df, model)
        rev_combined[model] = res.set_index("Channel")["Attributed_Revenue"]

    rev_df = pd.DataFrame(rev_combined).fillna(0)
    rev_df.columns = [m.replace(" ", "_") for m in all_models]

    fig2 = go.Figure()
    for i, model in enumerate(all_models):
        col_name = model.replace(" ", "_")
        fig2.add_trace(go.Bar(
            name=model,
            x=channels,
            y=rev_df[col_name],
            marker_color=colors[i]
        ))

    fig2.update_layout(
        barmode="group",
        title="Attributed Revenue ($) by Model",
        xaxis_title="Channel",
        yaxis_title="Revenue ($)",
        legend_title="Model",
        height=450
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ---- Insight: which channel shifts most across models ----

    st.subheader("Channel Sensitivity to Model Choice")

    compare_df["Range_%"] = (
        compare_df.max(axis=1) - compare_df.min(axis=1)
    ).round(2)

    most_sensitive = compare_df["Range_%"].idxmax()
    least_sensitive = compare_df["Range_%"].idxmin()

    st.warning(
        f"**{most_sensitive}** has the highest variance across models "
        f"({compare_df.loc[most_sensitive, 'Range_%']:.2f}% range). "
        "Its true contribution is most model-dependent."
    )

    st.success(
        f"**{least_sensitive}** is the most stable channel across all models "
        f"({compare_df.loc[least_sensitive, 'Range_%']:.2f}% range)."
    )

    st.dataframe(
        compare_df[["Range_%"]].sort_values("Range_%", ascending=False),
        use_container_width=True
    )

