import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.set_page_config(
    page_title="Budget Allocation Optimizer",
    layout="wide"
)

st.title("Marketing Budget Allocation Optimizer")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("datasets/campaigns.csv")

campaign_summary = (
    df.groupby("Campaign_Type")
    .agg({
        "Spend":       "sum",
        "Revenue":     "sum",
        "Conversions": "sum"
    })
    .reset_index()
)

campaign_summary["ROAS"] = (
    campaign_summary["Revenue"] /
    campaign_summary["Spend"]
)

campaign_summary["CPA"] = (
    campaign_summary["Spend"] /
    campaign_summary["Conversions"]
)

st.subheader("Current Campaign Performance")

st.dataframe(
    campaign_summary,
    use_container_width=True
)

# ==========================================
# SIDEBAR CONTROLS
# ==========================================

st.sidebar.header("Optimization Settings")

total_budget = st.sidebar.number_input(
    "Total Budget ($)",
    min_value=10_000,
    value=1_000_000,
    step=50_000
)

min_budget = st.sidebar.number_input(
    "Minimum Budget Per Campaign ($)",
    min_value=0,
    value=50_000,
    step=10_000
)

max_budget = st.sidebar.number_input(
    "Maximum Budget Per Campaign ($)",
    min_value=100_000,
    value=700_000,
    step=50_000
)

diminishing_factor = st.sidebar.slider(
    "Diminishing Returns Factor",
    min_value=0.1,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help=(
        "1.0 = linear returns (no diminishing). "
        "Lower values = stronger diminishing returns. "
        "Recommended: 0.5–0.7"
    )
)

# ==========================================
# DIMINISHING RETURNS MODEL
# Revenue(spend) = ROAS * spend ^ factor
# As spend doubles, revenue grows by 2^factor
# ==========================================

roas_values = campaign_summary["ROAS"].values
n_campaigns = len(roas_values)

def expected_revenue(budgets):
    """
    Revenue with diminishing returns:
    rev = ROAS_i * (spend_i ^ factor)
    Normalized so that at current spend the
    model matches actual ROAS.
    """
    current_spend = campaign_summary["Spend"].values
    # Scale factor so revenue at current spend == actual revenue
    scale = current_spend ** (1 - diminishing_factor)
    return np.sum(roas_values * scale * (budgets ** diminishing_factor))

def neg_revenue(budgets):
    return -expected_revenue(budgets)

# ==========================================
# OPTIMIZATION
# ==========================================

x0 = np.full(n_campaigns, total_budget / n_campaigns)

constraints = [
    {
        "type": "eq",
        "fun": lambda x: np.sum(x) - total_budget
    }
]

bounds = [
    (min_budget, max_budget)
    for _ in range(n_campaigns)
]

result = minimize(
    neg_revenue,
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 1000, "ftol": 1e-9}
)

# ==========================================
# RESULTS
# ==========================================

if result.success:

    campaign_summary["Recommended_Budget"] = result.x

    campaign_summary["Budget_Change_$"] = (
        campaign_summary["Recommended_Budget"] -
        campaign_summary["Spend"]
    )

    campaign_summary["Budget_Change_%"] = (
        campaign_summary["Budget_Change_$"] /
        campaign_summary["Spend"] * 100
    ).round(1)

    current_spend = campaign_summary["Spend"].values
    scale = current_spend ** (1 - diminishing_factor)

    campaign_summary["Expected_Revenue"] = (
        roas_values *
        scale *
        (result.x ** diminishing_factor)
    ).round(0)

    campaign_summary["Expected_ROAS"] = (
        campaign_summary["Expected_Revenue"] /
        campaign_summary["Recommended_Budget"]
    ).round(2)

    st.subheader("Optimized Budget Allocation")

    display_cols = [
        "Campaign_Type",
        "ROAS",
        "CPA",
        "Spend",
        "Recommended_Budget",
        "Budget_Change_$",
        "Budget_Change_%",
        "Expected_Revenue",
        "Expected_ROAS"
    ]

    st.dataframe(
        campaign_summary[display_cols],
        use_container_width=True
    )

    # ==========================================
    # BUDGET CHART
    # ==========================================

    st.subheader("Current vs Recommended Budget")

    chart_data = campaign_summary.set_index("Campaign_Type")[
        ["Spend", "Recommended_Budget"]
    ]

    st.bar_chart(chart_data)

    # ==========================================
    # REVENUE PROJECTION
    # ==========================================

    st.subheader("Revenue Projection")

    current_revenue  = campaign_summary["Revenue"].sum()
    expected_revenue_total = campaign_summary["Expected_Revenue"].sum()
    revenue_uplift   = expected_revenue_total - current_revenue
    uplift_pct       = (revenue_uplift / current_revenue) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Revenue",
        f"${current_revenue:,.0f}"
    )

    col2.metric(
        "Projected Revenue",
        f"${expected_revenue_total:,.0f}",
        delta=f"+${revenue_uplift:,.0f}"
    )

    col3.metric(
        "Revenue Uplift",
        f"{uplift_pct:.1f}%"
    )

    # ==========================================
    # INSIGHTS
    # ==========================================

    st.subheader("Optimization Insights")

    best = campaign_summary.sort_values(
        "Expected_ROAS", ascending=False
    ).iloc[0]

    worst = campaign_summary.sort_values(
        "Expected_ROAS", ascending=True
    ).iloc[0]

    st.success(
        f"Highest efficiency: **{best['Campaign_Type']}** "
        f"(Expected ROAS: {best['Expected_ROAS']:.2f}x)"
    )

    st.warning(
        f"Lowest efficiency: **{worst['Campaign_Type']}** "
        f"(Expected ROAS: {worst['Expected_ROAS']:.2f}x)"
    )

    st.info(
        f"Diminishing returns factor: **{diminishing_factor}** — "
        f"doubling spend yields **{2**diminishing_factor:.2f}x** revenue growth "
        f"instead of 2.0x (linear)."
    )

else:
    st.error(f"Optimization failed: {result.message}")
    st.write("Try increasing the maximum budget per campaign or reducing the minimum.")

