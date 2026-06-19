import streamlit as st
import pandas as pd
import numpy as np

from scipy.stats import beta
import plotly.graph_objects as go

st.set_page_config(
    page_title="Bayesian A/B Testing",
    layout="wide"
)

st.title(
    "Bayesian A/B Testing Framework"
)

# ==========================================
# Load Data
# ==========================================

df = pd.read_csv(
    "datasets/ab_test_data.csv"
)

# ==========================================
# Split Variants
# ==========================================

variant_a = df[
    df["Variant"] == "A"
]

variant_b = df[
    df["Variant"] == "B"
]

# ==========================================
# Basic Metrics
# ==========================================

visitors_a = len(
    variant_a
)

visitors_b = len(
    variant_b
)

conversions_a = (
    variant_a["Converted"]
    .sum()
)

conversions_b = (
    variant_b["Converted"]
    .sum()
)

cr_a = (
    conversions_a
    /
    visitors_a
)

cr_b = (
    conversions_b
    /
    visitors_b
)

# ==========================================
# Bayesian Posterior
# ==========================================

alpha_a = (
    conversions_a + 1
)

beta_a = (
    visitors_a
    -
    conversions_a
    + 1
)

alpha_b = (
    conversions_b + 1
)

beta_b = (
    visitors_b
    -
    conversions_b
    + 1
)

# ==========================================
# Monte Carlo Simulation
# ==========================================

samples = 100000

posterior_a = beta.rvs(
    alpha_a,
    beta_a,
    size=samples
)

posterior_b = beta.rvs(
    alpha_b,
    beta_b,
    size=samples
)

prob_b_better = np.mean(
    posterior_b >
    posterior_a
)

expected_lift = (
    (
        posterior_b.mean()
        -
        posterior_a.mean()
    )
    /
    posterior_a.mean()
) * 100

# ==========================================
# Metrics
# ==========================================

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Variant A CR",
    f"{cr_a:.2%}"
)

col2.metric(
    "Variant B CR",
    f"{cr_b:.2%}"
)

col3.metric(
    "P(B > A)",
    f"{prob_b_better:.2%}"
)

col4.metric(
    "Expected Lift",
    f"{expected_lift:.2f}%"
)

# ==========================================
# Results Interpretation
# ==========================================

st.subheader(
    "Bayesian Decision"
)

if prob_b_better > 0.95:

    st.success(
        f"""
        Variant B has a
        {prob_b_better:.2%}
        probability of outperforming
        Variant A.

        Recommendation:
        Roll out Variant B.
        """
    )

elif prob_b_better > 0.80:

    st.warning(
        f"""
        Variant B shows promise
        but more data is recommended.

        Current probability:
        {prob_b_better:.2%}
        """
    )

else:

    st.info(
        f"""
        No strong evidence
        that Variant B is superior.

        Probability:
        {prob_b_better:.2%}
        """
    )

# ==========================================
# Posterior Distribution Plot
# ==========================================

st.subheader(
    "Posterior Conversion Rate Distributions"
)

x = np.linspace(
    0,
    0.2,
    1000
)

y_a = beta.pdf(
    x,
    alpha_a,
    beta_a
)

y_b = beta.pdf(
    x,
    alpha_b,
    beta_b
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=x,
        y=y_a,
        mode="lines",
        name="Variant A"
    )
)

fig.add_trace(
    go.Scatter(
        x=x,
        y=y_b,
        mode="lines",
        name="Variant B"
    )
)

fig.update_layout(
    title="Bayesian Posterior Distribution",
    xaxis_title="Conversion Rate",
    yaxis_title="Density"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# Summary Table
# ==========================================

summary = pd.DataFrame({

    "Metric":[
        "Visitors",
        "Conversions",
        "Conversion Rate"
    ],

    "Variant A":[
        visitors_a,
        conversions_a,
        round(
            cr_a,
            4
        )
    ],

    "Variant B":[
        visitors_b,
        conversions_b,
        round(
            cr_b,
            4
        )
    ]
})

st.subheader(
    "Experiment Summary"
)

st.dataframe(
    summary,
    use_container_width=True
)

# ==========================================
# Business Impact
# ==========================================

monthly_visitors = st.number_input(
    "Monthly Visitors",
    value=100000
)

extra_conversions = (

    (cr_b - cr_a)

    *

    monthly_visitors

)

st.subheader(
    "Estimated Business Impact"
)

st.metric(
    "Additional Monthly Conversions",
    int(extra_conversions)
)