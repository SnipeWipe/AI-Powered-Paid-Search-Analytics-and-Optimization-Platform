import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Customer Quality Prediction",
    layout="wide"
)

st.title("Customer Quality Prediction")

# ==========================================
# LOAD + TRAIN
# ==========================================

df = pd.read_csv("datasets/campaigns.csv")

FEATURES = [
    "Impressions", "Clicks", "Avg_CPC", "Spend",
    "CTR", "CVR", "Conversions", "Revenue", "CPA", "ROAS"
]

X = df[FEATURES]
y = df["Customer_Quality_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

@st.cache_resource
def train_model():
    m = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    m.fit(X_train, y_train)
    return m

model = train_model()

predictions = model.predict(X_test)
score       = r2_score(y_test, predictions)

# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Model R²", round(score, 3))

col2.metric(
    "Score Range",
    f"{y.min():.0f} – {y.max():.0f}"
)

col3.metric(
    "Mean Quality Score",
    round(y.mean(), 1)
)

if score >= 0.90:
    st.success("🟢 Excellent Model Performance")
elif score >= 0.80:
    st.warning("🟡 Good Model Performance")
else:
    st.error("🔴 Model Needs Improvement")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

st.subheader("Feature Importance")

importance = (
    pd.DataFrame({
        "Feature":    FEATURES,
        "Importance": model.feature_importances_
    })
    .sort_values("Importance", ascending=False)
)

st.dataframe(importance, use_container_width=True)
st.bar_chart(importance.set_index("Feature")["Importance"])

# ==========================================
# PREDICTION FORM
# ==========================================

st.subheader("Predict Customer Quality Score")

col_a, col_b = st.columns(2)

with col_a:
    impressions = st.number_input("Impressions",  value=50_000, step=1_000)
    clicks      = st.number_input("Clicks",       value=3_000,  step=100)
    avg_cpc     = st.number_input("Avg CPC ($)",  value=20.0,   step=0.5)

with col_b:
    cvr = st.slider(
        "Conversion Rate (CVR)",
        min_value=0.01,
        max_value=0.30,
        value=0.08,
        step=0.01,
        format="%.2f"
    )

# ---- Auto-computed derived fields ----

spend       = clicks * avg_cpc
ctr         = clicks / impressions if impressions > 0 else 0
conversions = clicks * cvr
revenue     = conversions * 3_000
cpa         = spend / max(conversions, 1)
roas        = revenue / spend if spend > 0 else 0

# ==========================================
# INPUT SUMMARY CARD
# Shows the user exactly what will be sent
# to the model before they click Predict
# ==========================================

st.subheader("Input Summary (sent to model)")

summary_cols = st.columns(5)

summary_cols[0].metric("Impressions",  f"{impressions:,.0f}")
summary_cols[1].metric("Clicks",       f"{clicks:,.0f}")
summary_cols[2].metric("Avg CPC",      f"${avg_cpc:.2f}")
summary_cols[3].metric("Spend",        f"${spend:,.0f}")
summary_cols[4].metric("CTR",          f"{ctr*100:.2f}%")

summary_cols2 = st.columns(5)

summary_cols2[0].metric("CVR",          f"{cvr*100:.2f}%")
summary_cols2[1].metric("Conversions",  f"{conversions:,.0f}")
summary_cols2[2].metric("Revenue",      f"${revenue:,.0f}")
summary_cols2[3].metric("CPA",          f"${cpa:,.2f}")
summary_cols2[4].metric("ROAS",         f"{roas:.2f}x")

# ==========================================
# PREDICT
# ==========================================

if st.button("Predict Customer Quality Score", type="primary"):

    sample = pd.DataFrame({
        "Impressions":  [impressions],
        "Clicks":       [clicks],
        "Avg_CPC":      [avg_cpc],
        "Spend":        [spend],
        "CTR":          [ctr],
        "CVR":          [cvr],
        "Conversions":  [conversions],
        "Revenue":      [revenue],
        "CPA":          [cpa],
        "ROAS":         [roas]
    })

    predicted_score = model.predict(sample)[0]

    st.divider()

    col_score, col_band = st.columns(2)

    col_score.metric(
        "Predicted Customer Quality Score",
        f"{predicted_score:.2f} / 100"
    )

    if predicted_score >= 75:
        col_band.success("🟢 High Quality Customer Segment")
    elif predicted_score >= 50:
        col_band.warning("🟡 Medium Quality Customer Segment")
    else:
        col_band.error("🔴 Low Quality Customer Segment")

    # Show where this score sits vs the distribution
    pct_rank = (y < predicted_score).mean() * 100

    st.caption(
        f"This score is higher than **{pct_rank:.1f}%** "
        "of campaigns in the training dataset."
    )

