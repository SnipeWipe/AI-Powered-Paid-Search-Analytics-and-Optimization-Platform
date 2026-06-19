import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

st.title(
    "Customer Quality Prediction"
)

df = pd.read_csv(
    "datasets/campaigns.csv"
)

features = [
    "Impressions",
    "Clicks",
    "Avg_CPC",
    "Spend",
    "CTR",
    "CVR",
    "Conversions",
    "Revenue",
    "CPA",
    "ROAS"
]

X = df[features]

y = df[
    "Customer_Quality_Score"
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

score = r2_score(
    y_test,
    predictions
)

st.metric(
    "Model R²",
    round(score,3)
)

importance = pd.DataFrame({
    "Feature": features,
    "Importance":
    model.feature_importances_
})

importance = (
    importance
    .sort_values(
        "Importance",
        ascending=False
    )
)

st.subheader(
    "Feature Importance"
)

st.dataframe(
    importance,
    use_container_width=True
)

st.subheader(
    "Predict Customer Quality"
)

impressions = st.number_input(
    "Impressions",
    value=50000
)

clicks = st.number_input(
    "Clicks",
    value=3000
)

avg_cpc = st.number_input(
    "Avg CPC",
    value=20.0
)

spend = clicks * avg_cpc

ctr = clicks / impressions

cvr = st.slider(
    "CVR",
    0.01,
    0.30,
    0.08
)

conversions = clicks * cvr

revenue = conversions * 3000

cpa = spend / max(
    conversions,
    1
)

roas = revenue / spend

if st.button(
    "Predict"
):

    sample = pd.DataFrame({
        "Impressions":[impressions],
        "Clicks":[clicks],
        "Avg_CPC":[avg_cpc],
        "Spend":[spend],
        "CTR":[ctr],
        "CVR":[cvr],
        "Conversions":[conversions],
        "Revenue":[revenue],
        "CPA":[cpa],
        "ROAS":[roas]
    })

    score = model.predict(
        sample
    )[0]

    st.success(
        f"Predicted Customer Quality Score: {score:.2f}"
    )