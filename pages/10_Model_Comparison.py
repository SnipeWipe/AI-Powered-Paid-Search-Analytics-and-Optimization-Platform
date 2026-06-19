import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    PowerTransformer,
    PolynomialFeatures
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    RandomizedSearchCV
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    explained_variance_score
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap

st.set_page_config(page_title="ML Pipeline — Conversion Prediction", layout="wide")
st.title("Conversion Prediction — ML Pipeline")

st.markdown("""
**Why low R² before?** Conversions = Clicks × CVR in the raw data, but CVR is random noise —  
so raw features like Clicks and Spend can't fully explain Conversions without CVR itself.  
This pipeline adds **engineered features**, a **preprocessing layer**, and **trains multiple models** 
to find the best learner.
""")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("datasets/campaigns.csv")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Pipeline Settings")

st.sidebar.subheader("1. Feature Engineering")

use_log = st.sidebar.checkbox(
    "Log-transform skewed features", value=True,
    help="Applies log1p to Impressions, Clicks, Spend"
)

use_ratio = st.sidebar.checkbox(
    "Ratio Features", value=True,
    help="Adds CTR, Cost-per-Impression, Revenue-per-Click etc."
)

use_poly = st.sidebar.checkbox(
    "Polynomial / Interaction Features (degree 2)", value=True,
    help="Adds interaction terms: Clicks×CPC, Clicks×Impressions etc."
)

st.sidebar.subheader("2. Scaler")

scaler_choice = st.sidebar.selectbox(
    "Scaler",
    ["RobustScaler", "StandardScaler", "PowerTransformer"],
    help=(
        "RobustScaler: good for outliers (IQR-based). "
        "StandardScaler: z-score. "
        "PowerTransformer: Yeo-Johnson, makes features Gaussian."
    )
)

st.sidebar.subheader("3. Models to Train")

MODEL_DEFAULTS = {
    "Ridge Regression":   True,
    "Lasso Regression":   False,
    "ElasticNet":         False,
    "Random Forest":      True,
    "Extra Trees":        True,
    "Gradient Boosting":  True,
    "XGBoost":            True,
    "LightGBM":           True,
    "KNN Regressor":      False,
}

selected_models = {
    name: st.sidebar.checkbox(name, value=default)
    for name, default in MODEL_DEFAULTS.items()
}

st.sidebar.subheader("4. Hyperparameter Tuning")

run_tuning = st.sidebar.checkbox(
    "RandomizedSearchCV on XGBoost & LightGBM", value=False,
    help="Takes ~60s longer but often improves R² significantly."
)

n_iter_search = st.sidebar.slider("Search Iterations (per model)", 5, 30, 10, 5)

st.sidebar.subheader("5. Validation")

cv_folds  = st.sidebar.slider("CV Folds", 3, 10, 5)
test_size = st.sidebar.slider("Test Set Size", 0.10, 0.40, 0.20, 0.05)

run_btn = st.sidebar.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

def engineer_features(dataframe):
    d = dataframe.copy()
    if use_log:
        for col in ["Impressions", "Clicks", "Spend"]:
            d[f"log_{col}"] = np.log1p(d[col])
    if use_ratio:
        d["CTR_eng"]              = d["Clicks"] / d["Impressions"].replace(0, np.nan)
        d["CPC_efficiency"]       = d["Spend"]  / d["Clicks"].replace(0, np.nan)
        d["Cost_per_impression"]  = d["Spend"]  / d["Impressions"].replace(0, np.nan)
        d["Clicks_per_spend"]     = d["Clicks"] / d["Spend"].replace(0, np.nan)
    d = d.fillna(0)
    return d

# ==========================================
# HELPERS
# ==========================================

def get_scaler(name):
    return {
        "RobustScaler":    RobustScaler(),
        "StandardScaler":  StandardScaler(),
        "PowerTransformer": PowerTransformer(method="yeo-johnson"),
    }[name]

def get_models():
    return {
        "Ridge Regression":  Ridge(alpha=1.0),
        "Lasso Regression":  Lasso(alpha=0.01, max_iter=5000),
        "ElasticNet":        ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=42),
        "Extra Trees":       ExtraTreesRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42),
        "XGBoost":           XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
        "LightGBM":          LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        "KNN Regressor":     KNeighborsRegressor(n_neighbors=10, weights="distance"),
    }

XGB_PARAMS = {
    "regressor__n_estimators":     [100, 200, 300, 500],
    "regressor__max_depth":        [3, 4, 5, 6, 8],
    "regressor__learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "regressor__subsample":        [0.6, 0.7, 0.8, 0.9],
    "regressor__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "regressor__min_child_weight": [1, 3, 5, 7],
    "regressor__reg_alpha":        [0, 0.01, 0.1, 1.0],
    "regressor__reg_lambda":       [0.5, 1.0, 2.0, 5.0],
}

LGBM_PARAMS = {
    "regressor__n_estimators":      [100, 200, 300, 500],
    "regressor__max_depth":         [3, 4, 5, 6, 8],
    "regressor__learning_rate":     [0.01, 0.03, 0.05, 0.1],
    "regressor__num_leaves":        [15, 31, 63, 127],
    "regressor__subsample":         [0.6, 0.7, 0.8, 0.9],
    "regressor__colsample_bytree":  [0.6, 0.7, 0.8, 1.0],
    "regressor__min_child_samples": [5, 10, 20, 50],
    "regressor__reg_alpha":         [0, 0.01, 0.1, 1.0],
}

def evaluate(name, y_true, y_pred, cv_scores):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    evs  = explained_variance_score(y_true, y_pred)
    n    = len(y_true)
    adj  = 1 - ((1 - r2) * (n - 1) / (n - 5))
    return {
        "Model":              name,
        "R²":                 round(r2, 4),
        "Adj R²":             round(adj, 4),
        "CV R² (mean)":       round(cv_scores.mean(), 4),
        "CV R² (std)":        round(cv_scores.std(), 4),
        "RMSE":               round(rmse, 2),
        "MAE":                round(mae, 2),
        "MAPE (%)":           round(mape, 2),
        "Explained Variance": round(evs, 4),
    }

# ==========================================
# PIPELINE RUN
# ==========================================

if run_btn:
    active = [k for k, v in selected_models.items() if v]
    if not active:
        st.error("Select at least one model from the sidebar.")
        st.stop()

    with st.spinner("Engineering features..."):
        df_eng = engineer_features(df)

    feat_cols = (
        ["Impressions", "Clicks", "Avg_CPC", "Spend"]
        + [c for c in df_eng.columns if c.startswith("log_")]
        + [c for c in df_eng.columns if c in [
            "CTR_eng", "CPC_efficiency", "Cost_per_impression", "Clicks_per_spend"
        ]]
    )
    feat_cols = list(dict.fromkeys(feat_cols))   # deduplicate, preserve order

    X_raw = df_eng[feat_cols].copy()
    y     = df["Conversions"]

    # Impute
    imputer  = SimpleImputer(strategy="median")
    X_imp_np = imputer.fit_transform(X_raw)
    X_imp    = pd.DataFrame(X_imp_np, columns=X_raw.columns)

    # Polynomial
    if use_poly:
        poly       = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly_np  = poly.fit_transform(X_imp)
        X_final    = pd.DataFrame(X_poly_np, columns=poly.get_feature_names_out(X_imp.columns))
    else:
        X_final = X_imp

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=test_size, random_state=42
    )

    kf         = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    all_models = get_models()
    results, trained, preds = [], {}, {}
    prog = st.progress(0, text="Starting...")

    for i, mname in enumerate(active):
        prog.progress(i / len(active), text=f"Training {mname}...")
        pipe = Pipeline([("scaler", get_scaler(scaler_choice)), ("regressor", all_models[mname])])

        if run_tuning and mname == "XGBoost":
            rs = RandomizedSearchCV(pipe, XGB_PARAMS, n_iter=n_iter_search, cv=3, scoring="r2", n_jobs=-1, random_state=42, verbose=0)
            rs.fit(X_train, y_train)
            pipe = rs.best_estimator_
            st.sidebar.success(f"XGB tuned R²: {rs.best_score_:.3f}")
        elif run_tuning and mname == "LightGBM":
            rs = RandomizedSearchCV(pipe, LGBM_PARAMS, n_iter=n_iter_search, cv=3, scoring="r2", n_jobs=-1, random_state=42, verbose=0)
            rs.fit(X_train, y_train)
            pipe = rs.best_estimator_
            st.sidebar.success(f"LGBM tuned R²: {rs.best_score_:.3f}")
        else:
            pipe.fit(X_train, y_train)

        cv_scores = cross_val_score(pipe, X_final, y, cv=kf, scoring="r2", n_jobs=-1)
        y_pred    = pipe.predict(X_test)

        results.append(evaluate(mname, y_test, y_pred, cv_scores))
        trained[mname] = pipe
        preds[mname]   = y_pred

    prog.progress(1.0, text="Done!")

    results_df = pd.DataFrame(results).sort_values("R²", ascending=False)

    st.session_state.update({
        "results_df": results_df,
        "trained":    trained,
        "preds":      preds,
        "y_test":     y_test,
        "X_test":     X_test,
        "cv_folds":   cv_folds,
    })

# ==========================================
# DISPLAY
# ==========================================

if "results_df" not in st.session_state:
    st.info("Configure the pipeline in the sidebar and click **🚀 Run Pipeline**.")
    st.stop()

results_df = st.session_state["results_df"]
trained    = st.session_state["trained"]
preds      = st.session_state["preds"]
y_test     = st.session_state["y_test"]
X_test     = st.session_state["X_test"]
cv_folds   = st.session_state["cv_folds"]

best_name = results_df.iloc[0]["Model"]
best_r2   = results_df.iloc[0]["R²"]

# -- Winner banner --
st.subheader("🏆 Best Model")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model",        best_name)
c2.metric("R²",           best_r2)
c3.metric("CV R²",        results_df.iloc[0]["CV R² (mean)"])
c4.metric("MAPE",         f"{results_df.iloc[0]['MAPE (%)']:.2f}%")

if best_r2 >= 0.90:   st.success("🟢 Excellent — R² ≥ 0.90")
elif best_r2 >= 0.75: st.warning("🟡 Good — R² ≥ 0.75")
elif best_r2 >= 0.50: st.info("🔵 Moderate — R² ≥ 0.50")
else:                  st.error("🔴 Weak — see recommendations below")

# -- Leaderboard --
st.subheader("Model Leaderboard")

def color_r2(val):
    if val >= 0.90:  return "background-color:#1b5e20;color:white"
    elif val >= 0.75: return "background-color:#f57f17;color:white"
    elif val >= 0.50: return "background-color:#0d47a1;color:white"
    else:             return "background-color:#b71c1c;color:white"

st.dataframe(
    results_df.style.map(color_r2, subset=["R²"]),
    use_container_width=True,
    hide_index=True
)

# -- R² chart --
st.subheader("R² Score Comparison")

bar_colors = [
    "#4CAF50" if r >= 0.90 else "#FFC107" if r >= 0.75
    else "#2196F3" if r >= 0.50 else "#F44336"
    for r in results_df["R²"]
]

fig_r2 = go.Figure()
fig_r2.add_trace(go.Bar(
    x=results_df["Model"], y=results_df["R²"],
    marker_color=bar_colors,
    text=results_df["R²"].apply(lambda x: f"{x:.4f}"),
    textposition="outside", name="Test R²"
))
fig_r2.add_trace(go.Scatter(
    x=results_df["Model"], y=results_df["CV R² (mean)"],
    mode="markers+lines",
    marker=dict(size=10, color="white", symbol="diamond"),
    line=dict(color="white", dash="dot"),
    name=f"CV R² ({cv_folds}-fold mean)"
))
for y_val, label, color in [(0.9, "Excellent (0.90)", "#4CAF50"), (0.75, "Good (0.75)", "#FFC107")]:
    fig_r2.add_hline(y=y_val, line_dash="dash", line_color=color, annotation_text=label)
fig_r2.update_layout(
    title="Test R² vs Cross-Validation R²",
    yaxis=dict(title="R² Score", range=[0, 1.05]), height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig_r2, use_container_width=True)

# -- MAPE chart --
st.subheader("MAPE (%) — Lower is Better")
fig_mape = px.bar(
    results_df.sort_values("MAPE (%)"),
    x="Model", y="MAPE (%)",
    color="MAPE (%)", color_continuous_scale="RdYlGn_r",
    text="MAPE (%)"
)
fig_mape.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_mape.update_layout(height=380)
st.plotly_chart(fig_mape, use_container_width=True)

# -- Predicted vs Actual --
st.subheader(f"Predicted vs Actual — {best_name}")
y_pred_best = preds[best_name]
min_v, max_v = min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())

fig_pa = go.Figure()
fig_pa.add_trace(go.Scatter(
    x=y_test.values, y=y_pred_best, mode="markers",
    marker=dict(color=np.abs(y_test.values - y_pred_best),
                colorscale="RdYlGn_r", size=5, opacity=0.7,
                colorbar=dict(title="Abs Error")),
    name="Predictions"
))
fig_pa.add_trace(go.Scatter(
    x=[min_v, max_v], y=[min_v, max_v], mode="lines",
    line=dict(color="white", dash="dash", width=2),
    name="Perfect Prediction"
))
fig_pa.update_layout(
    title=f"{best_name} — Predicted vs Actual",
    xaxis_title="Actual", yaxis_title="Predicted", height=450
)
st.plotly_chart(fig_pa, use_container_width=True)

# -- Residuals --
st.subheader(f"Residual Plot — {best_name}")
residuals = y_test.values - y_pred_best
fig_res = go.Figure()
fig_res.add_trace(go.Scatter(
    x=y_pred_best, y=residuals, mode="markers",
    marker=dict(color=residuals, colorscale="RdBu", size=5, opacity=0.6,
                colorbar=dict(title="Residual")),
    name="Residuals"
))
fig_res.add_hline(y=0, line_dash="dash", line_color="white", line_width=2)
fig_res.update_layout(
    title="Residuals vs Predicted (random scatter around 0 = good)",
    xaxis_title="Predicted", yaxis_title="Residual (Actual − Predicted)", height=380
)
st.plotly_chart(fig_res, use_container_width=True)

# -- SHAP --
TREE_MODELS = {"Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "LightGBM"}

if best_name in TREE_MODELS:
    st.subheader(f"SHAP Feature Importance — {best_name}")
    with st.spinner("Computing SHAP values..."):
        try:
            pipe_best  = trained[best_name]
            regressor  = pipe_best.named_steps["regressor"]
            X_scaled   = pd.DataFrame(
                pipe_best.named_steps["scaler"].transform(X_test),
                columns=X_test.columns
            )
            X_shap = X_scaled.sample(min(500, len(X_scaled)), random_state=42)
            explainer   = shap.Explainer(regressor)
            shap_vals   = explainer(X_shap)
            imp_df = pd.DataFrame({
                "Feature":    X_shap.columns,
                "Importance": np.abs(shap_vals.values).mean(axis=0)
            }).sort_values("Importance", ascending=False).head(20)

            st.dataframe(imp_df, use_container_width=True)
            fig_shap = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                title=f"Top 20 SHAP Features — {best_name}",
                color="Importance", color_continuous_scale="Blues"
            )
            fig_shap.update_layout(height=550, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")

# -- CV Stability --
st.subheader(f"Cross-Validation Stability ({cv_folds}-fold)")
fig_cv = go.Figure()
fig_cv.add_trace(go.Bar(
    x=results_df["Model"],
    y=results_df["CV R² (mean)"],
    error_y=dict(type="data", array=results_df["CV R² (std)"], visible=True),
    marker_color="#42A5F5",
    name="CV R² ± std"
))
fig_cv.update_layout(
    title="CV R² with Std Dev (smaller error bar = more stable model)",
    yaxis_title="CV R²", height=380
)
st.plotly_chart(fig_cv, use_container_width=True)

# -- Pipeline summary expander --
with st.expander("📋 Pipeline Configuration & Preprocessing Explanation"):
    st.markdown(f"""
| Step | Setting |
|---|---|
| Log Transform | {use_log} |
| Ratio Features | {use_ratio} |
| Polynomial Features | {use_poly} |
| Scaler | {scaler_choice} |
| CV Folds | {cv_folds} |
| Test Size | {test_size} |
| Hyperparameter Tuning | {run_tuning} |
| Models Trained | {len(results_df)} |
| Total Features after Engineering | {X_test.shape[1]} |
    """)

    st.markdown("""
**Log Transform** — Impressions, Clicks and Spend are right-skewed (a few huge campaigns dominate). 
Log-transforming compresses the tail so tree splits aren't dominated by outliers and linear models 
aren't thrown off by scale.

**Ratio Features** — 10k clicks on 1M impressions vs 10k on 50k means very different things. 
CTR, Cost-per-Impression etc. encode efficiency signal that the model can't derive from raw counts alone.

**Polynomial / Interaction Features** — Adds Clicks×CPC, Clicks×Impressions etc. so linear models 
can capture non-linear effects without needing trees.

**RobustScaler** — Uses IQR (25th–75th percentile) instead of mean/std, so a single outlier campaign 
won't distort the scale.

**PowerTransformer** — Yeo-Johnson transformation makes each feature approximately Gaussian, 
which significantly benefits Ridge, Lasso, and ElasticNet.

**Cross-Validation** — A single split can be lucky. CV R² gives expected performance on unseen data 
averaged across k folds. Prefer models with high CV mean and low CV std.
    """)

# -- Recommendations --
st.subheader("💡 How to Improve Further")

st.markdown(f"""
**Current best: {best_name} | R² = {best_r2:.4f}**

| Fix | Expected Impact |
|---|---|
| **Add CVR as a feature** — it's the direct multiplier for Conversions | +0.3–0.5 R² |
| **Enable Hyperparameter Tuning** (RandomizedSearchCV, 20+ iterations) | +0.02–0.08 R² |
| **Stack top 3 models** — meta-learner on their predictions | +0.03–0.07 R² |
| **Target-encode Campaign_Type + Device** | +0.02–0.05 R² |
| **Re-define target as Revenue** (less noisy than Conversions) | Cleaner signal |
| **Add time features** if real data: day-of-week, month, campaign age | +0.05–0.15 R² |
""")

