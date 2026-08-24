"""
app.py
------
Streamlit app for the Supply Chain Late Delivery Risk Predictor.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this whole folder to a public GitHub repo.
    2. Go to https://share.streamlit.io -> "New app".
    3. Point it at your repo, branch = main, main file path = app.py.
    4. Deploy. Streamlit Cloud installs requirements.txt automatically.
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Supply Chain Delay Predictor", page_icon="📦", layout="wide")

st.title("📦 Supply Chain Late Delivery Risk Predictor")

# ----------------------------------------------------------------------
# Load artifacts (cached so it only loads once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("artifacts/model_pipeline.pkl")
    metadata = joblib.load("artifacts/metadata.pkl")
    return pipeline, metadata


@st.cache_data
def load_raw_data():
    return pd.read_csv("data/supply_chain_data.csv")


pipeline, metadata = load_artifacts()

cat_features = metadata["categorical_features"]
num_features = metadata["numeric_features"]
feature_options = metadata["feature_options"]
numeric_ranges = metadata["numeric_ranges"]
metrics = metadata["metrics"]
confusion_matrix_data = metadata["confusion_matrix"]
feature_importance = metadata["feature_importance"]

Tab_predict, Tab_insights = st.tabs(["🔮 Predict", "📊 Model Insights"])

# ========================================================================
# TAB 1 — Prediction form
# ========================================================================
with Tab_predict:
    st.write(
        "Enter order details below to predict whether this order is likely to be "
        "**delivered late**"
    )

    with st.form("prediction_form"):
        st.subheader("Order Details")

        user_input = {}

        col1, col2 = st.columns(2)
        for i, col in enumerate(cat_features):
            target_col = col1 if i % 2 == 0 else col2
            user_input[col] = target_col.selectbox(col.replace("_", " "), feature_options[col])

        col3, col4 = st.columns(2)
        for i, col in enumerate(num_features):
            low, high = numeric_ranges[col]
            target_col = col3 if i % 2 == 0 else col4
            default_val = (low + high) / 2
            user_input[col] = target_col.number_input(
                col.replace("_", " "), min_value=float(low), max_value=float(high), value=float(default_val)
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])[cat_features + num_features]

        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ High risk of **late delivery** probability: {probability:.1%}")
        else:
            st.success(f"Likely **on-time delivery** Late Risk Probability: {probability:.1%}")

        st.progress(min(float(probability), 1.0))

        with st.expander("See what you entered"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}))

    st.divider()
    st.caption(
        "Model: XGBoost Classifier | Preprocessing: OneHotEncoder + StandardScaler in a single sklearn Pipeline"
    )

# ========================================================================
# TAB 2 — Model Insights (EDA + evaluation graphs for your dashboard)
# ========================================================================
with Tab_insights:
    st.write("Everything below comes straight from the artifacts saved during training in `train.py`.")

    # ---- headline metrics --------------------------------------------
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("F1 Score (Late class)", f"{metrics['f1_score']:.3f}")
    m3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.divider()

    raw_df = load_raw_data()

    col_a, col_b = st.columns(2)

    # ---- 1. Class balance ----------------------------------------------
    with col_a:
        st.subheader("Target class balance")
        class_counts = raw_df["Late_Delivery_Risk"].map({0: "On time", 1: "Late"}).value_counts()
        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            hole=0.45,
            color=class_counts.index,
            color_discrete_map={"On time": "#2ecc71", "Late": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- 2. Confusion matrix --------------------------------------------
    with col_b:
        st.subheader("Confusion matrix (test set)")
        cm = np.array(confusion_matrix_data)
        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted: On time", "Predicted: Late"],
            y=["Actual: On time", "Actual: Late"],
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    # ---- 3. Feature importance ------------------------------------------
    with col_c:
        st.subheader("Top feature importances")
        fi_series = pd.Series(feature_importance).sort_values(ascending=True)
        fig = px.bar(x=fi_series.values, y=fi_series.index, orientation="h")
        fig.update_layout(xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # ---- 4. Late-delivery rate by shipping mode -------------------------
    with col_d:
        st.subheader("Late-delivery rate by Shipping Mode")
        rate = raw_df.groupby("Shipping_Mode")["Late_Delivery_Risk"].mean().sort_values(ascending=False)
        fig = px.bar(x=rate.index, y=rate.values, labels={"x": "Shipping Mode", "y": "Late rate"})
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col_e, col_f = st.columns(2)

    # ---- 5. Correlation heatmap (numeric features) -----------------------
    with col_e:
        st.subheader("Correlation heatmap (numeric features)")
        corr = raw_df[num_features + ["Late_Delivery_Risk"]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)

    # ---- 6. PCA projection (visualization only — NOT fed into the model) --
    with col_f:
        st.subheader("PCA projection of orders (2D)")
        st.caption("For visualization only — the model itself trains on the original features, not these components.")

        sample_df = raw_df.sample(min(2000, len(raw_df)), random_state=42)
        numeric_sample = sample_df[num_features]
        pca = PCA(n_components=2, random_state=42)
        components = pca.fit_transform(
            (numeric_sample - numeric_sample.mean()) / numeric_sample.std()
        )
        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
        pca_df["Late_Delivery_Risk"] = sample_df["Late_Delivery_Risk"].map({0: "On time", 1: "Late"}).values

        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Late_Delivery_Risk",
            opacity=0.6,
            color_discrete_map={"On time": "#2ecc71", "Late": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Explained variance: PC1 {pca.explained_variance_ratio_[0]:.1%}, "
            f"PC2 {pca.explained_variance_ratio_[1]:.1%}"
        )
