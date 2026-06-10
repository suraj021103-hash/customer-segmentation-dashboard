
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# LOAD DATA & MODELS
# =====================================================

df = pd.read_csv("Mall_Customers.csv")

model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# =====================================================
# PREPROCESSING
# =====================================================

df_encoded = df.copy()

df_encoded["Gender"] = df_encoded["Gender"].map({
    "Male": 0,
    "Female": 1
})

features = [
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df_encoded[features]

X_scaled = scaler.transform(X)

clusters = model.predict(X_scaled)

df["Cluster"] = clusters

# =====================================================
# SEGMENT NAMES
# =====================================================

cluster_names = {
    0: "Premium Customers",
    1: "Careful Rich Customers",
    2: "Average Customers",
    3: "Young Spenders",
    4: "Senior Customers"
}

df["Segment"] = df["Cluster"].map(cluster_names)

# =====================================================
# RECOMMENDATIONS
# =====================================================

recommendations = {
    "Premium Customers":
        "Offer VIP memberships, premium products, and exclusive discounts.",

    "Careful Rich Customers":
        "Provide personalized offers and loyalty rewards to increase spending.",

    "Average Customers":
        "Promote regular deals and cross-sell related products.",

    "Young Spenders":
        "Target with trendy products, social media campaigns, and seasonal offers.",

    "Senior Customers":
        "Focus on comfort products, healthcare-related offers, and personalized service."
}

# =====================================================
# BUSINESS INSIGHT FUNCTION
# =====================================================

def generate_business_insight(age, income, spending):

    if income > 80 and spending > 70:
        return (
            "High-value customers. Prioritize loyalty programs, "
            "VIP benefits, and premium products."
        )

    elif income > 80 and spending < 40:
        return (
            "Customers have strong purchasing power but spend cautiously. "
            "Use targeted promotions to improve engagement."
        )

    elif age > 50:
        return (
            "Older customers with conservative spending patterns. "
            "Focus on trust, service quality, and personalized experiences."
        )

    elif spending > 60:
        return (
            "Active spenders who respond well to marketing campaigns "
            "and trend-based products."
        )

    elif income < 60 and spending < 50:
        return (
            "Price-sensitive customers. Focus on discounts, bundles, "
            "and value-based offerings."
        )

    else:
        return (
            "Balanced customer segment suitable for broad marketing campaigns."
        )

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Go To",
    [
        "Home",
        "Dataset Overview",
        "Clustering Analysis",
        "Customer Segments",
        "Predict Customer"
    ]
)

# =====================================================
# HOME
# =====================================================

if page == "Home":

    st.title("📊 Customer Segmentation Dashboard")

    st.markdown("""
    This dashboard helps businesses understand customer behavior using:

    - K-Means Clustering
    - PCA Visualization
    - Customer Segmentation
    - Marketing Recommendations
    - Business Insights
    """)

    avg_income = df["Annual Income (k$)"].mean()
    avg_spending = df["Spending Score (1-100)"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers", df.shape[0])
    c2.metric("Segments", df["Segment"].nunique())
    c3.metric("Avg Income", f"${avg_income:.1f}k")
    c4.metric("Avg Spending", f"{avg_spending:.1f}")

    st.success(
        """
        This dashboard enables customer segmentation using K-Means clustering,
        PCA-based visualization, and explainable business insights for targeted marketing.
        """
    )

# =====================================================
# DATASET OVERVIEW
# =====================================================

elif page == "Dataset Overview":

    st.header("📂 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Customers", df.shape[0])
    col2.metric("Features", 5)
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Feature Information")
    st.write(df.dtypes)

    # AGE

    st.subheader("Age Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # INCOME

    st.subheader("Annual Income Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        df["Annual Income (k$)"],
        bins=20,
        kde=True,
        ax=ax
    )
    st.pyplot(fig)

    # SPENDING

    st.subheader("Spending Score Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        df["Spending Score (1-100)"],
        bins=20,
        kde=True,
        ax=ax
    )
    st.pyplot(fig)

    # HEATMAP

    st.subheader("Correlation Heatmap")

    numeric_df = df_encoded.select_dtypes(
        include=["int64", "float64"]
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig)

# =====================================================
# CLUSTERING ANALYSIS
# =====================================================

elif page == "Clustering Analysis":

    st.header("📈 Clustering Analysis")

    pca = PCA(n_components=2)

    pca_features = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        pca_features,
        columns=["PCA1", "PCA2"]
    )

    pca_df["Cluster"] = df["Cluster"]

    variance = pca.explained_variance_ratio_.sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Optimal Clusters", "5")
    col2.metric("PCA Variance Retained", f"{variance:.2%}")
    col3.metric("Model", "K-Means")

    st.success(
        """
        Explainable AI Insight

        The Elbow Method identified 5 meaningful customer segments.

        PCA retained most of the important customer information
        while reducing dimensionality for visualization.

        This enables non-technical users to understand customer
        groups without interpreting raw clustering metrics.
        """
    )

    st.subheader("PCA Cluster Visualization")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=pca_df,
        x="PCA1",
        y="PCA2",
        hue="Cluster",
        palette="deep",
        s=80,
        ax=ax
    )

    st.pyplot(fig)

# =====================================================
# CUSTOMER SEGMENTS
# =====================================================

elif page == "Customer Segments":

    st.header("👥 Customer Segments")
    st.subheader("Segment Distribution")

    segment_counts = df["Segment"].value_counts()

    st.bar_chart(segment_counts)

    cluster_summary = df.groupby("Segment").agg({
        "Age": "mean",
        "Annual Income (k$)": "mean",
        "Spending Score (1-100)": "mean"
    }).round(2)

    st.subheader("Segment Statistics")

    st.dataframe(cluster_summary)
    csv = cluster_summary.to_csv().encode("utf-8")

    st.download_button(
        "📥 Download Segment Report",
        csv,
        "customer_segments.csv",
        "text/csv"
    )

    st.subheader("Business Insights")

    for segment in cluster_summary.index:

        row = cluster_summary.loc[segment]

        insight = generate_business_insight(
            row["Age"],
            row["Annual Income (k$)"],
            row["Spending Score (1-100)"]
        )

        st.markdown(f"### {segment}")

        st.write(insight)

        st.write(
            f"Recommendation: {recommendations[segment]}"
        )

        st.divider()

# =====================================================
# PREDICT CUSTOMER
# =====================================================

elif page == "Predict Customer":

    st.header("🎯 Predict Customer Segment")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.slider(
        "Age",
        18,
        70,
        30
    )

    income = st.slider(
        "Annual Income (k$)",
        10,
        150,
        60
    )

    spending = st.slider(
        "Spending Score",
        1,
        100,
        50
    )

    if st.button("Predict Segment"):

        gender_encoded = 0 if gender == "Male" else 1

        new_customer = pd.DataFrame({
            "Gender": [gender_encoded],
            "Age": [age],
            "Annual Income (k$)": [income],
            "Spending Score (1-100)": [spending]
        })

        scaled_customer = scaler.transform(
            new_customer
        )

        cluster = model.predict(
            scaled_customer
        )[0]

        segment = cluster_names[cluster]

        recommendation = recommendations[segment]

        insight = generate_business_insight(
            age,
            income,
            spending
        )

        st.success(
            f"Predicted Segment: {segment}"
        )

        st.info(
            f"Recommendation: {recommendation}"
        )

        st.warning(
            f"Business Insight: {insight}"
        )

st.markdown("---")

st.caption(
    "Customer Segmentation Dashboard | K-Means • PCA • Streamlit • Explainable AI"
)