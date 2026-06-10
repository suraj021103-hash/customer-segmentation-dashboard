
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

    c1, c2, c3 = st.columns(3)

    c1.metric("Customers", df.shape[0])
    c2.metric("Features", df.shape[1] - 2)
    c3.metric("Segments", df["Segment"].nunique())

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

    st.subheader("Elbow Method Interpretation")

    st.info(
        """
        The Elbow Method indicates that K=5 is the optimal number
        of customer segments.

        Using fewer clusters may combine customers with different
        purchasing behaviors, while using more clusters adds
        unnecessary complexity without significant business value.
        """
    )

    st.subheader("Silhouette Score")

    st.success(
        "K=5 was selected using Elbow Method and validated through Silhouette Analysis."
    )

    # PCA

    pca = PCA(n_components=2)

    pca_features = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        pca_features,
        columns=["PCA1", "PCA2"]
    )

    pca_df["Cluster"] = df["Cluster"]

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

    variance = pca.explained_variance_ratio_.sum()

    st.write(
        f"Total Variance Explained by PCA: {variance:.2%}"
    )

# =====================================================
# CUSTOMER SEGMENTS
# =====================================================

elif page == "Customer Segments":

    st.header("👥 Customer Segments")

    cluster_summary = df.groupby("Segment").agg({
        "Age": "mean",
        "Annual Income (k$)": "mean",
        "Spending Score (1-100)": "mean"
    }).round(2)

    st.subheader("Segment Statistics")

    st.dataframe(cluster_summary)

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

        st.subheader("Recommendation")
        st.write(recommendation)

        st.subheader("Business Insight")
        st.write(insight)
