import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 End-to-End Sales Forecasting & Demand Intelligence System")

# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv("train.csv")

df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month

st.success("Dataset Loaded Successfully!")

# -------------------------------
# Sidebar Filters
# -------------------------------

st.sidebar.header("Dashboard Filters")

category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + sorted(df["Category"].unique().tolist())
)

region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + sorted(df["Region"].unique().tolist())
)

filtered_df = df.copy()

if category != "All":
    filtered_df = filtered_df[
        filtered_df["Category"] == category
    ]

if region != "All":
    filtered_df = filtered_df[
        filtered_df["Region"] == region
    ]

# -------------------------------
# KPI Cards
# -------------------------------

st.header("📈 Sales Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Sales",
    f"${filtered_df['Sales'].sum():,.2f}"
)

col2.metric(
    "Total Orders",
    filtered_df["Order ID"].nunique()
)

col3.metric(
    "Total Customers",
    filtered_df["Customer ID"].nunique()
)

# -------------------------------
# Yearly Sales
# -------------------------------

st.subheader("Yearly Sales")

yearly_sales = filtered_df.groupby("Year")["Sales"].sum()

fig, ax = plt.subplots(figsize=(8,4))

yearly_sales.plot(
    kind="bar",
    ax=ax,
    color="steelblue"
)

ax.set_xlabel("Year")
ax.set_ylabel("Sales")
ax.set_title("Yearly Sales")

st.pyplot(fig)

# -------------------------------
# Monthly Sales Trend
# -------------------------------

st.subheader("Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .set_index("Order Date")
    .resample("M")["Sales"]
    .sum()
)

fig2, ax2 = plt.subplots(figsize=(10,4))

monthly_sales.plot(ax=ax2)

ax2.set_xlabel("Date")
ax2.set_ylabel("Sales")
ax2.set_title("Monthly Sales Trend")

st.pyplot(fig2)

# -------------------------------
# Category Sales
# -------------------------------

st.subheader("Sales by Category")

category_sales = filtered_df.groupby("Category")["Sales"].sum()

fig3, ax3 = plt.subplots(figsize=(8,4))

category_sales.plot(
    kind="bar",
    ax=ax3,
    color="orange"
)

ax3.set_xlabel("Category")
ax3.set_ylabel("Sales")
ax3.set_title("Sales by Category")

st.pyplot(fig3)

# -------------------------------
# Region Sales
# -------------------------------

st.subheader("Sales by Region")

region_sales = filtered_df.groupby("Region")["Sales"].sum()

fig4, ax4 = plt.subplots(figsize=(8,4))

region_sales.plot(
    kind="bar",
    ax=ax4,
    color="green"
)

ax4.set_xlabel("Region")
ax4.set_ylabel("Sales")
ax4.set_title("Sales by Region")

st.pyplot(fig4)

# -------------------------------
# Best Forecasting Model
# -------------------------------

st.header("🤖 Model Comparison")

comparison = pd.DataFrame({

    "Model":[
        "SARIMA",
        "Prophet",
        "XGBoost"
    ],

    "MAE":[
        18031.40,
        40970.33,
        14443.46
    ],

    "RMSE":[
        19009.18,
        53868.95,
        17069.09
    ],

    "MAPE":[
        18.97,
        40.04,
        14.45
    ]

})

st.dataframe(comparison)

st.success("🏆 Best Model: XGBoost")

# -------------------------------
# Saved Charts
# -------------------------------

st.header("📊 Analysis Charts")

chart_list = [

    ("Monthly Sales Trend","charts/monthly_sales_trend.png"),

    ("Time Series Decomposition","charts/time_series_decomposition.png"),

    ("Category Forecast","charts/category_region_forecast.png"),

    ("Isolation Forest","charts/isolation_forest.png"),

    ("Z-Score Anomaly","charts/zscore_anomaly.png"),

    ("Product Clusters","charts/product_clusters.png")

]

for title, path in chart_list:

    st.subheader(title)

    try:
        st.image(path, use_container_width=True)
    except:
        st.warning(f"{path} not found.")

# -------------------------------
# Dataset Preview
# -------------------------------

st.header("Dataset Preview")

st.dataframe(filtered_df.head(20))

# -------------------------------
# Footer
# -------------------------------

st.markdown("---")

st.success("Project Developed by Akshitha")

st.caption("End-to-End Sales Forecasting & Demand Intelligence System")