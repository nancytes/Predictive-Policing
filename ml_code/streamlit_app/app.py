

# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import joblib
# from datetime import datetime
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ======================================================
# # Page Configuration
# # ======================================================
# st.set_page_config(
#     page_title="Crime Prediction Dashboard",
#     page_icon="🚔",
#     layout="wide"
# )

# # ======================================================
# # Load Dataset
# # ======================================================
# DATA_PATH = "ml_code/models/daily_features.csv"

# try:
#     df = pd.read_csv(DATA_PATH)
#     df["date"] = pd.to_datetime(df["date"], errors="coerce")
# except Exception:
#     st.error("❌ Dataset not found or invalid. Ensure 'daily_features.csv' is in ml_code/models/")
#     st.stop()

# # Detect target column
# target_col = next((c for c in ["crime_count", "count", "daily_count", "value"] if c in df.columns), None)
# if target_col is None:
#     st.error("❌ No valid crime count column found.")
#     st.stop()
# df[target_col] = df[target_col].fillna(0).astype(int)

# # ======================================================
# # Load Models
# # ======================================================
# MODEL_DIR = "ml_code/models"
# model_paths = {
#     "Random Forest": f"{MODEL_DIR}/RandomForest.pkl",
#     "Gradient Boosting": f"{MODEL_DIR}/GradientBoosting.pkl",
#     "Linear Regression": f"{MODEL_DIR}/LinearRegression.pkl",
#     "Final Model (Random Forest)": f"{MODEL_DIR}/final_model.pkl"
# }

# models = {}
# for name, path in model_paths.items():
#     try:
#         try:
#             models[name] = joblib.load(path)
#         except:
#             with open(path, "rb") as f:
#                 models[name] = pickle.load(f)
#     except Exception:
#         models[name] = None

# available_models = [m for m in models if models[m] is not None]

# # ======================================================
# # Custom Sidebar — Professional & Modern
# # ======================================================
# with st.sidebar:
#     st.markdown("## 🚔 Crime Prediction Dashboard")
#     st.markdown("Interactive ML dashboard for crime trend analysis and prediction.")
#     st.markdown("---")

#     st.markdown("### 📍 Navigation")
#     page = st.radio(
#         label="Select Section",
#         options=["Overview", "Prediction", "Model Evaluation"],
#         index=0
#     )

#     st.markdown("---")

#     st.markdown("### ⚙️ Prediction Settings")
#     selected_model = st.selectbox("🔹 Choose Model", available_models)
#     selected_date = st.date_input("📅 Select Prediction Date", value=df["date"].max().date())

#     st.markdown("---")

#     with st.expander("🧠 Advanced Options", expanded=False):
#         st.markdown("Optional controls for advanced users.")
#         show_feature_importance = st.checkbox("Show Feature Importance (RF)", value=True)
#         show_metrics_table = st.checkbox("Show Full Metrics Table", value=True)
#         show_heatmap = st.checkbox("Show Correlation Heatmap", value=True)

#     st.markdown("---")
#     st.markdown("### ❓ Help & Info")
#     st.markdown(
#         "Developed for Big Data Analytics project.\n"
#         "Predictions are for academic analysis only."
#     )

# # ======================================================
# # ------------------- Overview Page -------------------
# # ======================================================
# if page == "Overview":
#     st.title("🚔 Crime Prediction Dashboard")
#     st.markdown("""
#     **Overview:** This dashboard demonstrates ML-based crime prediction.
#     It includes historical trends, feature insights, and real-time prediction capabilities.
#     """)

#     # KPI Metrics
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("📊 Total Records", f"{len(df):,}")
#     col2.metric("🏙️ Cities Covered", "2")
#     col3.metric("🤖 Models Available", f"{len(available_models)}")
#     col4.metric("📅 Date Range", f"{df['date'].min().date()} → {df['date'].max().date()}")

#     st.markdown("---")
#     st.header("📈 Historical Crime Trends")
#     crime_ts = df.groupby("date")[target_col].sum().reset_index()
#     st.line_chart(crime_ts.set_index("date")[target_col])

#     st.markdown("---")
#     st.header("📊 Crime Trends by Month")
#     df['month'] = df['date'].dt.month
#     monthly = df.groupby('month')[target_col].sum().reset_index()
#     st.bar_chart(monthly.set_index('month'))

#     st.markdown("---")
#     st.header("🗂 Dataset Preview (First 100 Rows)")
#     st.dataframe(df.head(100), use_container_width=True)

#     st.markdown("---")
#     st.subheader("📊 Dataset Summary")
#     summary_df = pd.DataFrame({
#         "Column": df.columns,
#         "Data Type": [str(dtype) for dtype in df.dtypes],
#         "Missing Values": df.isna().sum(),
#         "Unique Values": df.nunique()
#     })
#     st.table(summary_df)

#     # Correlation Heatmap
#     if 'show_heatmap' in locals() and show_heatmap:
#         st.markdown("---")
#         st.header("🔥 Correlation Heatmap")
#         corr = df.select_dtypes(include=np.number).corr()
#         fig, ax = plt.subplots(figsize=(10,6))
#         sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
#         st.pyplot(fig)

# # ======================================================
# # ------------------- Prediction Page -----------------
# # ======================================================
# if page == "Prediction":
#     st.title("🔍 Real-Time Crime Prediction")
#     input_features = pd.DataFrame([{
#         "day": selected_date.day,
#         "month": selected_date.month,
#         "year": selected_date.year,
#         "day_of_week": selected_date.weekday(),
#         "is_weekend": 1 if selected_date.weekday() >= 5 else 0
#     }])

#     try:
#         model = models[selected_model]
#         prediction = model.predict(input_features)[0]
#         st.subheader(f"📈 Predicted Crime Count: **{int(prediction):,}**")
#         st.caption("Expected crimes for the selected date based on historical patterns.")
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")

# # ======================================================
# # ------------------- Model Evaluation ----------------
# # ======================================================
# if page == "Model Evaluation":
#     st.title("📊 Model Performance Comparison")
#     metrics = []
#     feature_cols = ["day", "month", "year", "day_of_week", "is_weekend"]
#     X = df[feature_cols].astype(float)
#     y = df[target_col]

#     for name, model in models.items():
#         if model is not None:
#             try:
#                 preds = model.predict(X)
#                 metrics.append({
#                     "Model": name,
#                     "R² Score": round(r2_score(y, preds), 3),
#                     "MAE": round(mean_absolute_error(y, preds), 2),
#                     "RMSE": round(np.sqrt(mean_squared_error(y, preds)), 2)
#                 })
#             except:
#                 metrics.append({"Model": name, "R² Score": "Error", "MAE": "Error", "RMSE": "Error"})

#     perf_df = pd.DataFrame(metrics)
#     if show_metrics_table:
#         st.table(perf_df)

#     # Feature Importance for Random Forest
#     if show_feature_importance and "Random Forest" in models and models["Random Forest"] is not None:
#         st.header("📌 Feature Importance (Random Forest)")
#         importances = models["Random Forest"].feature_importances_
#         fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values("Importance", ascending=False)
#         st.bar_chart(fi_df.set_index("Feature"))

#     st.markdown("---")
#     st.markdown("""
#     ### 🏆 Final Model Selection
#     - **Random Forest** selected as final model due to highest predictive accuracy
#     - Effectively captures non-linear patterns in temporal crime data
#     """)



import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# Page Configuration
# ======================================================
st.set_page_config(
    page_title="Crime Prediction Dashboard",
    page_icon="🚔",
    layout="wide"
)

# ======================================================
# Load Dataset
# ======================================================
DATA_PATH = "ml_code/models/daily_features.csv"

try:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
except Exception:
    st.error("❌ Dataset not found or invalid. Ensure 'daily_features.csv' is in ml_code/models/")
    st.stop()

# Detect target column
target_col = next((c for c in ["crime_count", "count", "daily_count", "value"] if c in df.columns), None)
if target_col is None:
    st.error("❌ No valid crime count column found.")
    st.stop()
df[target_col] = df[target_col].fillna(0).astype(int)

# ======================================================
# Load Models
# ======================================================
MODEL_DIR = "ml_code/models"
model_paths = {
    "Random Forest": f"{MODEL_DIR}/RandomForest.pkl",
    "Gradient Boosting": f"{MODEL_DIR}/GradientBoosting.pkl",
    "Linear Regression": f"{MODEL_DIR}/LinearRegression.pkl",
    "Final Model (Random Forest)": f"{MODEL_DIR}/final_model.pkl"
}

models = {}
for name, path in model_paths.items():
    try:
        try:
            models[name] = joblib.load(path)
        except:
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    except Exception:
        models[name] = None

available_models = [m for m in models if models[m] is not None]

# ======================================================
# Custom Sidebar
# ======================================================
with st.sidebar:
    st.markdown("## 🚔 Crime Prediction Dashboard")
    # st.markdown("Interactive ML dashboard for crime trend analysis and prediction.")
    # st.markdown("---")

    st.markdown("### 📍 Navigation")
    page = st.radio(
        label="Select Section",
        options=["Overview", "Prediction", "Model Evaluation"],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Prediction Settings")
    selected_model = st.selectbox("🔹 Choose Model", available_models)
    selected_date = st.date_input("📅 Select Prediction Date", value=df["date"].max().date())

    st.markdown("---")
    with st.expander("🧠 Advanced Options", expanded=False):
        st.markdown("Optional controls for advanced users.")
        show_feature_importance = st.checkbox("Show Feature Importance (RF)", value=True)
        show_metrics_table = st.checkbox("Show Full Metrics Table", value=True)
        show_heatmap = st.checkbox("Show Correlation Heatmap", value=True)

    st.markdown("---")
    st.markdown("### ❓ Help & Info")
    st.markdown(
        "Developed for Big Data Analytics project.\n"
        "Predictions are for academic analysis only."
    )

# ======================================================
# ------------------- Overview Page -------------------
# ======================================================
if page == "Overview":
    st.title("🚔 Crime Prediction Dashboard")
    st.markdown("""
    **Overview:** This dashboard demonstrates ML-based crime prediction.
    It includes historical trends, feature insights, and real-time prediction capabilities.
    """)

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Records", f"{len(df):,}")
    col2.metric("🏙️ Cities Covered", "2")
    col3.metric("🤖 Models Available", f"{len(available_models)}")
    col4.metric("📅 Date Range", f"{df['date'].min().date()} → {df['date'].max().date()}")

    st.markdown("---")
    st.header("📈 Historical Crime Trends")
    crime_ts = df.groupby("date")[target_col].sum().reset_index()
    st.line_chart(crime_ts.set_index("date")[target_col])

    st.markdown("---")
    st.header("📊 Crime Trends by Month")
    df['month'] = df['date'].dt.month
    monthly = df.groupby('month')[target_col].sum().reset_index()
    st.bar_chart(monthly.set_index('month'))

    st.markdown("---")
    st.header("🗂 Dataset Preview (First 100 Rows)")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Dataset Summary")
    summary_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": df.isna().sum(),
        "Unique Values": df.nunique()
    })
    st.table(summary_df)

    # Correlation Heatmap
    if 'show_heatmap' in locals() and show_heatmap:
        st.markdown("---")
        st.header("🔥 Correlation Heatmap")
        corr = df.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(fig)

# ======================================================
# ------------------- Prediction Page -----------------
# ======================================================
if page == "Prediction":
    st.title("🔍 Real-Time Crime Prediction")

    # ---------------- Input Features -------------------
    input_features = pd.DataFrame([{
        "day": selected_date.day,
        "month": selected_date.month,
        "year": selected_date.year,
        "day_of_week": selected_date.weekday(),
        "is_weekend": 1 if selected_date.weekday() >= 5 else 0
    }])

    # ---------------- Prediction -----------------------
    try:
        model = models[selected_model]
        prediction = model.predict(input_features)[0]
        st.subheader(f"📈 Predicted Crime Count: **{int(prediction):,}**")
        st.caption("Expected crimes for the selected date based on historical patterns.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.header("📊 Crime Insights & Visualizations")

    # ---------------- Recent 30 Days ----------------
    st.subheader("1️⃣ Comparison with Recent 30 Days")
    recent_dates = df.sort_values("date", ascending=False).head(30)
    recent_dates_chart = recent_dates[['date', target_col]].set_index('date')
    st.line_chart(recent_dates_chart)

    # ---------------- Day of Week -----------------
    st.subheader("2️⃣ Average Crime by Day of Week")
    df['day_of_week'] = df['date'].dt.dayofweek
    dow = df.groupby('day_of_week')[target_col].mean().reindex([0,1,2,3,4,5,6])
    dow.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    st.bar_chart(dow)

    # ---------------- Distribution -----------------
    st.subheader("3️⃣ Crime Count Distribution")
    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(df[target_col], bins=30, kde=True, color='skyblue', ax=ax)
    ax.set_xlabel("Crime Count")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # ---------------- Rolling Average --------------------
    st.subheader("4️⃣ 7-Day Rolling Average")
    df_sorted = df.sort_values("date")
    df_sorted['rolling_7'] = df_sorted[target_col].rolling(7).mean()
    st.line_chart(df_sorted.set_index("date")[['rolling_7']])

    # ---------------- Heatmap -------------------------------
    st.subheader("5️⃣ Heatmap: Crime by Month vs Day of Week")
    df['month'] = df['date'].dt.month
    heatmap_data = df.pivot_table(index='day_of_week', columns='month', values=target_col, aggfunc='mean')
    heatmap_data.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlOrRd', ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Day of Week")
    st.pyplot(fig)

# ======================================================
# ------------------- Model Evaluation ----------------
# ======================================================
if page == "Model Evaluation":
    st.title("📊 Model Performance Comparison")
    metrics = []
    feature_cols = ["day", "month", "year", "day_of_week", "is_weekend"]
    X = df[feature_cols].astype(float)
    y = df[target_col]

    for name, model in models.items():
        if model is not None:
            try:
                preds = model.predict(X)
                metrics.append({
                    "Model": name,
                    "R² Score": round(r2_score(y, preds), 3),
                    "MAE": round(mean_absolute_error(y, preds), 2),
                    "RMSE": round(np.sqrt(mean_squared_error(y, preds)), 2)
                })
            except:
                metrics.append({"Model": name, "R² Score": "Error", "MAE": "Error", "RMSE": "Error"})

    perf_df = pd.DataFrame(metrics)
    if show_metrics_table:
        st.table(perf_df)

    # Feature Importance for Random Forest
    if show_feature_importance and "Random Forest" in models and models["Random Forest"] is not None:
        st.header("📌 Feature Importance (Random Forest)")
        importances = models["Random Forest"].feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values("Importance", ascending=False)
        st.bar_chart(fi_df.set_index("Feature"))

    st.markdown("---")
    st.markdown("""
    ### 🏆 Final Model Selection
    - **Random Forest** selected as final model due to highest predictive accuracy
    - Effectively captures non-linear patterns in temporal crime data
    """)
