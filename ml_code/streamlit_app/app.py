
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# from datetime import datetime
# import joblib
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# # ------------------------------------------------------
# # Page Config
# # ------------------------------------------------------
# st.set_page_config(
#     page_title="Crime Prediction Dashboard",
#     page_icon="🚔",
#     layout="wide"
# )

# # ------------------------------------------------------
# # Styling – Red → Black gradient
# # ------------------------------------------------------
# st.markdown("""
# <style>
# :root { --bg1:  #00a0ef; --bg2: #011753; }

# .reportview-container {
#     background: linear-gradient(180deg, var(--bg1), var(--bg2));
#     color: #f5f5f5;
# }
# .stApp {
#     background: linear-gradient(180deg, var(--bg1), var(--bg2));
# }
# .kpi {
#     background: rgba(255,255,255,0.07);
#     padding: 18px;
#     border-radius: 12px;
#     border: 1px solid rgba(255,255,255,0.2);
#     text-align: center;
# }
# .kpi h3 { margin: 0; color: #fff; }
# .kpi p { margin: 0; font-size: 22px; color: #ffdfdf; }
# .big-title { font-size: 38px; font-weight: 700; color: #fff; }
# </style>
# """, unsafe_allow_html=True)

# # ------------------------------------------------------
# # Load Data
# # ------------------------------------------------------
# DATA_PATH = "ml_code/models/daily_features.csv"

# try:
#     df = pd.read_csv(DATA_PATH)
# except:
#     st.error("❌ daily_features.csv not found.")
#     st.stop()

# # Ensure date column exists
# if "date" not in df.columns:
#     st.error("❌ No 'date' column found in the data.")
#     st.stop()

# df["date"] = pd.to_datetime(df["date"], errors="coerce")

# # Locate count column (or build one)
# count_col = None
# for c in ["count", "crime_count", "daily_count", "value"]:
#     if c in df.columns:
#         count_col = c
#         break

# if count_col is None:
#     df["count"] = 1
#     count_col = "count"

# df[count_col] = df[count_col].fillna(0).astype(int)

# # ------------------------------------------------------
# # Load Models
# # ------------------------------------------------------
# MODEL_DIR = "ml_code/models"
# model_files = {
#     "Random Forest": f"{MODEL_DIR}/RandomForest.pkl",
#     "Gradient Boosting": f"{MODEL_DIR}/GradientBoosting.pkl",
#     "Linear Regression": f"{MODEL_DIR}/LinearRegression.pkl",
#     "Final Model": f"{MODEL_DIR}/final_model.pkl",
# }

# # models = {}
# # for name, path in model_files.items():
# #     try:
# #         with open(path, "rb") as f:
# #             models[name] = pickle.load(f)
# #     except:
# #         models[name] = None

# # available_models = [m for m in models if models[m] is not None]
# models = {}

# for name, path in model_files.items():
#     try:
#         # joblib first (handles pipelines)
#         try:
#             models[name] = joblib.load(path)
#         except:
#             # fallback to pickle
#             with open(path, "rb") as f:
#                 models[name] = pickle.load(f)
#     except Exception as e:
#         models[name] = None
#         st.sidebar.error(f"❌ Failed to load {name}: {e}")

# # Create list of successfully loaded models
# available_models = [m for m in models if models[m] is not None]
# # ------------------------------------------------------
# # Sidebar Controls
# # ------------------------------------------------------
# st.sidebar.title("⚙️ Controls")

# selected_model = st.sidebar.selectbox("Select Model", available_models)

# selected_date = st.sidebar.date_input(
#     "Select Date",
#     value=df["date"].max().date()
# )

# # ------------------------------------------------------
# # Header + KPIs
# # ------------------------------------------------------
# st.markdown("<div class='big-title'>🚔 Crime Prediction Dashboard</div>", unsafe_allow_html=True)

# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.markdown("<div class='kpi'><h3>📊 Total Records</h3><p>{:,}</p></div>".format(len(df)), unsafe_allow_html=True)

# with col2:
#     st.markdown("<div class='kpi'><h3>🏙️ Cities Covered</h3><p>2</p></div>", unsafe_allow_html=True)

# with col3:
#     st.markdown("<div class='kpi'><h3>🤖 ML Models</h3><p>{}</p></div>".format(len(available_models)), unsafe_allow_html=True)

# with col4:
#     st.markdown(f"<div class='kpi'><h3>📅 Date Range</h3><p>{df['date'].min().date()} → {df['date'].max().date()}</p></div>", unsafe_allow_html=True)

# st.markdown("---")

# # ------------------------------------------------------
# # Real-Time Prediction (NOW FIXED)
# # ------------------------------------------------------
# st.header("🔍 Real-Time Crime Prediction")

# try:
#     model = models[selected_model]

#     # Use FULL 5-feature input like your original working script
#     X = pd.DataFrame([{
#         "day": selected_date.day,
#         "month": selected_date.month,
#         "year": selected_date.year,
#         "day_of_week": selected_date.weekday(),
#         "is_weekend": 1 if selected_date.weekday() in [5, 6] else 0
#     }])

#     pred = model.predict(X)[0]

#     st.subheader(f"Predicted Crime Count: **{int(pred):,}**")

# except Exception as e:
#     st.error(f"⚠️ Unable to predict. Model expected different feature inputs.\n{str(e)}")


# # ------------------------------------------------------
# # Crime Trend Chart
# # ------------------------------------------------------
# st.header("📈 Crime Trend Over Time")

# crime_by_date = df.groupby("date")[count_col].sum().reset_index()
# crime_by_date = crime_by_date.sort_values("date")

# st.line_chart(crime_by_date.set_index("date")[count_col])



#  # ------------------------------------------------------
# # # Model Performance Table
# # # ------------------------------------------------------
# st.header("📊 Model Performance")

# metrics_list = []

# feature_cols = ["day", "month", "year", "day_of_week", "is_weekend"]
# target_col = "crime_count"

# X_full = df[feature_cols].astype(float)
# y_full = df[target_col]

# for name, model in models.items():
#     if model is not None:
#         try:
#             y_pred = model.predict(X_full)
            
#             r2 = r2_score(y_full, y_pred)
#             mae = mean_absolute_error(y_full, y_pred)
#             rmse = np.sqrt(mean_squared_error(y_full, y_pred))

#             metrics_list.append({
#                 "Model": name,
#                 "R² Score": round(r2, 3),
#                 "MAE": round(mae, 2),
#                 "RMSE": round(rmse, 2)
#             })

#         except Exception as e:
#             metrics_list.append({
#                 "Model": name,
#                 "R² Score": f"Error ({e})",
#                 "MAE": f"Error ({e})",
#                 "RMSE": f"Error ({e})"
#             })

# perf_df = pd.DataFrame(metrics_list)
# st.table(perf_df)






#2
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import joblib
# from datetime import datetime
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# # ======================================================
# # Page Configuration
# # ======================================================
# st.set_page_config(
#     page_title="Crime Prediction Dashboard",
#     page_icon="🚔",
#     layout="wide"
# )

# # ======================================================
# # Custom Styling
# # ======================================================
# st.markdown("""
# <style>
# :root { --bg1: #00a0ef; --bg2: #011753; }

# .stApp {
#     background: linear-gradient(180deg, var(--bg1), var(--bg2));
#     color: #f5f5f5;
# }
# .kpi {
#     background: rgba(255,255,255,0.08);
#     padding: 18px;
#     border-radius: 14px;
#     border: 1px solid rgba(255,255,255,0.2);
#     text-align: center;
# }
# .kpi h3 { margin: 0; color: #ffffff; }
# .kpi p { margin: 0; font-size: 22px; color: #e8f4ff; }
# .big-title { font-size: 40px; font-weight: 700; color: #ffffff; }
# </style>
# """, unsafe_allow_html=True)

# # ======================================================
# # Load Dataset
# # ======================================================
# DATA_PATH = "ml_code/models/daily_features.csv"

# try:
#     df = pd.read_csv(DATA_PATH)
# except FileNotFoundError:
#     st.error("❌ Required dataset (daily_features.csv) not found.")
#     st.stop()

# if "date" not in df.columns:
#     st.error("❌ Dataset must contain a 'date' column.")
#     st.stop()

# df["date"] = pd.to_datetime(df["date"], errors="coerce")

# # Detect target column
# target_col = None
# for col in ["crime_count", "count", "daily_count", "value"]:
#     if col in df.columns:
#         target_col = col
#         break

# if target_col is None:
#     st.error("❌ No valid crime count column found.")
#     st.stop()

# df[target_col] = df[target_col].fillna(0).astype(int)

# # ======================================================
# # Load Trained Models
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
# # Sidebar Controls
# # ======================================================
# st.sidebar.title("⚙️ Configuration")

# selected_model = st.sidebar.selectbox(
#     "Select Prediction Model",
#     available_models
# )

# selected_date = st.sidebar.date_input(
#     "Select Prediction Date",
#     value=df["date"].max().date()
# )

# # ======================================================
# # Header & Project Overview
# # ======================================================
# st.markdown("<div class='big-title'>🚔 Crime Prediction Dashboard</div>", unsafe_allow_html=True)

# st.markdown("""
# ### 📌 Project Overview
# This dashboard demonstrates a **machine learning–based crime prediction system**
# developed using historical crime data.

# The application integrates:
# - Large-scale data engineering (Hadoop & Spark)
# - Feature engineering and regression modeling
# - Real-time inference through an interactive dashboard
# """)

# # ======================================================
# # KPI Section
# # ======================================================
# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.markdown(
#         f"<div class='kpi'><h3>📊 Total Records</h3><p>{len(df):,}</p></div>",
#         unsafe_allow_html=True
#     )

# with col2:
#     st.markdown(
#         "<div class='kpi'><h3>🏙️ Cities Covered</h3><p>2</p></div>",
#         unsafe_allow_html=True
#     )

# with col3:
#     st.markdown(
#         f"<div class='kpi'><h3>🤖 Models Available</h3><p>{len(available_models)}</p></div>",
#         unsafe_allow_html=True
#     )

# with col4:
#     st.markdown(
#         f"<div class='kpi'><h3>📅 Date Range</h3><p>{df['date'].min().date()} → {df['date'].max().date()}</p></div>",
#         unsafe_allow_html=True
#     )

# st.markdown("---")

# # ======================================================
# # Real-Time Prediction
# # ======================================================
# st.header("🔍 Real-Time Crime Prediction")

# st.markdown("""
# Predictions are generated using **temporal features** extracted from the selected date:
# - Day of month
# - Month
# - Year
# - Day of week
# - Weekend indicator
# """)

# try:
#     model = models[selected_model]

#     input_features = pd.DataFrame([{
#         "day": selected_date.day,
#         "month": selected_date.month,
#         "year": selected_date.year,
#         "day_of_week": selected_date.weekday(),
#         "is_weekend": 1 if selected_date.weekday() >= 5 else 0
#     }])

#     prediction = model.predict(input_features)[0]

#     st.subheader(f"📈 Predicted Crime Count: **{int(prediction):,}**")

#     st.caption(
#         "This value represents the expected number of crimes for the selected date "
#         "based on historical temporal patterns."
#     )

# except Exception as e:
#     st.error(f"Prediction failed: {e}")

# # ======================================================
# # Crime Trend Visualization
# # ======================================================
# st.header("📈 Historical Crime Trends")

# st.markdown("""
# The time-series chart below illustrates historical crime trends,
# highlighting seasonality and long-term patterns.
# """)

# crime_ts = df.groupby("date")[target_col].sum().reset_index()
# crime_ts = crime_ts.sort_values("date")

# st.line_chart(crime_ts.set_index("date")[target_col])

# # ======================================================
# # Model Performance Evaluation
# # ======================================================
# st.header("📊 Model Performance Comparison")

# st.markdown("""
# All models were evaluated using identical features and the same dataset
# to ensure a fair comparison.
# """)

# metrics = []
# feature_cols = ["day", "month", "year", "day_of_week", "is_weekend"]

# X = df[feature_cols].astype(float)
# y = df[target_col]

# for name, model in models.items():
#     if model is not None:
#         try:
#             preds = model.predict(X)
#             metrics.append({
#                 "Model": name,
#                 "R² Score": round(r2_score(y, preds), 3),
#                 "MAE": round(mean_absolute_error(y, preds), 2),
#                 "RMSE": round(np.sqrt(mean_squared_error(y, preds)), 2)
#             })
#         except Exception as e:
#             metrics.append({
#                 "Model": name,
#                 "R² Score": "Error",
#                 "MAE": "Error",
#                 "RMSE": "Error"
#             })

# perf_df = pd.DataFrame(metrics)
# st.table(perf_df)

# # ======================================================
# # Model Selection Summary
# # ======================================================
# st.markdown("""
# ### 🏆 Final Model Selection

# - **Random Forest** demonstrated the highest predictive accuracy
# - Achieved the strongest R² score with the lowest error metrics
# - Effectively captures non-linear temporal patterns in crime data

# 📌 **Random Forest is selected as the final production model.**
# """)

# st.markdown("---")
# st.caption(
#     "Developed as part of a Big Data Analytics project. "
#     "Predictions are based on historical data and should be used for decision support only."
# )





import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
# Load Trained Models
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
# ------------------- Sidebar -------------------------
# ======================================================
with st.sidebar:
    st.markdown("## 🚔 Crime Prediction System")
    st.markdown(
        "Predict daily crime counts using temporal features and compare ML models."
    )
    st.markdown("---")

    # Navigation
    st.markdown("### 📍 Navigate")
    page = st.radio(
        label="Choose section",
        options=["Overview", "Prediction"],
        index=0
    )

    st.markdown("---")

    # Prediction settings
    st.markdown("### ⚙️ Prediction Settings")
    selected_model = st.selectbox("🔹 Choose model", available_models)
    selected_date = st.date_input("📅 Select prediction date", value=df["date"].max().date())

    st.markdown("---")

    # Advanced options
    with st.expander("🧠 Advanced Options", expanded=False):
        st.markdown("Optional controls for advanced users.")
        show_feature_importance = st.checkbox("Show feature importance (RF only)", value=True)
        show_metrics_table = st.checkbox("Show full model metrics table", value=True)

    st.markdown("---")
    # Help
    st.markdown("### ❓ Help & Info")
    st.markdown(
        "Developed as part of a Big Data Analytics project.\n"
        "Predictions are academic and for analysis only.\n"
        "See README for details."
    )

# ======================================================
# ------------------- Overview Page -------------------
# ======================================================
if page == "Overview":
    st.title("🚔 Crime Prediction Dashboard")
    st.markdown("""
    This dashboard demonstrates a **machine learning–based crime prediction system**.
    It integrates:
    - Data preprocessing and feature engineering
    - Regression modeling for crime forecasting
    - Interactive visualization for historical trends and predictions
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
    crime_ts = crime_ts.sort_values("date")
    st.line_chart(crime_ts.set_index("date")[target_col])

    st.markdown("---")
    st.header("🗂 Dataset Preview (First 100 Rows)")
    st.markdown("Below is a preview of the dataset used for modeling. You can scroll horizontally to see all columns.")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Dataset Summary")
    st.markdown("Quick overview of dataset features and missing values:")
    summary_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": df.isna().sum(),
        "Unique Values": df.nunique()
    })
    st.table(summary_df)

    # Optional download button
    st.download_button(
        label="📥 Download first 100 rows as CSV",
        data=df.head(100).to_csv(index=False),
        file_name="crime_dataset_preview.csv",
        mime="text/csv"
    )

# ======================================================
# ------------------- Prediction Page -----------------
# ======================================================
if page == "Prediction":
    st.title("🔍 Real-Time Crime Prediction")

    # Input features for prediction
    input_features = pd.DataFrame([{
        "day": selected_date.day,
        "month": selected_date.month,
        "year": selected_date.year,
        "day_of_week": selected_date.weekday(),
        "is_weekend": 1 if selected_date.weekday() >= 5 else 0
    }])

    # Generate prediction
    try:
        model = models[selected_model]
        prediction = model.predict(input_features)[0]
        st.subheader(f"📈 Predicted Crime Count: **{int(prediction):,}**")
        st.caption(
            "This is the expected number of crimes for the selected date "
            "based on historical temporal patterns."
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # Model Performance Comparison
    st.header("📊 Model Performance Comparison")
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

    # Feature Importance (Random Forest)
    if show_feature_importance and "Random Forest" in models and models["Random Forest"] is not None:
        st.header("📌 Feature Importance (Random Forest)")
        try:
            importances = models["Random Forest"].feature_importances_
            fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values("Importance", ascending=False)
            st.bar_chart(fi_df.set_index("Feature"))
        except Exception as e:
            st.warning(f"Cannot display feature importance: {e}")

    # Final Model Justification
    st.markdown("""
    ### 🏆 Final Model Selection
    - **Random Forest** demonstrated the highest predictive accuracy
    - Captures non-linear temporal patterns effectively
    - Selected as the final production model
    """)

    st.markdown("---")
    st.caption(
        "Developed for Big Data Analytics. Predictions are based on historical data and intended for academic use only."
    )
