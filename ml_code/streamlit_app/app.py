
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------------------------------------
# Page Config
# ------------------------------------------------------
st.set_page_config(
    page_title="Crime Prediction Dashboard",
    page_icon="🚔",
    layout="wide"
)

# ------------------------------------------------------
# Styling – Red → Black gradient
# ------------------------------------------------------
st.markdown("""
<style>
:root { --bg1:  #00a0ef; --bg2: #011753; }

.reportview-container {
    background: linear-gradient(180deg, var(--bg1), var(--bg2));
    color: #f5f5f5;
}
.stApp {
    background: linear-gradient(180deg, var(--bg1), var(--bg2));
}
.kpi {
    background: rgba(255,255,255,0.07);
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.2);
    text-align: center;
}
.kpi h3 { margin: 0; color: #fff; }
.kpi p { margin: 0; font-size: 22px; color: #ffdfdf; }
.big-title { font-size: 38px; font-weight: 700; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# Load Data
# ------------------------------------------------------
DATA_PATH = "ml_code/models/daily_features.csv"

try:
    df = pd.read_csv(DATA_PATH)
except:
    st.error("❌ daily_features.csv not found.")
    st.stop()

# Ensure date column exists
if "date" not in df.columns:
    st.error("❌ No 'date' column found in the data.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Locate count column (or build one)
count_col = None
for c in ["count", "crime_count", "daily_count", "value"]:
    if c in df.columns:
        count_col = c
        break

if count_col is None:
    df["count"] = 1
    count_col = "count"

df[count_col] = df[count_col].fillna(0).astype(int)

# ------------------------------------------------------
# Load Models
# ------------------------------------------------------
MODEL_DIR = "ml_code/models"
model_files = {
    "Random Forest": f"{MODEL_DIR}/RandomForest.pkl",
    "Gradient Boosting": f"{MODEL_DIR}/GradientBoosting.pkl",
    "Linear Regression": f"{MODEL_DIR}/LinearRegression.pkl",
    "Final Model": f"{MODEL_DIR}/final_model.pkl",
}

# models = {}
# for name, path in model_files.items():
#     try:
#         with open(path, "rb") as f:
#             models[name] = pickle.load(f)
#     except:
#         models[name] = None

# available_models = [m for m in models if models[m] is not None]
models = {}

for name, path in model_files.items():
    try:
        # joblib first (handles pipelines)
        try:
            models[name] = joblib.load(path)
        except:
            # fallback to pickle
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    except Exception as e:
        models[name] = None
        st.sidebar.error(f"❌ Failed to load {name}: {e}")

# Create list of successfully loaded models
available_models = [m for m in models if models[m] is not None]
# ------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------
st.sidebar.title("⚙️ Controls")

selected_model = st.sidebar.selectbox("Select Model", available_models)

selected_date = st.sidebar.date_input(
    "Select Date",
    value=df["date"].max().date()
)

# ------------------------------------------------------
# Header + KPIs
# ------------------------------------------------------
st.markdown("<div class='big-title'>🚔 Crime Prediction Dashboard</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='kpi'><h3>📊 Total Records</h3><p>{:,}</p></div>".format(len(df)), unsafe_allow_html=True)

with col2:
    st.markdown("<div class='kpi'><h3>🏙️ Cities Covered</h3><p>2</p></div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='kpi'><h3>🤖 ML Models</h3><p>{}</p></div>".format(len(available_models)), unsafe_allow_html=True)

with col4:
    st.markdown(f"<div class='kpi'><h3>📅 Date Range</h3><p>{df['date'].min().date()} → {df['date'].max().date()}</p></div>", unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------
# Real-Time Prediction (NOW FIXED)
# ------------------------------------------------------
st.header("🔍 Real-Time Crime Prediction")

try:
    model = models[selected_model]

    # Use FULL 5-feature input like your original working script
    X = pd.DataFrame([{
        "day": selected_date.day,
        "month": selected_date.month,
        "year": selected_date.year,
        "day_of_week": selected_date.weekday(),
        "is_weekend": 1 if selected_date.weekday() in [5, 6] else 0
    }])

    pred = model.predict(X)[0]

    st.subheader(f"Predicted Crime Count: **{int(pred):,}**")

except Exception as e:
    st.error(f"⚠️ Unable to predict. Model expected different feature inputs.\n{str(e)}")


# ------------------------------------------------------
# Crime Trend Chart
# ------------------------------------------------------
st.header("📈 Crime Trend Over Time")

crime_by_date = df.groupby("date")[count_col].sum().reset_index()
crime_by_date = crime_by_date.sort_values("date")

st.line_chart(crime_by_date.set_index("date")[count_col])



 # ------------------------------------------------------
# # Model Performance Table
# # ------------------------------------------------------
st.header("📊 Model Performance")

metrics_list = []

feature_cols = ["day", "month", "year", "day_of_week", "is_weekend"]
target_col = "crime_count"

X_full = df[feature_cols].astype(float)
y_full = df[target_col]

for name, model in models.items():
    if model is not None:
        try:
            y_pred = model.predict(X_full)
            
            r2 = r2_score(y_full, y_pred)
            mae = mean_absolute_error(y_full, y_pred)
            rmse = np.sqrt(mean_squared_error(y_full, y_pred))

            metrics_list.append({
                "Model": name,
                "R² Score": round(r2, 3),
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2)
            })

        except Exception as e:
            metrics_list.append({
                "Model": name,
                "R² Score": f"Error ({e})",
                "MAE": f"Error ({e})",
                "RMSE": f"Error ({e})"
            })

perf_df = pd.DataFrame(metrics_list)
st.table(perf_df)


