import pandas as pd
import numpy as np
import subprocess
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

HDFS_PATH = "/user/nan25/processed/daily_features.csv"
LOCAL_FILE = "daily_features.csv"

def hdfs_get():
    cmd = ["hdfs", "dfs", "-get", HDFS_PATH, LOCAL_FILE]
    try:
        subprocess.check_call(cmd)
        print("✓ Downloaded daily_features.csv to local.")
    except:
        print("Already downloaded or HDFS not accessible.")

# -------------------------------
# 1. Get file from HDFS
# -------------------------------
hdfs_get()

# -------------------------------
# 2. Load features
# -------------------------------
df = pd.read_csv(LOCAL_FILE)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Feature engineering
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

df["lag1"] = df["count"].shift(1)
df["lag7"] = df["count"].shift(7)
df["lag30"] = df["count"].shift(30)

df["rolling7"] = df["count"].rolling(7).mean()
df["rolling30"] = df["count"].rolling(30).mean()

df = df.dropna().reset_index(drop=True)

X = df[["day_of_week","month","week_of_year","lag1","lag7","lag30","rolling7","rolling30"]]
y = df["count"]

# Time-based split
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------
# 3. Train models
# -------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

os.makedirs("models", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\n{name}")
    print("MAE:", mae)
    print("RMSE:", rmse)

    joblib.dump(model, f"models/{name}.pkl")

print("\n Models saved in 'models/' directory.")
