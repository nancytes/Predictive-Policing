import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# ------------------------------------------------------
# Load processed dataset
# ------------------------------------------------------
DATA_PATH = "ml_code/models/daily_features.csv"

df = pd.read_csv(DATA_PATH)

# Ensure date is parsed correctly (if needed)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# ------------------------------------------------------
# Feature + target selection
# ------------------------------------------------------
required_features = ["day", "month", "year", "day_of_week", "is_weekend"]

# Validate columns
missing = [col for col in required_features if col not in df.columns]
if missing:
    raise ValueError(f"❌ Missing required columns: {missing}")

X = df[required_features]
y = df["crime_count"]

# ------------------------------------------------------
# Build machine learning pipeline
# ------------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# ------------------------------------------------------
# Train model
# ------------------------------------------------------
pipeline.fit(X, y)

# ------------------------------------------------------
# Save final model (joblib)
# ------------------------------------------------------
OUTPUT_PATH = "ml_code/models/final_model.pkl"
joblib.dump(pipeline, OUTPUT_PATH)

print("✅ final_model.pkl created successfully!")
print(f"✔ Saved to: {OUTPUT_PATH}")
