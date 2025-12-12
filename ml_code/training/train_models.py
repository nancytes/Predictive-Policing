import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# -------------------------
# Load features
# -------------------------
DATA_FILE = "ml_code/models/daily_features.csv"
df = pd.read_csv(DATA_FILE)

# -------------------------
# Prepare data
# -------------------------
X = df[["day", "month", "year", "day_of_week", "is_weekend"]]
y = df["crime_count"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# -------------------------
# Select ML Algorithms
# -------------------------
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "LinearRegression": LinearRegression()
}

os.makedirs("ml_code/models", exist_ok=True)

# -------------------------
# Train & Save Models
# -------------------------
for name, model in models.items():
    model.fit(X_train, y_train)

    with open(f"ml_code/models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"[SAVED] {name}.pkl")

print("[DONE] All models trained successfully.")
