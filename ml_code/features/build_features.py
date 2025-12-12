import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col
import os

# -------------------------------
# 1. Spark session
# -------------------------------
spark = SparkSession.builder.appName("Build_Features").getOrCreate()

# -------------------------------
# 2. Load merged dataset from HDFS
# -------------------------------
MERGED_HDFS_PATH = "hdfs://localhost:9000/user/nan25/processed/merged_city_crimes.csv"

df = spark.read.option("header", True).csv(MERGED_HDFS_PATH)

# -------------------------------
# 3. Basic Cleaning
# -------------------------------
df = df.withColumn("date", to_date(col("date_parsed")))
df = df.dropna(subset=["date"])

# -------------------------------
# 4. Aggregate to daily crime counts
# -------------------------------
daily = (
    df.groupBy("date")
      .count()
      .withColumnRenamed("count", "crime_count")
      .sort("date")
)

# Convert to Pandas for ML
daily_pd = daily.toPandas()

# -------------------------------
# 5. Fix datetime conversion
# -------------------------------
daily_pd["date"] = pd.to_datetime(daily_pd["date"], errors="coerce")
daily_pd = daily_pd.dropna(subset=["date"])

# -------------------------------
# 6. Feature Engineering
# -------------------------------
daily_pd["day"] = daily_pd["date"].dt.day
daily_pd["month"] = daily_pd["date"].dt.month
daily_pd["year"] = daily_pd["date"].dt.year
daily_pd["day_of_week"] = daily_pd["date"].dt.dayofweek
daily_pd["is_weekend"] = daily_pd["day_of_week"].isin([5, 6]).astype(int)

# -------------------------------
# 7. Save to local dataset
# -------------------------------
OUTPUT_FILE = "ml_code/models/daily_features.csv"
os.makedirs("ml_code/models", exist_ok=True)

daily_pd.to_csv(OUTPUT_FILE, index=False)

print(f"[SUCCESS] Features saved to {OUTPUT_FILE}")
spark.stop()







