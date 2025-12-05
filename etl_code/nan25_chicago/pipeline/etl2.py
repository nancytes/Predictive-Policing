from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
import subprocess

spark = SparkSession.builder.appName("Chicago_ETL2").getOrCreate()

# 1) READ INPUT
input_path = "hdfs://localhost:9000/user/nan25/processed/crime_type_date.csv"

df = spark.read.option("header", True).csv(input_path)

# Utility save function
def save_single_csv(df, temp_dir, final_path):
    df.coalesce(1).write.option("header", True).mode("overwrite").csv(temp_dir)

    files = subprocess.check_output(["hdfs", "dfs", "-ls", temp_dir]).decode()
    part = [f.split()[-1] for f in files.split("\n") if "part-" in f][0]

    subprocess.call(["hdfs", "dfs", "-mv",
                     f"{temp_dir}/{part}",
                     final_path])
    subprocess.call(["hdfs", "dfs", "-rm", "-r", temp_dir])

# 2) CRIME (ALL)
crime_occ = (
    df.groupBy("Date")
      .agg(count("*").alias("Count"))
      .orderBy("Date")
)

save_single_csv(crime_occ,
                "temp_chicago_crime",
                "/user/nan25/processed/crime_occurrence_per_day.csv")

# 3) THEFT
df_theft = df.filter(col("Primary Type") == "THEFT")

theft_occ = (
    df_theft.groupBy("Date")
            .agg(count("*").alias("Count"))
            .orderBy("Date")
)

save_single_csv(theft_occ,
                "temp_chicago_theft",
                "/user/nan25/processed/theft_occurrence_per_day.csv")

# 4) BATTERY
df_battery = df.filter(col("Primary Type") == "BATTERY")

battery_occ = (
    df_battery.groupBy("Date")
              .agg(count("*").alias("Count"))
              .orderBy("Date")
)

save_single_csv(battery_occ,
                "temp_chicago_battery",
                "/user/nan25/processed/battery_occurrence_per_day.csv")
