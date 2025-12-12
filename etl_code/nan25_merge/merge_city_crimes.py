from pyspark.sql import SparkSession
from pyspark.sql.functions import expr

# Create Spark session
spark = SparkSession.builder.appName("Merge_Chicago_NYC").getOrCreate()

# Read Chicago CSV
chicago_df = spark.read.option("header", True).csv("hdfs://localhost:9000/user/nan25/processed/chicago_cleaned.csv")

# Read NYC CSV
nyc_df = spark.read.option("header", True).csv("hdfs://localhost:9000/user/nan25/processed/nyc_cleaned.csv")

# Convert date columns safely using try_to_timestamp
chicago_df = chicago_df.withColumn(
    "date_parsed",
    expr("try_to_timestamp(date, 'MM/dd/yyyy hh:mm:ss a')")
)

nyc_df = nyc_df.withColumn(
    "date_parsed",
    expr("try_to_timestamp(date, 'MM/dd/yyyy hh:mm:ss a')")
)

# Merge dataframes
merged_df = chicago_df.unionByName(nyc_df)

# Save merged CSV
merged_df.write.option("header", True).mode("overwrite").csv("hdfs://localhost:9000/user/nan25/processed/merged_city_crimes.csv")

# Stop Spark session
spark.stop()