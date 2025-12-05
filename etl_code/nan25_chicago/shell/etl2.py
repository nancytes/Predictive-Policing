from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

spark = SparkSession.builder.appName("Chicago_Shell_ETL2").getOrCreate()

#######################################
# 1) Load cleaned Chicago data
#######################################
df_clean = spark.read.option("header", True).csv(
    "hdfs://localhost:9000/user/nan25/processed/crime_type_date.csv"
)

#######################################
# 2) CRIME OCCURRENCE PER DAY
#######################################
crime_occ = (
    df_clean.groupBy("Date")
            .agg(count("*").alias("Count"))
            .orderBy("Date")
)

crime_occ.show()

crime_occ.write.coalesce(1).option("header", True).mode("overwrite").csv(
    "hdfs://localhost:9000/user/nan25/processed/crime_occurrence_per_day"
)

#######################################
# 3) THEFT OCCURRENCE PER DAY
#######################################
df_theft = df_clean.filter(col("Primary Type") == "THEFT")

df_theft.write.coalesce(1).option("header", True).mode("overwrite").csv(
    "hdfs://localhost:9000/user/nan25/processed/theft_occurrence_per_day"
)

#######################################
# 4) BATTERY OCCURRENCE PER DAY
#######################################
df_battery = df_clean.filter(col("Primary Type") == "BATTERY")

df_battery.write.coalesce(1).option("header", True).mode("overwrite").csv(
    "hdfs://localhost:9000/user/nan25/processed/battery_occurrence_per_day"
)
