from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, to_date
from py4j.java_gateway import java_import

# ---------------------------------------------------------
# CREATE SPARK SESSION
# ---------------------------------------------------------
spark = SparkSession.builder.appName("Chicago_Cleaning_Windows").getOrCreate()
sc = spark.sparkContext

# Import Hadoop FileSystem classes
java_import(sc._jvm, 'org.apache.hadoop.fs.FileSystem')
java_import(sc._jvm, 'org.apache.hadoop.fs.Path')

fs = sc._jvm.FileSystem.get(sc._jsc.hadoopConfiguration())

# ---------------------------------------------------------
# 1) LOAD RAW DATA FROM HDFS
# ---------------------------------------------------------
raw_path = "hdfs://localhost:9000/user/nan25/raw/Chi_Crimes_2001_to_Present.csv"

df_raw = spark.read.option("header", True).csv(raw_path)

# ---------------------------------------------------------
# 2) SELECT NEEDED COLUMNS
# ---------------------------------------------------------
df_cleaned = df_raw.select(
    col("ID").alias("id"),
    col("Date").alias("date"),
    col("Primary Type").alias("type"),
    col("Description").alias("desc")
)

# ---------------------------------------------------------
# 3) DROP NULL VALUES
# ---------------------------------------------------------
df_cleaned = df_cleaned.na.drop("any")

# ---------------------------------------------------------
# 4) SAVE CLEANED RAW FILE (NO SUBPROCESS)
# ---------------------------------------------------------
temp_path = "hdfs://localhost:9000/user/nan25/processed/chicago_cleaned_temp"
final_path = "/user/nan25/processed/chicago_cleaned.csv"

df_cleaned.coalesce(1).write.format("csv") \
    .mode("overwrite").option("header", True).save(temp_path)

# Find Spark part file
status = fs.listStatus(sc._jvm.Path(temp_path))
part_file = None

for file in status:
    name = file.getPath().getName()
    if name.startswith("part-"):
        part_file = name
        break

# Move part file to final .csv name
fs.rename(
    sc._jvm.Path(f"{temp_path}/{part_file}"),
    sc._jvm.Path(final_path)
)

# Delete temp directory
fs.delete(sc._jvm.Path(temp_path), True)

# ---------------------------------------------------------
# 5) LOAD CLEANED CSV BACK FROM HDFS
# ---------------------------------------------------------
df = spark.read.option("header", True).csv("hdfs://localhost:9000" + final_path)

# FIX DATE — FULL DATETIME TO DATE ONLY
df = df.withColumn("id", col("id").cast("int"))
df = df.withColumn("date", to_timestamp(col("date"), "MM/dd/yyyy hh:mm:ss a"))
df = df.withColumn("date", to_date(col("date")))

# ---------------------------------------------------------
# 6) CREATE DAILY GROUPS
# ---------------------------------------------------------
df_battery_day = (
    df.filter(col("type") == "BATTERY")
      .groupBy("date").count().sort("date")
      .coalesce(1)
)

df_theft_day = (
    df.filter(col("type") == "THEFT")
      .groupBy("date").count().sort("date")
      .coalesce(1)
)

df_crime_day = (
    df.groupBy("date").count().sort("date")
      .coalesce(1)
)

# ---------------------------------------------------------
# 7) SAVE FUNCTION (HDFS)
# ---------------------------------------------------------
def save_csv(df, temp_dir, final_output):
    temp_hdfs = f"hdfs://localhost:9000/user/nan25/data/{temp_dir}"

    df.write.format("csv").mode("overwrite").option("header", True).save(temp_hdfs)

    status = fs.listStatus(sc._jvm.Path(temp_hdfs))
    part = None

    for f in status:
        n = f.getPath().getName()
        if n.startswith("part-"):
            part = n
            break

    fs.rename(
        sc._jvm.Path(f"{temp_hdfs}/{part}"),
        sc._jvm.Path(final_output)
    )

    fs.delete(sc._jvm.Path(temp_hdfs), True)

# ---------------------------------------------------------
# 8) SAVE FINAL DATASETS TO HDFS
# ---------------------------------------------------------
save_csv(df_battery_day,
         "temp_battery",
         "/user/nan25/data/battery_occurrence_per_day.csv")

save_csv(df_crime_day,
         "temp_crime",
         "/user/nan25/data/crime_occurrence_per_day.csv")

save_csv(df_theft_day,
         "temp_theft",
         "/user/nan25/data/theft_occurrence_per_day.csv")
