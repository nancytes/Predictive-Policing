from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
from py4j.java_gateway import java_import

# -------------------------------
# 1. Start Spark
# -------------------------------
spark = SparkSession.builder.appName("Daily_Features").getOrCreate()
sc = spark.sparkContext

# Hadoop FS imports
java_import(sc._jvm, 'org.apache.hadoop.fs.FileSystem')
java_import(sc._jvm, 'org.apache.hadoop.fs.Path')
fs = sc._jvm.FileSystem.get(sc._jsc.hadoopConfiguration())

HDFS = "hdfs://localhost:9000"

# -------------------------------
# 2. Load merged dataset
# -------------------------------
merged_path = f"{HDFS}/user/nan25/processed/merged_city_crimes.csv"
df = spark.read.option("header", True).csv(merged_path)

# -------------------------------
# 3. Parse date column
# -------------------------------
# Your merge script created `date_parsed`
df = df.withColumn("date_final", to_date(col("date_parsed")))

# Drop nulls (invalid dates)
df = df.filter(col("date_final").isNotNull())

# -------------------------------
# 4. Group to daily total crime counts
# -------------------------------
daily = (
    df.groupBy("date_final")
      .count()
      .orderBy("date_final")
      .withColumnRenamed("date_final", "date")
)

# -------------------------------
# 5. Save daily features
# -------------------------------
temp_path = "/user/nan25/processed/daily_features_temp"
final_path = "/user/nan25/processed/daily_features.csv"

daily.write.format("csv") \
     .mode("overwrite") \
     .option("header", True) \
     .save(HDFS + temp_path)

# Find part file
status = fs.listStatus(sc._jvm.Path(HDFS + temp_path))
part_file = None
for f in status:
    name = f.getPath().getName()
    if name.startswith("part-"):
        part_file = name
        break

# Move to final output CSV
fs.rename(
    sc._jvm.Path(f"{HDFS}{temp_path}/{part_file}"),
    sc._jvm.Path(final_path)
)

# Delete temp directory
fs.delete(sc._jvm.Path(HDFS + temp_path), True)

print("✓ Daily features saved to:", final_path)
spark.stop()
