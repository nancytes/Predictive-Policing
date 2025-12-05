from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("IngestChicago").getOrCreate()

df = spark.read.csv(
"hdfs://localhost:9000/user/nan25/raw/Chi_Crimes_2001_to_Present.csv",
header=True, inferSchema=True
)

df.write.mode("overwrite").parquet("hdfs://localhost:9000/user/nan25/processed/chicago_raw.parquet")
