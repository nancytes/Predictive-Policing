from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("IngestNYC").getOrCreate()

df = spark.read.csv(
"hdfs://localhost:9000/user/nan25/raw/nypd_raw_data.csv",
header=True, inferSchema=True
)

df.write.mode("overwrite").parquet("hdfs://localhost:9000/user/nan25/processed/nyc_raw.parquet")
