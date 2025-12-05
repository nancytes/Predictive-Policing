from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

spark = SparkSession.builder.appName("NYC_Shell_ETL1").getOrCreate()

########################################
# 1) Read raw NYC data
########################################
input_path = "hdfs://localhost:9000/user/nan25/raw/nypd_raw_data.csv"

df = spark.read.option("header", True).csv(input_path)

########################################
# 2) Select the 4 relevant columns
########################################
df_clean = df.select(
    col(df.columns[0]).alias("id"),
    col(df.columns[1]).alias("date"),
    col(df.columns[4]).alias("typeId"),
    col(df.columns[5]).alias("typeDesc")
)

df_clean.printSchema()
df_clean.show()

########################################
# 3) Convert date column
########################################
df_clean = df_clean.withColumn("date", to_date(col("date"), "MM/dd/yyyy"))

df_clean.printSchema()
df_clean.show()

########################################
# 4) Remove NA rows
########################################
df_clean = df_clean.na.drop()

count_rows = df_clean.count()
print(f"Total cleaned NYC rows: {count_rows}")

########################################
# 5) Save directly to HDFS
########################################
output_path = "hdfs://localhost:9000/user/nan25/processed/nypd_type_date.csv"

df_clean.coalesce(1).write \
    .option("header", True) \
    .mode("overwrite") \
    .csv(output_path)
