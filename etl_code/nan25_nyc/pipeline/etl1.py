from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
import subprocess

spark = SparkSession.builder.appName("NYC_ETL1").getOrCreate()

############################
# 1. LOAD RAW NYC DATA
############################
input_path = "hdfs://localhost:9000/user/nan25/raw/nypd_raw_data.csv"

df = spark.read.option("header", True).csv(input_path)

###################################
# 2. SELECT ONLY NEEDED COLUMNS
###################################
# Equivalent to Scala row.getString(0,1,4,5)
df_clean = df.select(
    col(df.columns[0]).alias("id"),
    col(df.columns[1]).alias("date"),
    col(df.columns[4]).alias("typeId"),
    col(df.columns[5]).alias("typeDesc")
)

###################################
# 3. PARSE DATE COLUMN
###################################
df_clean = df_clean.withColumn("date", to_date(col("date"), "MM/dd/yyyy"))

###################################
# 4. DROP NULLS
###################################
df_clean = df_clean.na.drop()

###################################
# 5. SAVE USING TEMP → RENAME
###################################
temp_path = "temp_nyc_etl1"
final_path = "/user/nan25/processed/nypd_type_date.csv"

df_clean.coalesce(1).write.option("header", "true") \
    .mode("overwrite").csv(temp_path)

files = subprocess.check_output(["hdfs","dfs","-ls",temp_path]).decode()
part_file = [f.split()[-1] for f in files.split("\n") if "part-" in f][0]

subprocess.call(["hdfs","dfs","-mv",
                 f"{temp_path}/{part_file}",
                 final_path])

subprocess.call(["hdfs","dfs","-rm","-r",temp_path])
