from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
import subprocess

spark = SparkSession.builder.appName("Chicago_ETL1").getOrCreate()

############################
# 1. LOAD RAW CHICAGO DATA
############################
input_path = "hdfs://localhost:9000/user/nan25/raw/Chi_Crimes_2001_to_Present.csv"

options = {
    "header": "true",
    "inferSchema": "true",
    "delimiter": ","
}

df = spark.read.options(**options).csv(input_path)

###################################
# 2. DROP UNNEEDED COLUMNS
###################################
cols_to_drop = [
    "ID","Case Number","Block","IUCR","Location Description","Arrest","Domestic",
    "Beat","District","Ward","Community Area","FBI Code","X Coordinate",
    "Y Coordinate","Updated On","Latitude","Longitude","Location"
]

df_clean = df.drop(*cols_to_drop)

###################################
# 3. PARSE DATE COLUMN
###################################
format_str = "MM/dd/yyyy hh:mm:ss a"
df_clean = df_clean.withColumn("Date", to_date(col("Date"), format_str))

###################################
# 4. DROP ROWS WITH ANY NULLS
###################################
df_clean = df_clean.na.drop()

###################################
# 5. SAVE USING TEMP → RENAME
###################################
temp_path = "temp_chicago_etl1"
final_path = "/user/nan25/processed/crime_type_date.csv"

df_clean.coalesce(1).write.option("header", "true") \
    .mode("overwrite").csv(temp_path)

# find part-file and rename
files = subprocess.check_output(["hdfs","dfs","-ls",temp_path]).decode()
part_file = [f.split()[-1] for f in files.split("\n") if "part-" in f][0]

subprocess.call(["hdfs","dfs","-mv",
                 f"{temp_path}/{part_file}",
                 final_path])

subprocess.call(["hdfs","dfs","-rm","-r",temp_path])
