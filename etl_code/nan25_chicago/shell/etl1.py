from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

spark = SparkSession.builder.appName("Chicago_Shell_ETL1").getOrCreate()

########################################
# 1) Read raw Chicago data
########################################
input_path = "hdfs://localhost:9000/user/nan25/raw/Chi_Crimes_2001_to_Present.csv"

df = spark.read.option("header", True).option("inferSchema", True).csv(input_path)

########################################
# 2) Drop unnecessary columns
########################################
cols_to_drop = [
    "ID", "Case Number", "Block", "IUCR", "Location Description",
    "Arrest", "Domestic", "Beat", "District", "Ward", "Community Area",
    "FBI Code", "X Coordinate", "Y Coordinate", "Updated On",
    "Latitude", "Longitude", "Location"
]

df_clean = df.drop(*cols_to_drop)

df_clean.printSchema()
df_clean.show()

########################################
# 3) Convert date column
########################################
format_str = "MM/dd/yyyy hh:mm:ss a"
df_clean = df_clean.withColumn("Date", to_date(col("Date"), format_str))

df_clean.printSchema()
df_clean.show()

########################################
# 4) Drop NA rows
########################################
df_clean = df_clean.na.drop()

count_rows = df_clean.count()
print(f"Total cleaned rows: {count_rows}")

########################################
# 5) Save directly to HDFS
########################################
output_path = "hdfs://localhost:9000/user/nan25/processed/crime_type_date.csv"

df_clean.coalesce(1).write \
    .option("header", True) \
    .mode("overwrite") \
    .csv(output_path)
