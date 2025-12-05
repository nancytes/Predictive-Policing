from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

spark = SparkSession.builder.appName("NYC_Shell_ETL2").getOrCreate()

#######################################
# 1) Load cleaned NYC data
#######################################
df_clean = spark.read.option("header", True).csv(
    "hdfs://localhost:9000/user/nan25/processed/nypd_type_date.csv"
)

#######################################
# 2) ALL NYC CRIME OCCURRENCE PER DAY
#######################################
occ_all = (
    df_clean.groupBy("date")
            .agg(count("*").alias("Count"))
            .orderBy("date")
)

occ_all.write.coalesce(1).option("header", True).mode("overwrite").csv(
    "hdfs://localhost:9000/user/nan25/processed/nypd_all"
)

#######################################
# 3) ASSAULT
#######################################
df_assault = df_clean.filter(col("typeDesc").rlike(".*ASSAULT.*"))

df_assault.write.coalesce(1).option("header", True).mode("overwrite").csv(
    "hdfs://localhost:9000/user/nan25/processed/nypd_assault"
)

#######################################
# 4) LARCENY
#######################################
df_larceny = df_clean.filter(col("typeDesc").rlike(".*LARCENY.*"))

df_larceny.write.coalesce(1).option("header", True).mode("overwrite").csv(
    "hdfs://localhost:9000/user/nan25/processed/nypd_larceny"
)
