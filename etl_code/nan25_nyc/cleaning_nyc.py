from pyspark.sql import SparkSession 
from pyspark.sql.functions import col, to_date, upper 
from py4j.java_gateway import java_import

# ———————————————————

# CREATE SPARK SESSION

# ———————————————————

spark =SparkSession.builder.appName("NYC_Cleaning_Windows").getOrCreate() 
sc =spark.sparkContext

# Import Hadoop FileSystem classes

java_import(sc._jvm, 'org.apache.hadoop.fs.FileSystem')
java_import(sc._jvm, 'org.apache.hadoop.fs.Path')

fs = sc._jvm.FileSystem.get(sc._jsc.hadoopConfiguration())

# ———————————————————

# 1) LOAD RAW NYC DATA

# ———————————————————

raw_path = "hdfs://localhost:9000/user/nan25/raw/nypd_raw_data.csv"
df_raw = spark.read.option("header", True).csv(raw_path)

# ———————————————————

# 2) SELECT ONLY THE COLUMNS YOU HAVE

# ———————————————————

df_cleaned = df_raw.select( col("ARREST_KEY").alias("id"),
col("ARREST_DATE").alias("date"), upper(col("OFNS_DESC")).alias("type"),
col("PD_DESC").alias("desc") )

# ———————————————————

# 3) DROP NULL ROWS

# ———————————————————

df_cleaned = df_cleaned.na.drop("any")

# ———————————————————

# 4) SAVE CLEANED RAW FILE (NO SUBPROCESS)

# ———————————————————

temp_path ="hdfs://localhost:9000/user/nan25/processed/nyc_cleaned_temp" 
final_path= "/user/nan25/processed/nyc_cleaned.csv"

df_cleaned.coalesce(1).write.format("csv").mode("overwrite").option("header", True).save(temp_path)

# find part file

status = fs.listStatus(sc._jvm.Path(temp_path))
part_file = None

for file in status:
    name = file.getPath().getName()
    if name.startswith("part-"):
        part_file = name
        break

# move final file

fs.rename( sc._jvm.Path(f"{temp_path}/{part_file}"),
sc._jvm.Path(final_path) )

# delete temp folder

fs.delete(sc._jvm.Path(temp_path), True)

# ———————————————————

# 5) LOAD CLEANED CSV BACK

# ———————————————————

df = spark.read.option("header", True).csv("hdfs://localhost:9000" +
final_path)

df = df.withColumn("id", col("id").cast("int")) 
df = df.withColumn("date", to_date(col("date"), "MM/dd/yyyy"))

# ———————————————————

# 6) GROUP DAILY CRIME RECORDS

# ———————————————————

df_assault = ( df.filter(col("type").contains("ASSAULT"))
.groupBy("date").count().sort("date") .coalesce(1) )

df_larceny = ( df.filter(col("type").contains("LARCENY"))
.groupBy("date").count().sort("date") .coalesce(1) )

df_total = ( df.groupBy("date").count().sort("date") .coalesce(1) )

# ———————————————————

# 7) SAVE FUNCTION (CLEAN + MATCHES CHICAGO STYLE)

# ———————————————————

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


# ———————————————————

# 8) SAVE FINAL NYPD DATASETS

# ———————————————————

save_csv(df_assault, "nyc_temp_assault",
"/user/nan25/data/nyc_assault_occurrence_per_day.csv")

save_csv(df_larceny, "nyc_temp_larceny",
"/user/nan25/data/nyc_larceny_occurrence_per_day.csv")

save_csv(df_total, "nyc_temp_total",
"/user/nan25/data/nyc_crime_occurrence_per_day.csv")
