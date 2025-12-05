from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
import subprocess

spark = SparkSession.builder.appName("NYC_ETL2").getOrCreate()

# 1) INPUT FROM ETL1
input_path = "hdfs://localhost:9000/user/nan25/processed/nypd_type_date.csv"

df = spark.read.option("header", True).csv(input_path)

# Save helper
def save_single_csv(df, temp_dir, final_path):
    df.coalesce(1).write.option("header", True).mode("overwrite").csv(temp_dir)

    files = subprocess.check_output(["hdfs","dfs","-ls",temp_dir]).decode()
    part = [f.split()[-1] for f in files.split("\n") if "part-" in f][0]

    subprocess.call(["hdfs","dfs","-mv",
                     f"{temp_dir}/{part}",
                     final_path])
    subprocess.call(["hdfs","dfs","-rm","-r",temp_dir])


# 2) ALL CRIME COUNTS (NYC)
all_occ = (
    df.groupBy("date")
      .agg(count("*").alias("Count"))
      .orderBy("date")
)

save_single_csv(all_occ,
                "temp_nyc_all",
                "/user/nan25/processed/nypd_all.csv")


# 3) ASSAULT
df_assault = df.filter(col("typeDesc").rlike(".*ASSAULT.*"))

assault_occ = (
    df_assault.groupBy("date")
              .agg(count("*").alias("Count"))
              .orderBy("date")
)

save_single_csv(assault_occ,
                "temp_nyc_assault",
                "/user/nan25/processed/nypd_assault.csv")


# 4) LARCENY
df_larceny = df.filter(col("typeDesc").rlike(".*LARCENY.*"))

larceny_occ = (
    df_larceny.groupBy("date")
              .agg(count("*").alias("Count"))
              .orderBy("date")
)

save_single_csv(larceny_occ,
                "temp_nyc_larceny",
                "/user/nan25/processed/nypd_larceny.csv")
