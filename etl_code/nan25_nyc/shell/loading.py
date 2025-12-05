from pyspark.sql import SparkSession

# Start Spark
spark = SparkSession.builder.appName("NYC_Loading").getOrCreate()

###########################################
# Load NYC raw dataset from HDFS
###########################################
input_path = "hdfs://localhost:9000/user/nan25/raw/nypd_raw_data.csv"

df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(input_path)

###########################################
# Show schema & preview data
###########################################
df.printSchema()
df.show(20)
