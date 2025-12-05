from pyspark.sql import SparkSession

# Start Spark
spark = SparkSession.builder.appName("Chicago_Loading").getOrCreate()

###########################################
# Load Chicago raw dataset from HDFS
###########################################
input_path = "hdfs://localhost:9000/user/nan25/raw/Chi_Crimes_2001_to_Present.csv"

df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(input_path)

###########################################
# Show schema & preview data
###########################################
df.printSchema()
df.show(20)
