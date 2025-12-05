import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import java.io._

val RAW_PATH = "C:/crime_project/data_raw/Chi_Crimes_2001_to_Present.csv"
val OUTPUT_SUMMARY = "C:/crime_project/profiling_chicago_summary.txt"

// Create Spark session
val spark = SparkSession.builder
    .appName("Profiling Chicago Crime Data")
    .getOrCreate()

import spark.implicits._

println(s"Loading CSV from: $RAW_PATH")

val options = Map(
  "header" -> "true",
  "inferSchema" -> "true",
  "delimiter" -> ","
)

val df = spark.read.options(options).csv(RAW_PATH)

// Show schema
println("\n=== Schema (raw) ===")
df.printSchema()

println("\n=== Sample rows (raw) ===")
df.show(10, truncate = false)

// Trim column names
val cols = df.columns.map(_.trim)

println(s"\nColumns found (${cols.length}):")
println(cols.mkString(", "))

// Save summary to file
val writer = new PrintWriter(new File(OUTPUT_SUMMARY))

writer.println("=== RAW DATA SUMMARY ===")
writer.println(s"Source file: $RAW_PATH")
writer.println(s"Row count: ${df.count()}")
writer.println(s"Column count: ${cols.length}")
writer.println("Columns:")
writer.println(cols.mkString(", "))

writer.close()

println(s"\nSummary saved to: $OUTPUT_SUMMARY")

spark.stop()
