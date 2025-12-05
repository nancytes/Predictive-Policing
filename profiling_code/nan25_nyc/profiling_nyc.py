import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import java.io._

// ---------------- CONFIG ----------------
// Change these if your paths are different
val RAW_PATH = "C:/crime_project/data_raw/nypd_raw_data.csv"
val OUTPUT_SUMMARY = "C:/crime_project/profiling_nyc_summary.txt"
// -----------------------------------------

val spark = SparkSession.builder()
    .appName("Profiling_NYC_Crime_Local")
    .master("local[*]")
    .getOrCreate()

import spark.implicits._

println(s"\n=== Loading NYC CSV from: $RAW_PATH ===")

// Read raw NYC data
var df = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("delimiter", ",")
    .csv(RAW_PATH)

println("\n=== RAW Schema ===")
df.printSchema()

println("\n=== Sample Rows (Raw) ===")
df.show(10, truncate = 80)


// -------------------------------------------------------
// FIX DATE + CAST COLUMNS
// -------------------------------------------------------

println("\n=== Cleaning & Casting Columns ===")

// Convert date format
df = df.withColumn("date", to_date(col("date"), "MM/dd/yyyy"))

// Convert numeric columns safely
df = df.withColumn("id", col("id").cast("int"))
df = df.withColumn("typeId", col("typeId").cast("int"))

println("\n=== Schema After Casting ===")
df.printSchema()


// -------------------------------------------------------
// ADD WEEK NUMBER
// -------------------------------------------------------

println("\n=== Add week_number column ===")
df = df.withColumn("week_number", weekofyear(col("date")))

df.show(10)
df.printSchema()


// -------------------------------------------------------
// DISTINCT VALUES
// -------------------------------------------------------

println("\n=== Distinct Crime Types ===")
val distinct_crime_type = df.select("typeId").distinct()
println(s"Distinct crime types count = ${distinct_crime_type.count()}")

println("\n=== Distinct Dates ===")
val distinct_dates = df.select("date").distinct()
println(s"Distinct dates count = ${distinct_dates.count()}")


println("\n=== Frequent Crime Types (count > 1000) ===")
df.groupBy("typeId", "typeDesc")
  .count()
  .filter(col("count") > 1000)
  .sort(desc("count"))
  .show(false)


// -------------------------------------------------------
// SAVE SUMMARY TO FILE
// -------------------------------------------------------

println(s"\n=== Writing summary to $OUTPUT_SUMMARY ===")

val file = new PrintWriter(new File(OUTPUT_SUMMARY))

file.write("NYC Crime Data Profiling Summary\n")
file.write("--------------------------------\n")
file.write(s"Rows: ${df.count()}\n")
file.write(s"Distinct crime types: ${distinct_crime_type.count()}\n")
file.write(s"Distinct dates: ${distinct_dates.count()}\n")

file.close()

println("\n=== NYC PROFILING COMPLETE ===")

spark.stop()
