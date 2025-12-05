Week 2 — Data Engineering Pipeline Development

Week 2 focuses on transforming raw crime data into a structured, cleaned, and analysable format using Hadoop + PySpark.
This week establishes the data ingestion, ETL pipeline, and initial cleaned datasets that will be used in later phases (EDA + ML).

 🔹 2.1 Data Ingestion (Raw Data → HDFS)

Tasks completed:

* Downloaded raw datasets from internet sources:

  * `Chi_Crimes_2001_to_Present.csv`
  * `nypd_raw_data.csv`
* Organised the datasets into the local project folder:

  ```
  raw_data/
      Chi_Crimes_2001_to_Present.csv
      nypd_raw_data.csv
  ```
* Uploaded the raw datasets to Hadoop HDFS using:

  ```
  hdfs dfs -mkdir /user/<id>/final_project/raw/
  hdfs dfs -put Chi_Crimes_2001_to_Present.csv /raw/
  hdfs dfs -put nypd_raw_data.csv /raw/
  ```
* Verified access and schema using `pyspark` and `spark-shell`.

 🔹 2.2 ETL Pipeline Setup 


```
etl_code/
  ├── nan25_nyc/
  │     ├── cleaning.py
  │     ├── pipeline/
  │     │      ├── etl1.py
  │     │      └── etl2.py
  │     └── shell/
  │            ├── etl1.py
  │            ├── etl2.py
  │            └── loading.py
  └── nan25_chicago/
        ├── cleaning.py
        ├── pipeline/
        │      ├── etl1.py
        │      └── etl2.py
        └── shell/
               ├── etl1.py
               ├── etl2.py
               └── loading.py
```

The pipeline is split into two processing phases:

 ETL1 — Cleaning + Column Selection

* Remove unnecessary columns
* Convert date fields to proper Spark `date` objects
* Drop rows with missing values
* Save cleaned output back to HDFS

 ETL2 — Aggregation & Feature Generation

* Compute total crime occurrences per day
* Compute per-type occurrences (Battery, Theft, Assault, etc.)
* Save grouped datasets back to HDFS
* This generates Week 2 processed datasets:

```
data/
  battery_occurrence_per_day.csv
  crime_occurrence_per_day.csv
  nypd_all.csv
  nypd_assault.csv
  nypd_larceny.csv
  theft_occurrence_per_day.csv
```

 🔹 2.3 Initial Data Merging (Start)

* Started merging NYC & Chicago metrics into unified “merged” datasets
* Ensured consistent column naming and date formatting
* Stored intermediate merged datasets as Parquet files for performance

 🔹 2.4 Summary of Week 2 Outputs

* Raw data successfully in HDFS
* Cleaning pipeline (ETL1) working for both NYC & Chicago
* Aggregation pipeline (ETL2) producing daily crime summaries

---

 ✅ Week 3 — Exploratory BI Report + Dashboard

Week 3 uses the cleaned and processed datasets created in Week 2 to perform exploratory analysis and build a preliminary BI dashboard.

 🔹 3.1 Complete ETL, Cleaning, and Merging

* Finalized all ETL stages for both cities
* Ensured both datasets have identical structures for comparison
* Created merged datasets for all crimes and type-specific crime types
* Saved merged outputs in:



 🔹 3.2 Exploratory Data Analysis (EDA)

Performed using Jupyter notebooks in:

```
ana_code/
     analysis_chicago_all.ipynb
     analysis_chicago_type1.ipynb
     analysis_chicago_type2.ipynb
     analysis_merged_all.ipynb
     analysis_merged_type1.ipynb
     analysis_merged_type2.ipynb
     analysis_nyc_all.ipynb
     analysis_nyc_type1.ipynb
     analysis_nyc_type2.ipynb
    ...
```

Key EDA steps:

* Time-series crime trends
* Daily/weekly/monthly patterns
* Top crime types in each city
* Compare Chicago vs NYC crime volume
* Missing data analysis
* Summary statistics

 🔹 3.3 Early BI Visuals

Created initial data visualizations:

* Line charts: crime occurrence per day
* Bar charts: top crime types
* Heatmaps: crime distribution across time
* City comparison charts

Tools used:

* Matplotlib / Seaborn
* Plotly (optional, for interactive charts)
* Power BI (dashboard draft)

Visualization outputs stored under:

```
output/
  chicago_all_pred.jpg
  chicago_type1_pred.jpg
  nyc_all_pred.jpg
  ...
```

 🔹 3.4 Dashboard v1 / Draft Report

Deliverables completed:

* Power BI dashboard (version 1)
* Draft EDA report summarizing:

  * Purpose
  * Methodology
  * Data description
  * Key observations
  * Early insights
* Notebook visuals exported into the report

