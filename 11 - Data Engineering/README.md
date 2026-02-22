
# Data Engineering

## Introduction
 
### Kaggle Data Set
https://www.kaggle.com/datasets

### What is data? 
Data is a collection of facts, numbers, words, observations or other useful information. 
Through data processing and data analysis, organizations transform raw data points into 
valuable insights that improve decision-making and drive better business outcomes.

### Structured data
Structured data is data that has a standardized format for efficient access by software 
and humans alike. It is typically tabular with rows and columns that clearly define data 
attributes. Computers can effectively process structured data for insights due to its 
quantitative nature.
* Data that contains rows and columns -> RDBMS

### Semi Structured data
Semi-structured data is a form of structured data that does not obey the tabular structure 
of data models associated with relational databases or other forms of data tables, but 
nonetheless contains tags or other markers to separate semantic elements and enforce 
hierarchies of records and fields within the data.
* Data that is stored in csv, json, xml 

### Unstructured data
Unstructured data is information that does not have a predefined format. 
Unstructured datasets are massive (often terabytes or petabytes of data) 
and contain 90% of all enterprise-generated data.
* Data that is in pdf, emails etc

### Binary data
Binary data is just a way of saying that it's data which is not text. In other words, 
it doesn't actually give you a lot of insight as to what the data is, rather it gives 
you insight as to what the data isn't.
It's all ones and zeros. But the ones and zeros live differently on your computer. 
* Data that is stored as Mp3, Video files, images etc

### Data Engineer
Data engineers work in various settings to build systems that collect, manage, and convert 
raw data into usable information for data scientists and business analysts to interpret. 
Their ultimate goal is to make data accessible so that organisations can use it to evaluate 
and optimise their performance.
Key functions:
* Data Mining - Data mining is the process of finding anomalies, patterns and correlations 
within large data sets to predict outcomes.
* Big Data - Extremely large and diverse collections of structured, unstructured, and 
semi-structured data that continues to grow exponentially over time.
* Data Pipeline - Is a series of processing steps to prepare enterprise data for analysis. 

### Real world example
Steam of Water (from different sources) -> Flows to a Lake -> Sent throught pipes
Sent to Desalination plants -> Sent throught pipes -> Finally To consumers (Homes)
Different sources of data -> Data Lake -> Data Pipelines -> Data Warehouse -> 
Data Pipelines -> Analytics and Data Modelling
Eg. IoT, Mobile Phones, Apps -> Azure Fabric, Databricks -> Apache Kafka, Azure Event Hub, 
Amazon Kensis -> MongoDB, Elastic, BigQuery -> PowerBI, Azure ML, Colab

### ETL - Extract, Transform, Load
An ETL Pipeline is a crucial data processing tool used to extract, transform, and load data 
from various sources into a destination system. The ETL process begins with the extraction of 
raw data from multiple databases, applications, or external sources. The data then undergoes 
transformation, where it is cleaned, formatted, and integrated to fit the target system's 
requirements. Finally, the transformed data is loaded into a data warehouse, database, 
or other storage systems. ETL pipelines are essential for ensuring data quality, improving data 
consistency, and enabling efficient data analysis and reporting.

### Types of Database
Relational DB (Transactional-ACID), NoSQL (Distributed), NewSQL (best of both)


### OLTP/OLAP
OLTP -> Online Transaction Processing (OLTP) is a computing pattern that supports fast, concurrent, 
and real-time operational data processing, such as banking transactions, e-commerce orders, or 
inventory updates. OLTP databases prioritize data integrity, using ACID compliance to handle high 
volumes of short, precise reads and writes. 

OLAP -> Online Analytical Processing (OLAP) is a technology designed for high-speed, multi-dimensional 
analysis of large, complex, and historical data, typically sourced from data warehouses. It enables 
business intelligence, forecasting, and complex queries, using multidimensional structures called "cubes" 
to allow users to slice, dice, and drill down into data for actionable insights. 


### Hadoop
It is an open-source framework that enables distributed storage and processing of massive datasets 
(petabytes) across clusters of commodity hardware. It provides high availability by automatically 
handling hardware failures, using HDFS for storage, YARN for resource management, and MapReduce 
for parallel processing.

### Apache Spark
It is an open-source unified analytics engine for large-scale data processing, known for its speed 
and ease of use in big data and machine learning workloads. It was developed to address the 
limitations of older systems like Hadoop MapReduce by performing operations in memory, making it up 
to 100 times faster for certain tasks.

### Apache Flink
It is an open-source, distributed stream-processing framework designed for high-throughput, 
low-latency stateful computations over both unbounded (streaming) and bounded (batch) data sets. 
It enables real-time analytics, fraud detection, and event-driven applications with "exactly-once" 
consistency guarantees. Flink supports Java, Python, and SQL, and runs on clusters like Kubernetes 
or YARN.

### Apache Kafka
It is an open-source, distributed event streaming platform designed for high-throughput, 
low-latency processing of real-time data feeds. It acts as a unified, persistent, and 
fault-tolerant log, enabling applications to publish, subscribe, store, and process streaming 
records in parallel across a cluster of brokers. 



### Reference
```xml
https://techdifferences.com/difference-between-oltp-and-olap.html
https://www.yugabyte.com/key-concepts/acid-properties/

https://www.kaggle.com/datasets
https://www.geeksforgeeks.org/software-testing/what-is-an-etl-pipeline/

```
