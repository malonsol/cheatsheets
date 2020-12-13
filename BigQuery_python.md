Source: https://www.kaggle.com/dansbecker/getting-started-with-sql-and-bigquery

# BigQuery and SQL
## 1. Getting started with SQL and BigQuery commands
### Your first BigQuery commands

```python
from google.cloud import bigquery
```

In BigQuery, each dataset is contained in a corresponding project.

```python
# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "hacker_news" dataset
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)
```

Every dataset is just a collection of tables. You can think of a dataset as a spreadsheet file containing multiple tables, all composed of rows and columns.

```python
# List all the tables in the "hacker_news" dataset
tables = list(client.list_tables(dataset))

# Print names of all tables in the dataset (there are four!)
for table in tables:  
    print(table.table_id)

# Construct a reference to the "full" table
table_ref = dataset_ref.table("full")

# API request - fetch the table
table = client.get_table(table_ref)
```

![BigQuery structure](https://i.imgur.com/biYqbUB.png)

### Table schema

The structure of a table is called its **schema**. We need to understand a table's schema to effectively pull out the data we want.

```python
# Print information on all the columns in the "full" table in the "hacker_news" dataset
table.schema
```

Each `SchemaField` tells us about a specific column (which we also refer to as a **field**). In order, the information is:

- The **name** of the column
- The **field** type (or datatype) in the column
- The **mode** of the column (`'NULLABLE'` means that a column allows NULL values, and is the default)
- A **description** of the data in that column

```python
# Preview the first five lines of the "full" table
client.list_rows(table, max_results=5).to_dataframe()

# Preview the first five entries in the "by" column of the "full" table
client.list_rows(table, selected_fields=table.schema[:1], max_results=5).to_dataframe()
```


## 2. Select, From & Where
### Introduction
We'll begin by using the keywords SELECT, FROM, and WHERE to get data from specific columns based on conditions you specify.

### SELECT ... FROM
The most basic SQL query selects a single column from a single table. To do this,

- specify the column you want after the word **SELECT**, and then
- specify the table after the word **FROM**.

![First queries](https://i.imgur.com/c3GxYRt.png)

Note that when writing an SQL query, the argument we pass to FROM is not in single or double quotation marks (' or "). It is in backticks ( \` ).

### WHERE ...
```python
# Query to select all the items from the "city" column where the "country" column is 'US'
query = """
        SELECT city
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """
```

### Submitting the query to the dataset
```python
# Create a "Client" object
client = bigquery.Client()

# Set up the query
query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame
us_cities = query_job.to_dataframe()

# What five cities have the most measurements?
us_cities.city.value_counts().head()
```

### More queries
If you want multiple columns, you can select them with a comma between the names:
```python
query = """
        SELECT city, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """
```
You can select all columns with a * like this:
```python
query = """
        SELECT *
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """
```

### Working with big datasets
BigQuery datasets can be huge. To begin,you can estimate the size of any query before running it. Here is an example using the (*very large!*) Hacker News dataset. To see how much data a query will scan, we create a `QueryJobConfig` object and set the `dry_run` parameter to `True`.
```python
# Query to get the score column from every row where the type column has value "job"
query = """
        SELECT score, title
        FROM `bigquery-public-data.hacker_news.full`
        WHERE type = "job" 
        """

# Create a QueryJobConfig object to estimate size of query without running it
dry_run_config = bigquery.QueryJobConfig(dry_run=True)

# API request - dry run query to estimate costs
dry_run_query_job = client.query(query, job_config=dry_run_config)

print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))
```
You can also specify a parameter when running the query to limit how much data you are willing to scan. Here's an example with a low limit.
```python
# Only run the query if it's less than 1 MB
ONE_MB = 1000*1000
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_MB)

# Set up the query (will only run if it's less than 1 MB)
safe_query_job = client.query(query, job_config=safe_config)

# API request - try to run the query, and return a pandas DataFrame
safe_query_job.to_dataframe()
```
In this case, the query was cancelled, because the limit of 1 MB was exceeded. However, we can increase the limit to run the query successfully!
```python
# Only run the query if it's less than 1 GB
ONE_GB = 1000*1000*1000
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_GB)

# Set up the query (will only run if it's less than 1 GB)
safe_query_job = client.query(query, job_config=safe_config)

# API request - try to run the query, and return a pandas DataFrame
job_post_scores = safe_query_job.to_dataframe()

# Print average score for job posts
job_post_scores.score.mean()
```


## 3. Group By, Having & Count
Made-up table of information on pets:
![Table](https://i.imgur.com/fI5Pvvp.png)
### COUNT()
**COUNT()** returns a count of things.
![COUNT()](https://i.imgur.com/Eu5HkXq.png)

### GROUP BY()
**GROUP BY** takes the name of one or more columns, and treats all rows with the same value in that column as a single group when you apply aggregate functions like **COUNT()**.
![GROUP BY](https://i.imgur.com/tqE9Eh8.png)

### GROUP BY ... HAVING()
**HAVING** is used in combination with **GROUP BY** to ignore groups that don't meet certain criteria.
![GROUP BY ... HAVING()](https://i.imgur.com/2ImXfHQ.png)
```python
# Query to select comments that received more than 10 replies
query_popular = """
                SELECT parent, COUNT(id)
                FROM `bigquery-public-data.hacker_news.comments`
                GROUP BY parent
                HAVING COUNT(id) > 10
                """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 10 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query_popular, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
popular_comments = query_job.to_dataframe()

# Print the first five rows of the DataFrame
popular_comments.head()
```


## 4. Order By
Made-up table of information on pets:
![Table with dates](https://i.imgur.com/b99zTLv.png)
### ORDER BY
![Numbers](https://i.imgur.com/6o9LuTA.png)
![Text](https://i.imgur.com/ooxuzw3.png)
Descending order: **DESC**
![Desc](https://i.imgur.com/IElLJrR.png)

### Dates (DATE / DATETIME)
The **DATE** format has the year first, then the month, and then the day. It looks like this:
```python
YYYY-[M]M-[D]D
```
The **DATETIME** format is like the date format ... but with time added at the end.

### EXTRACT
Often you'll want to look at part of a date, like the year or the day. You can do this with **EXTRACT**.
![Date](https://i.imgur.com/vhvHIh0.png)
![Day](https://i.imgur.com/PhoWBO0.png)
![Week](https://i.imgur.com/A5hqGxY.png)

### Example: Which day of the week has the most fatal motor accidents?
```python
# Query to find out the number of accidents for each day of the week
query = """
        SELECT COUNT(consecutive_number) AS num_accidents, 
               EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        GROUP BY day_of_week
        ORDER BY num_accidents DESC
        """
# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)
query_job = client.query(query, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
accidents_by_day = query_job.to_dataframe()

# Print the DataFrame
accidents_by_day
```


## 5. As & With
Made-up table of information on pets:
![Table with years](https://i.imgur.com/MXrsiAZ.png)
### AS
Here's an example of a query *without* an **AS** clause:
![without_AS](https://i.imgur.com/VelX9tP.png)
And here's an example of the same query, but *with* **AS**:
![with_AS](https://i.imgur.com/teF84tU.png)

### WITH ... AS
A **common table expression** (or **CTE**) is a temporary table that you return within your query. CTEs are helpful for splitting your queries into readable chunks, and you can write queries against them.
![WITH_AS_incomplete](https://i.imgur.com/0Kz8q4x.png)
While this incomplete query above won't return anything, it creates a CTE that we can then refer to (as `Seniors`) while writing the rest of the query.
We can finish the query by pulling the information that we want from the CTE. The complete query below first creates the CTE, and then returns all of the IDs from it.
![WITH_AS_complete](https://i.imgur.com/3xQZM4p.png)
Also, it's important to note that CTEs only exist inside the query where you create them, and you can't reference them in later queries. So, any query that uses a CTE is always broken into two parts:
1. first, we create the CTE, and then
2. we write a query that uses the CTE.

### Example: How many Bitcoin transactions are made per day?
```python
# Query to select the number of transactions per date, sorted by date
query_with_CTE = """ 
                 WITH time AS 
                 (
                     SELECT DATE(block_timestamp) AS trans_date
                     FROM `bigquery-public-data.crypto_bitcoin.transactions`
                 )
                 SELECT COUNT(1) AS transactions,
                        trans_date
                 FROM time
                 GROUP BY trans_date
                 ORDER BY trans_date
                 """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 10 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query_with_CTE, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
transactions_by_date = query_job.to_dataframe()

# Print the first five rows
transactions_by_date.head()

transactions_by_date.set_index('trans_date').plot()
```
![WITH_AS_example](https://www.kaggleusercontent.com/kf/43796883/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..jsKNz9xpuEw9Z2MCHygaOw.vr6hEhtmQxWu5ZEnWjTDCt2WMs0PFpp6OAbWkHPedmqf6rFcedHZSdIpy52UpYTnkCFw8TWk2neUivKUW6IUah5QHXiXUXw3s50jfe-hBBzmgnMAmuD-nMUX1q_QVRMXxFe_k5pL8fWe0r8GV4fnOPfg_Gg8aq0gYK1DKCH4IeKdG2Qn_AoWwPef--ta73Xy92fxfdMHcHi2LbKCimF0sMD42OgTDK1jlr0LuBRoGm822gfggOjv2Hmq9y5ILTnMDhjP8jQbxxG8GLww3pzBp6LH8Vw5-f8Y3nMc-9oAWDK1Wd2ryAtvBhINzLBXwnLnMlkdaThbTQ78_W6PHk8CDcPoRs1GZiAp7Uh5Dzf-MbIhfWa2aWfOqW9mlx6rjgNQcPsCoYoxgQ93aSsHSotREf5mlfnH6VQBtbWxPWefcBOS0Mlt_Zl0s_vn3tsgwjbvnBP8P2NlkJ_7o4ce3RPAnhliYDSO6lS-BHJWtZ4YZAx8fPhaNxj30033G1Cfx1-lNmp8yd69Ad0WfrNtPsxbG8NpEMRb3uVTNmAYyuGQWZ6M4wV3nWt0lMaMC-gJXTdPl2Sa5CczDBFkohi39CaWJbVrH3q6BEMhyvkWAjbbM9nPZnSjZ4Ky_I1tNgn0yFvj.7VPJtQ406JdK84DSPW5vng/__results___files/__results___6_1.png)

As you can see, common table expressions (CTEs) let you shift a lot of your data cleaning into SQL. That's an especially good thing in the case of BigQuery, because **it is vastly faster than doing the work in Pandas**.


## 6. Joining Data
Made-up tables of information on pets and owners:
![JOIN_Two_tables](https://i.imgur.com/eXvIORm.png)
![JOIN_Two_tables_Query](https://i.imgur.com/fLlng42.png)
- *In general, when you're joining tables, it's a good habit to specify which table each of your columns comes from. That way, you don't have to pull up the schema every time you go back to read the query.*

The type of **JOIN** we're using today is called an **INNER JOIN**. That means that a row will only be put in the final output table if the value in the columns you're using to combine them shows up in both the tables you're joining. For example, if Tom's ID number of 4 didn't exist in the pets table, we would only get 3 rows back from this query. There are other types of **JOIN**.

### Example: How many files are covered by each type of software license?
```python
# Query to determine the number of files per license, sorted by number of files
query = """
        SELECT L.license, COUNT(1) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` AS sf
        INNER JOIN `bigquery-public-data.github_repos.licenses` AS L 
            ON sf.repo_name = L.repo_name
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 10 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
file_count_by_license = query_job.to_dataframe()
```
