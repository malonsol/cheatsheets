Source: https://www.kaggle.com/dansbecker/getting-started-with-sql-and-bigquery

# BigQuery and SQL
## Your first BigQuery commands

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

## Table schema

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



```python

```

