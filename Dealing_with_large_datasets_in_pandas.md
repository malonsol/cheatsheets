# [Dealing with Large Datasets in Pandas](https://medium.com/datadriveninvestor/dealing-with-large-datasets-in-pandas-ad47100a9424)
*Credit: Ahmad Sami*

## 1. Loading the Dataset in chunks
There is no fixed chunksize you should be using. It all comes by trial and error to know what’s best for your machine. Once you load the data in chunks, you can loop through the chunks and process them individually.

```python
# empty initially
chunk_list = []

df_chunks = pd.read_csv('file_path.csv',chunksize = 5000)

for chunk in df_chunks:
  # perform your data processing 
  processed_chunk = clean_data(chunk)
  
  chunk_list.append(processed_chunk)
  
df_concated = pd.concat(chunk_list,ignore_index = True)
```

## 2. Dropping unnecessary columns
You can either drop the unnecessary columns or just select the ones you need. I honestly prefer to use the drop function to get rid of them as it clearly shows what has been disregarded.

```python
df_concated.drop(['col_1','col_2','col_3'],axis = 1,inplace = True)
```

## 3. Changing Datatypes
You can store and process larger datasets when using efficient data types. Some values can be stored in less memory demanding data types. You can use the `astype()` function to change a column’s data type as shown below.

```python
df_concated['fruit_name'] = df_concated['fruit_name'].astype('category')
```

Moreover, you can use the pandas `to_numeric()` function to downcast integer values to the smallest possible data type.

```python
df_concated['id'] = pd.to_numeric(df_concated['id'], downcast='unsigned')
```

## 4. Vectorization instead of Looping
Looping through a dataframe using `iterrows()` or `itertuples()` should only be considered as a last resort option when pandas vectorization cannot be applied.

```python
df_concated['calc'] = df_concated['num1'] * df_concated['num2']
```

Furthermore, if you need to filter out from a dataframe rows having certain values in a particular column, you can use the `isin()` function instead of looping through the dataframe as shown below.

```python
# selcting rows with fruit_name being apple,'orange,or pineapple

df_concated[df_concated['fruit_name'].isin(['apple','orange','pineapple'])]
```

## 5. Saving the Dataset in chunks
Just like we loaded the CSV file in chunks, we can also save the data in chunks using the chunksize parameter in the `to_csv()` function.

```python
df_concated.to_csv('file_path.csv',chunksize = 5000 ,index = False)
```
