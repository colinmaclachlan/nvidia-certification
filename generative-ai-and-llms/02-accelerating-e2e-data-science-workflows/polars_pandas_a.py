import polars as pl
import time

start_time = time.time()

polars_df = pl.read_csv('./data/uk_pop.csv')

polars_time = time.time() - start_time

print(f"Time Taken: {polars_time:.4f} seconds")

polars_df.head()

start_time = time.time()

#load data
polars_df = pl.read_csv('./data/uk_pop.csv')

# Filter for ages above 0
filtered_df = polars_df.filter(pl.col('age') > 0.0)

#Sort by name
sorted_df = filtered_df.sort('name', descending=True)

print(sorted_df.head())
polars_time = time.time() - start_time
print(f"Time Taken: {polars_time:.4f} seconds")