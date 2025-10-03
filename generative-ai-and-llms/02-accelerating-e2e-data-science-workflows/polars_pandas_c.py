import polars as pl
import time

start_time = time.time()

# Create a lazy DataFrame
lazy_df = pl.scan_csv('./data/uk_pop.csv')

# Define the lazy operations
lazy_result = (
    lazy_df
    .filter(pl.col('age') > 0.0)
    .sort('name', descending=True)
)

# Execute the lazy query and collect the results
result = lazy_result.collect()

print(result.head())
polars_time = time.time() - start_time
print(f"Time Taken: {polars_time:.4f} seconds")

# Show unoptimized Graph
lazy_result.show_graph(optimized=False)

# Show optimized Graph
lazy_result.show_graph(optimized=True)

lazy_df = pl.scan_csv('./data/uk_pop.csv')

result = (
    lazy_df.filter(pl.col("age") < 30)
    .group_by("name")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
    .limit(5)
    .select(["name", "count"])
)

top_5_names=result.collect()
print(top_5_names)

lazy_df = pl.scan_csv('./data/uk_pop.csv').collect(engine="gpu")