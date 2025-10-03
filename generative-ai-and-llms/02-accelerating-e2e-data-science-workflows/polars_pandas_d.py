# lazy_df = pl.scan_csv('./data/uk_pop.csv').collect(engine="gpu")

import polars as pl
import time

gpu_engine = pl.GPUEngine(
    device=0, # This is the default
    raise_on_fail=True, # Fail loudly if we can't run on the GPU.
)

lazy_df = pl.scan_csv('./data/uk_pop.csv').collect(engine=gpu_engine)

start_time = time.time()

# Create a lazy DataFrame
lazy_df = pl.scan_csv('./data/uk_pop.csv')

# Define the lazy operations
lazy_result = (
    lazy_df
    .filter(pl.col('age') > 0.0)
    .sort('name', descending=True)
)

# Switch to gpu_engine
result = lazy_result.collect(engine=gpu_engine)

print(result.head())
polars_time = time.time() - start_time
print(f"Time Taken: {polars_time:.4f} seconds")

from polars.testing import assert_frame_equal

# Run on the CPU
result_cpu = lazy_result.collect()

# Run on the GPU
result_gpu = lazy_result.collect(engine="gpu")

# assert both result are equal - Will error if not equal, return None otherwise
if (assert_frame_equal(result_gpu, result_cpu) == None):
    print("The test frames are equal")

result = (
    lazy_df
    .with_columns(pl.col('age').rolling_mean(window_size=7).alias('age_rolling_mean'))
    .filter(pl.col('age') > 0.0)  
    .collect(engine=gpu_engine)
)
print(result[::7])

gpu_engine_with_fallback = pl.GPUEngine(
    device=0, # This is the default
    raise_on_fail=False, # Fallback to CPU if we can't run on the GPU (this is the default)
)

result = (
    lazy_df
    .with_columns(pl.col('age').rolling_mean(window_size=7).alias('age_rolling_mean'))
    .filter(pl.col('age') > 0.0)  
    .collect(engine=gpu_engine_with_fallback)
)
print(result[::7])

# Create the lazy query with column pruning
lazy_query = (
    lazy_df
    .select(["county", "lat", "long"])  # Column pruning: select only necessary columns
    .group_by("county")
    .agg([
        pl.col("lat").mean().alias("avg_latitude"),
        pl.col("long").mean().alias("avg_longitude")
    ])
    .sort("county")
)

# Execute the query
result = lazy_query.collect(engine="gpu")

print("\nAverage latitude and longitude for each county:")
print(result.head())  # Display first few rows

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)