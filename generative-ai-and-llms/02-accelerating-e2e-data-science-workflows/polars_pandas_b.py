import pandas as pd
import time
start_time = time.time()
pandas_df = pd.read_csv('./data/uk_pop.csv')

filtered_df = pandas_df[pandas_df['age'] > 0.0]

sorted_df = filtered_df.sort_values(by=['name'], ascending=False)

pandas_time = time.time() - start_time
print(f"Time Taken for cuDF Pandas: {pandas_time:.4f} seconds\n")

df = pl.read_csv('./data/uk_pop.csv')

print(df.head())


filtered = (
    df.filter(pl.col("age") >= 65)
    .sort("age", descending=False)
)

print(filtered)


agg = (
    df.group_by("county")
    .agg([
        pl.len().alias("population"),
        pl.mean("age").alias("average_age")
    ])
    .sort("population", descending=True)
)

print(agg.head())

gender = (
    df.group_by("sex")
    .agg(pl.len().alias("count"))
    .with_columns(
        (pl.col("count") / df.shape[0] * 100).alias("percentage")
    )
)

print(gender)