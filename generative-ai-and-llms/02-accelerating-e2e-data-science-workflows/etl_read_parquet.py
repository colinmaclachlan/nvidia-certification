import pandas as pd
import time


sel=[('county', '=', 'BLACKPOOL')]
parquet_df=pd.read_parquet('senior_df.parquet', columns=['age', 'sex', 'county', 'lat', 'long', 'name', 'R'], filters=sel)
parquet_df=parquet_df.loc[parquet_df['county']=='BLACKPOOL']

parquet_df['county'].unique()


df=pd.read_csv('./senior_df.csv', usecols=['age', 'sex', 'county', 'lat', 'long', 'name', 'R'])
df=df.loc[df['county']=='BLACKPOOL']

df['county'].unique()

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)