import pandas as pd
import random
import time

df=pd.read_csv('./data/uk_pop.csv')

# preview
df.head()

# pandas memory utilization
mem_usage_df=df.memory_usage(deep=True)
mem_usage_df

suffixes = ['B', 'kB', 'MB', 'GB', 'TB', 'PB']
def make_decimal(nbytes):
    i=0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes/=1024.
        i+=1
    f=('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

make_decimal(mem_usage_df.sum())

# get number of rows
num_rows=len(df)

# 64-bit numbers uses 8 bytes of memory
print(f'Numerical columns use {num_rows*8} bytes of memory')

# check random string-typed column
string_cols=[col for col in df.columns if df[col].dtype=='object' ]
column_to_check=random.choice(string_cols)

overhead=49
pointer_size=8

# nan==nan when value is not a number
# nan uses 32 bytes of memory
string_col_mem_usage_df=df[column_to_check].map(lambda x: len(x)+overhead+pointer_size if x else 32)
string_col_mem_usage=string_col_mem_usage_df.sum()
print(f'{column_to_check} column uses {string_col_mem_usage} bytes of memory.')

df['age']=df['age'].astype('int8')

df.dtypes

df['lat']=df['lat'].astype('float32')
df['long']=df['long'].astype('float32')

df.select_dtypes(include='object').nunique()

df['sex']=df['sex'].astype('category')
df['county']=df['county'].astype('category')

display(df['county'].cat.categories)
print('-'*40)
display(df['county'].cat.codes)

start=time.time()
df=pd.read_csv('./data/uk_pop.csv')
duration=time.time()-start

mem_usage_df=df.memory_usage(deep=True)
display(mem_usage_df)

print(f'Loading {make_decimal(mem_usage_df.sum())} took {round(duration, 2)} seconds.')

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)