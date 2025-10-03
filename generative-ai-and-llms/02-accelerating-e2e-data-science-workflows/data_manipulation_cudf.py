import cudf
import cupy as cp
import numpy as np

from datetime import datetime
import random
import time

start=time.time()
df=cudf.read_csv('./data/uk_pop.csv')
# print(f'Duration: {round(time.time()-start, 2)} seconds')

df.info(memory_usage='deep')
df.head()

# DO NOT CHANGE THIS CELL
# get first cell
display(df.loc[0, 'age'])
print('-'*40)

# get multiple rows and columns
display(df.loc[[0, 1, 2], ['age', 'sex', 'county']])
print('-'*40)

# slice a range of rows and columns
display(df.loc[0:5, 'age':'county'])
print('-'*40)

# slice a range of rows and columns
display(df.loc[:10, :'name'])

# get current year
current_year=datetime.now().year

# derive the birth year
display(current_year-df.loc[:, 'age'])

# get the age array (CuPy for cuDF)
age_ary=df.loc[:, 'age'].values

# derive the birth year
current_year-age_ary

# Exercise #1 - Convert county Column to Title Case
df['county'].str.title()

df['age']>=18

# DO NOT CHANGE THIS CELL
df[['lat', 'long']].mean()

# DO NOT CHNAGE THIS CELL
# define a function to check if age is greater than or equal to 18
start=time.time()
def is_adult(row): 
    if row['age']>=18: 
        return 1
    else: 
        return 0

# derive the birth year
display(df.apply(is_adult, axis=1))
print(f'Duration: {round(time.time()-start, 2)} seconds')

# derive the birth year
start=time.time()
display(df.apply(lambda x: 1 if x['age']>=18 else 0, axis=1))
print(f'Duration: {round(time.time()-start, 2)} seconds')

# derive the birth year
start=time.time()
display((df['age']>=18).astype('int'))
print(f'Duration: {round(time.time()-start, 2)} seconds')

# Below we use Series.map() to determine the number of characters in each person's name.
df['name'].map(lambda x: len(x))

boolean_mask=df['name'].str.startswith('E')
df.loc[boolean_mask]

df[(df['age']>=18) | (df['name'].str.startswith('E'))]

# Exercise #2 - Counties North of Sunderland

sunderland_residents=df.loc[df['county'] == 'SUNDERLAND']
northmost_sunderland_lat=sunderland_residents['lat'].max()
df.loc[df['lat'] > northmost_sunderland_lat]['county'].unique()

# get current year
current_year=datetime.now().year

# numerical operations
df['birth_year']=current_year-df['age']

# string operations
df['sex_normalize']=df['sex'].str.upper()
df['county_normalize']=df['county'].str.title().str.replace(' ', '_')
df['name']=df['name'].str.title()

# preview
df.head()

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)