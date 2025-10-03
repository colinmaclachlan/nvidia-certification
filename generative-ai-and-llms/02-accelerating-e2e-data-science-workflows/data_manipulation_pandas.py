import pandas as pd
import time
from datetime import datetime

start=time.time()

df=pd.read_csv('./data/uk_pop.csv')
current_year=datetime.now().year

df['birth_year']=current_year-df['age']

df['sex_normalize']=df['sex'].str.upper()
df['county_normalize']=df['county'].str.title().str.replace(' ', '_')
df['name']=df['name'].str.title()

print(f'Duration: {round(time.time()-start, 2)} seconds')

display(df.head())