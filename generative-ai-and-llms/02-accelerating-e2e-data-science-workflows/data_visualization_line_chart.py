import pandas as pd

dtype_dict={
    'age': 'int8', 
    'sex': 'object', 
    'county': 'object', 
    'lat': 'float32', 
    'long': 'float32', 
    'name': 'object'
}
        
df=pd.read_csv('./data/uk_pop.csv', dtype=dtype_dict)
df.head()

df.groupby('county').size().sort_values(ascending=False).head().plot(kind='bar')

bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
df['age_bucket']=pd.cut(df['age'], bins=bins, right=True, include_lowest=True, labels=False)
df.groupby('age_bucket').size().plot(kind='bar')

df.groupby('sex').size().plot(kind='bar')

# sample a very small percentage of the data
small_df=df.sample(1000)

small_df.plot(kind='scatter', x='lat', y='long')

import time
import matplotlib.pyplot as plt

fig, ax=plt.subplots()
exec_times={}

for size in (5*(10**i) for i in range(1, 8)): 
    start=time.time()
    df.sample(size).plot(kind='scatter', x='long', y='lat', ax=ax)
    duration=time.time()-start
    exec_times[size]=duration
    ax.clear()

ax.plot(exec_times.keys(), exec_times.values(), marker='o')
ax.set_xscale('log')
ax.set_xlabel('Data Size')
ax.set_ylabel('Execution Time')
ax.set_title("Scatter Plot Doesn't Scale Well With Data Size")

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)