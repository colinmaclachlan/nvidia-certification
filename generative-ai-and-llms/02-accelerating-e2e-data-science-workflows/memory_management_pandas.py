import pandas as pd
import time

suffixes = ['B', 'kB', 'MB', 'GB', 'TB', 'PB']
def make_decimal(nbytes):
    i=0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes/=1024.
        i+=1
    f=('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

%%cudf.pandas.line_profile
# DO NOT CHANGE THIS CELL
start=time.time()

# define data types for each column
dtype_dict={
    'age': 'int8', 
    'sex': 'category', 
    'county': 'category', 
    'lat': 'float64', 
    'long': 'float64', 
    'name': 'category'
}
        
efficient_df=pd.read_csv('./data/uk_pop.csv', dtype=dtype_dict)
duration=time.time()-start

mem_usage_df=efficient_df.memory_usage('deep')
display(mem_usage_df)

print(f'Loading {make_decimal(mem_usage_df.sum())} took {round(duration, 2)} seconds.')

mem_capacity=16*1073741824

mem_per_record=mem_usage_df.sum()/len(efficient_df)

print(f'We can load {int(mem_capacity/2/mem_per_record)} number of rows.')

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)