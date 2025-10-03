# DO NOT CHANGE THIS CELL
import time
import matplotlib.pyplot as plt

import datashader as ds
import datashader.transfer_functions as tf

# DO NOT CHANGE THIS CELL
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

# DO NOT CHANGE THIS CELL
start=time.time()

# get points
ds_points_pandas=ds.Canvas().points(df,'long','lat')
display(ds_points_pandas)

# plot points
plt.imshow(tf.shade(ds_points_pandas))

print(f'Duration: {round(time.time()-start, 2)} seconds')

# DO NOT CHANGE THIS CELL
import cudf

dtype_dict={
    'age': 'int8', 
    'sex': 'object', 
    'county': 'object', 
    'lat': 'float32', 
    'long': 'float32', 
    'name': 'object'
}
        
gdf=cudf.read_csv('./data/uk_pop.csv', dtype=dtype_dict)
gdf.head()

# DO NOT CHANGE THIS CELL
start=time.time()

# get points
ds_points_cudf=ds.Canvas().points(gdf,'long','lat')
display(ds_points_cudf)

# plot points
plt.imshow(tf.shade(ds_points_cudf))

print(f'Duration: {round(time.time()-start, 2)} seconds')

import cuxfilter as cxf

# factorize county for multiselect widget
gdf['county'], county_names = gdf['county'].factorize()
county_map = dict(zip(list(range(len(county_names))), county_names.to_arrow()))

# create cuxfilter DataFrame
cxf_data = cxf.DataFrame.from_dataframe(gdf)

# create Datashader scatter plot
scatter_chart = cxf.charts.scatter(x='long', y='lat')

# create Bokeh bar charts
chart_3=cxf.charts.bar('age')
chart_2=cxf.charts.bar('sex')

# define layout
layout_array=[[1, 2, 2], 
              [3, 2, 2]]

# create multiselect widget
county_widget = cxf.charts.panel_widgets.multi_select('county', label_map=county_map)

# define layout
dash = cxf_data.dashboard(charts=[chart_2, scatter_chart, chart_3],sidebar=[county_widget], theme=cxf.themes.dark, data_size_widget=True, layout_array=layout_array)

dash.app()

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)