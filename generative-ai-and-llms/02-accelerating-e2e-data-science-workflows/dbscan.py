import cudf
import cuml

import cuxfilter as cxf

gdf = cudf.read_csv('./data/pop_sample.csv', dtype=['float32', 'float32', 'float32'])
print(gdf.dtypes)
gdf.shape

gdf.head()
gdf['infected'].value_counts()

dbscan = cuml.DBSCAN(eps=5000)
# dbscan = cuml.DBSCAN(eps=10000)

infected_df = gdf[gdf['infected'] == 1].reset_index()
infected_df['cluster'] = dbscan.fit_predict(infected_df[['northing', 'easting']])
infected_df['cluster'].nunique()

dbscan = cuml.DBSCAN(eps=10000)

infected_df = gdf[gdf['infected'] == 1].reset_index()
infected_df['cluster'] = dbscan.fit_predict(infected_df[['northing', 'easting']])
infected_df['cluster'].nunique()

infected_df.to_pandas().plot(kind='scatter', x='easting', y='northing', c='cluster')

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)