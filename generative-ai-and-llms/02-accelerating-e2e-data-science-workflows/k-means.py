import cudf
import cuml

import cuxfilter as cxf

gdf = cudf.read_csv('./data/clean_uk_pop.csv', usecols=['easting', 'northing'])
print(gdf.dtypes)
gdf.shape
gdf.head()

# instantaite
km = cuml.KMeans(n_clusters=5)

# fit
km.fit(gdf)

# assign cluster as new column
gdf['cluster'] = km.labels_
km.cluster_centers_

km = cuml.KMeans(n_clusters=6)

km.fit(gdf)
gdf['cluster'] = km.labels_
km.cluster_centers_

# associate a data source with cuXfilter
cxf_data = cxf.DataFrame.from_dataframe(gdf)

# define charts
scatter_chart = cxf.charts.datashader.scatter(x='easting', y='northing')

# define widget using the `cluster` column for multiselect
# use the same technique to scale the scatterplot, then add a widget to let us select which cluster to look at
cluster_widget = cxf.charts.panel_widgets.multi_select('cluster')

dash = cxf_data.dashboard(charts=[scatter_chart],sidebar=[cluster_widget], theme=cxf.themes.dark, data_size_widget=True)

dash.app()

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)