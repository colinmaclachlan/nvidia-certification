import cudf
import cuml

# road_nodes = cudf.read_csv('./data/road_nodes_2-06.csv', dtype=['str', 'float32', 'float32', 'str'])
road_nodes = cudf.read_csv('./data/road_nodes.csv', dtype=['str', 'float32', 'float32', 'str'])

road_nodes.dtypes
road_nodes.shape
road_nodes.head()

hospitals = cudf.read_csv('./data/clean_hospitals_full.csv')
hospitals.dtypes
hospitals.shape
hospitals.head()

knn = cuml.NearestNeighbors(n_neighbors=3)

road_locs = road_nodes[['east', 'north']]
knn.fit(road_locs)

distances, indices = knn.kneighbors(hospitals[['easting', 'northing']], 3) # order has to match the knn fit order (east, north)

SELECTED_RESULT = 10
print('hospital coordinates:\n', hospitals.loc[SELECTED_RESULT, ['easting', 'northing']], sep='')

nearest_road_nodes = indices.iloc[SELECTED_RESULT, 0:3]
print('node_id:\n', nearest_road_nodes, sep='')

print('road_node coordinates:\n', road_nodes.loc[nearest_road_nodes, ['east', 'north']], sep='')

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)