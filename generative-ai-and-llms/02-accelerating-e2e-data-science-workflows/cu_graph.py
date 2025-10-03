import warnings
warnings.filterwarnings('ignore')

import cudf
import cugraph as cg

import cuxfilter as cxf
from bokeh.palettes import Magma, Turbo256, Plasma256, Viridis256

road_graph = cudf.read_csv('./data/road_graph.csv', dtype=['int32', 'int32', 'float32'])
road_graph.head()

speed_graph = cudf.read_csv('./data/road_graph_speed.csv', dtype=['int32', 'int32', 'float32'])
speed_graph.head()

road_nodes = cudf.read_csv('./data/road_nodes.csv', dtype=['str', 'float32', 'float32', 'str'])
road_nodes = road_nodes.drop_duplicates() # again, some road nodes appeared on multiple map tiles in the original source
road_nodes.head()

road_nodes.shape
speed_graph.src.max()

G = cg.Graph()
%time G.from_cudf_edgelist(road_graph, source='src', destination='dst', edge_attr='length')
G.number_of_nodes()
G.number_of_edges()

deg_df = G.degree()
deg_df['degree'].describe()[1:]

deg_df[deg_df['degree'].mod(2) == 1]

road_graph.loc[road_graph.src == road_graph.dst]

demo_node = deg_df.nlargest(1, 'degree')
demo_node_graph_id = demo_node['vertex'].iloc[0]
demo_node_graph_id

%time shortest_distances_from_demo_node = cg.sssp(G, demo_node_graph_id)
shortest_distances_from_demo_node.head()

shortest_distances_from_demo_node['distance'].loc[shortest_distances_from_demo_node['distance'] < 2**32].describe()[1:]

G_ex = cg.Graph()
G_ex.from_cudf_edgelist(speed_graph, source='src', destination='dst', edge_attr='length_s')

ex_node = ex_deg.nlargest(1, 'degree')

%time ex_dist = cg.sssp(G_ex, ex_node['vertex'].iloc[0])

ex_dist['distance'].loc[ex_dist['distance'] < 2**32].describe()[1:]

mapping = cudf.read_csv('./data/node_graph_map.csv')
mapping.head()

ex_dist.head()

road_nodes = road_nodes.merge(mapping, on='node_id')
road_nodes = road_nodes.merge(ex_dist, left_on='graph_id', right_on='vertex')
road_nodes.head()

gdf = road_nodes[['east', 'north', 'distance']]
gdf = gdf[gdf['distance'] < 2**32]
gdf['distance'] = gdf['distance'].pow(1/2).mul(-1)

cxf_data = cxf.DataFrame.from_dataframe(gdf)

heatmap_chart = cxf.charts.datashader.scatter(x='east', y='north', 
                                              aggregate_col='distance',
                                              aggregate_fn='mean',
                                              point_size=1)

dash = cxf_data.dashboard([heatmap_chart], theme=cxf.themes.dark, data_size_widget=True)

dash.app()

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)