import warnings
warnings.filterwarnings('ignore')

import cudf
import cugraph as cg

import networkx as nx

road_nodes = cudf.read_csv('./data/road_nodes.csv')
road_nodes.head()
road_nodes.dtypes
road_nodes.shape
road_nodes['type'].unique()

road_edges = cudf.read_csv('./data/road_edges.csv')
road_edges.head()
road_edges.dtypes
road_edges.shape
road_edges['type'].unique()
road_edges['form'].unique()

road_edges['src_id'] = road_edges['src_id'].str.lstrip('#')
road_edges['dst_id'] = road_edges['dst_id'].str.lstrip('#')
road_edges[['src_id', 'dst_id']].head()

print(f'{road_edges.shape[0]} edges, {road_nodes.shape[0]} nodes')

G = cg.Graph()
%time G.from_cudf_edgelist(road_edges, source='src_id', destination='dst_id', edge_attr='length')

road_edges_cpu = road_edges.to_pandas()
%time G_cpu = nx.convert_matrix.from_pandas_edgelist(road_edges_cpu, source='src_id', target='dst_id', edge_attr='length')


road_nodes = road_nodes.set_index('node_id', drop=True)
%time road_nodes = road_nodes.sort_index()
road_nodes.head()

G.number_of_nodes()
G.number_of_edges()

deg_df = G.degree()
deg_df['degree'].describe()[1:]

road_edges.dtypes
road_edges['type'].unique()

speed_gdf = cudf.DataFrame()

speed_gdf['type'] = speed_limits.keys()
speed_gdf['limit_mph'] = [speed_limits[key] for key in speed_limits.keys()]
speed_gdf

# We will have road distances in meters (m), so to get road distances in seconds (s), we need to multiply by meters/mile and divide by seconds/hour
# 1 mile ~ 1609.34 m
speed_gdf['limit_m/s'] = speed_gdf['limit_mph'] * 1609.34 / 3600
speed_gdf

%time road_edges = road_edges.merge(speed_gdf, on='type')

road_edges['length_s'] = road_edges['length'] / road_edges['limit_m/s']
road_edges['length_s'].head()

G_ex = cg.Graph()
G_ex.from_cudf_edgelist(road_edges, source='src_id', destination='dst_id', edge_attr='length_s')

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
