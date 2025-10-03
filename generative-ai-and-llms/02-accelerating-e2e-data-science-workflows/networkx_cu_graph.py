# !NETWORKX_AUTOMATIC_BACKENDS=cugraph python -m cudf.pandas scripts/networkx.py

import warnings
warnings.filterwarnings('ignore')

%load_ext cudf.pandas
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
road_graph = pd.read_csv('./data/road_graph.csv', dtype=['int32', 'int32', 'float32'], nrows=1000)

# Create an empty graph
G = nx.from_pandas_edgelist(road_graph, source='src', target='dst', edge_attr='length')
b = nx.betweenness_centrality(G, k=1000, backend="cugraph")

import networkx as nx
import nx_cugraph as nxcg

# Loading data from previous cell
G = nx.from_pandas_edgelist(road_graph, source='src', target='dst', edge_attr='length') 

nxcg_G = nxcg.from_networkx(G)             # conversion happens once here
b = nx.betweenness_centrality(nxcg_G, k=1000)  # nxcg Graph type causes cugraph backend to be used, no conversion necessary

# Create a graph from already loaded dataframe
G = nx.from_pandas_edgelist(road_graph, source='src', target='dst', edge_attr='length')

b = nx.betweenness_centrality(G, backend="cugraph")

d = nx.degree_centrality(G, backend="cugraph")

k = nx.katz_centrality(G, backend="cugraph")

p = nx.pagerank(G, max_iter=10, tol=1.0e-3, backend="cugraph")

e = nx.eigenvector_centrality(G, max_iter=1000, tol=1.0e-3, backend="cugraph")

from IPython.display import display_html
dc_top = pd.DataFrame(sorted(d.items(), key=lambda x:x[1], reverse=True)[:5], columns=["vertex", "degree_centrality"])
bc_top = pd.DataFrame(sorted(b.items(), key=lambda x:x[1], reverse=True)[:5], columns=["vertex", "betweenness_centrality"])
katz_top = pd.DataFrame(sorted(k.items(), key=lambda x:x[1], reverse=True)[:5], columns=["vertex", "katz_centrality"])
pr_top = pd.DataFrame(sorted(p.items(), key=lambda x:x[1], reverse=True)[:5], columns=["vertex", "pagerank"])
ev_top = pd.DataFrame(sorted(e.items(), key=lambda x:x[1], reverse=True)[:5], columns=["vertex", "eigenvector_centrality"])

df1_styler = dc_top.style.set_table_attributes("style='display:inline'").set_caption('Degree').hide(axis='index')
df2_styler = bc_top.style.set_table_attributes("style='display:inline'").set_caption('Betweenness').hide(axis='index')
df3_styler = katz_top.style.set_table_attributes("style='display:inline'").set_caption('Katz').hide(axis='index')
df4_styler = pr_top.style.set_table_attributes("style='display:inline'").set_caption('PageRank').hide(axis='index')
df5_styler = ev_top.style.set_table_attributes("style='display:inline'").set_caption('EigenVector').hide(axis='index')

display_html(df1_styler._repr_html_()+df2_styler._repr_html_()+df3_styler._repr_html_()+df4_styler._repr_html_()+df5_styler._repr_html_(), raw=True)

import networkx as nx
import nx_cugraph as nxcg

# Loading data from previous cell
G = nx.from_pandas_edgelist(road_graph, source='src', target='dst', edge_attr='length') 

nxcg_G = nxcg.from_networkx(G)             # conversion happens once here
p = nx.pagerank(nxcg_G, max_iter=10, tol=1.0e-3) # nxcg Graph type causes cugraph backend to be used, no conversion necessary

pd.DataFrame(sorted(p.items(), key=lambda x:x[1], reverse=True)[:5], columns=["vertex", "pagerank"])

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)