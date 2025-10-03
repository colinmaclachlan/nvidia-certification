import subprocess
import logging

from dask.distributed import Client, wait, progress
from dask_cuda import LocalCUDACluster

import cudf
import dask_cudf

import cuml
from cuml.dask.cluster import KMeans

# create cluster
cmd = "hostname --all-ip-addresses"
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
IPADDR = str(output.decode()).split()[0]

cluster = LocalCUDACluster(ip=IPADDR, silence_logs=logging.ERROR)
client = Client(cluster)

ddf = dask_cudf.read_csv('./data/uk_pop5x_coords.csv', dtype=['float32', 'float32'])

# %%time
dkm = KMeans(n_clusters=20)
dkm.fit(ddf)

cluster_centers = dkm.cluster_centers_
cluster_centers.columns = ddf.columns
cluster_centers.dtypes

south_idx = cluster_centers.nsmallest(1, 'northing').index[0]
labels_predicted = dkm.predict(ddf)
labels_predicted[labels_predicted==south_idx].compute().shape[0]

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)