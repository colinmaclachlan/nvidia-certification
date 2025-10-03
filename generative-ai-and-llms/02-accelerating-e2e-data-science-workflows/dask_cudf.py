import subprocess # we will use this to obtain our local IP using the following command
cmd = "hostname --all-ip-addresses"

process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
IPADDR = str(output.decode()).split()[0]

from dask_cuda import LocalCUDACluster
cluster = LocalCUDACluster(ip=IPADDR)

from dask.distributed import Client, progress

client = Client(cluster)

# get the file size of `pop5x_1-07.csv` in GB
!ls -sh data/uk_pop5x.csv

import dask_cudf

ddf = dask_cudf.read_csv('./data/uk_pop5x.csv', dtype=['float32', 'str', 'str', 'float32', 'float32', 'str'])

ddf.dtypes

!nvidia-smi

ddf.visualize(format='svg') # This visualization is very large, and using `format='svg'` will make it easier to view.

ddf.npartitions

mean_age = ddf['age'].sum()
mean_age.visualize(format='svg')

mean_age.compute()

!nvidia-smi

ddf = ddf.persist()

!nvidia-smi

ddf.visualize(format='svg')

ddf['age'].mean().compute()

ddf.head() # As a convenience, no need to `.compute` the `head()` method

ddf.count().compute()

ddf.dtypes

sunderland_residents = ddf.loc[ddf['county'] == 'Sunderland']
northmost_sunderland_lat = sunderland_residents['lat'].max()
counties_with_pop_north_of = ddf.loc[ddf['lat'] > northmost_sunderland_lat]['county'].unique()
results=counties_with_pop_north_of.compute()
results.head()

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)