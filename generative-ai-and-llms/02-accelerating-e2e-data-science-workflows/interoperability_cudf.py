import cudf
import time

num_ele = 1000000

df = cudf.DataFrame(
    {
        "a": range(num_ele),
        "b": range(10, num_ele + 10),
        "c": range(100, num_ele + 100),
        "d": range(1000, num_ele + 1000)
    }
)

# preview
df.head()

start=time.time()
display(df.sum(axis=1))
time.time()-start

arr=df.values

start=time.time()
# alternative approach
# arr=df.to_cupy()

display(arr.sum(axis=1))
time.time()-start

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)