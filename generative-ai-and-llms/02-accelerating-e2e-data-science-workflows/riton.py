import xgboost as xgb
model = xgb.Booster({'nthread': 4})  # init model
model.load_model('xgboost_model.json')  # load model data

import os

# Create the model repository directory. The name of this directory is arbitrary.
REPO_PATH = os.path.abspath('models')
os.makedirs(REPO_PATH, exist_ok=True)

# The name of the model directory determines the name of the model as reported by Triton
model_dir = os.path.join(REPO_PATH, "virus_prediction")

# We can store multiple versions of the model in the same directory. In our case, we have just one version, so we will add a single directory, named '1'.
version_dir = os.path.join(model_dir, '1')
os.makedirs(version_dir, exist_ok=True)

# The default filename for XGBoost models saved in json format is 'xgboost.json'.
# It is recommended that you use this filename to avoid having to specify a name in the configuration file.
model_file = os.path.join(version_dir, 'xgboost.json')
model.save_model(model_file)

config_text = f"""backend: "fil"
max_batch_size: {32768}
input [                                 
 {{  
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 4 ]                    
  }} 
]
output [
 {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]
instance_group [{{ kind: KIND_GPU }}]
parameters [
  {{
    key: "model_type"
    value: {{ string_value: "xgboost_json" }}
  }},
  {{
    key: "output_class"
    value: {{ string_value: "false" }}
  }},
  {{
    key: "storage_type"
    value: {{ string_value: "AUTO" }}
  }}
]

dynamic_batching {{
  max_queue_delay_microseconds: 100
}}"""
config_path = os.path.join(model_dir, 'config.pbtxt')
with open(config_path, 'w') as file_:
    file_.write(config_text)

# !curl -v triton:8000/v2/health/ready
# !curl -X POST http://triton:8000/v2/repository/index

import time
import tritonclient.grpc as triton_grpc
from tritonclient import utils as triton_utils
HOST = "triton"
PORT = 8001
TIMEOUT = 60

client = triton_grpc.InferenceServerClient(url=f'{HOST}:{PORT}')

import cudf 
import numpy as np
df = cudf.read_csv('./data/clean_uk_pop_full.csv', usecols=['age', 'sex', 'northing', 'easting', 'infected'], nrows=5000000)
df = df.sample(32768)
input_data = df.drop('infected', axis=1)
target = df[['infected']]
print(target)

converted_df = input_data.to_numpy(dtype='float32')

# %%time
batched_data = converted_df[:32768]
# Prepare the input tensor
input_tensor = triton_grpc.InferInput("input__0", batched_data.shape, 'FP32')
input_tensor.set_data_from_numpy(batched_data)

# Prepare the output
output = triton_grpc.InferRequestedOutput("output__0")

# Send inference request
response = client.infer("virus_prediction", [input_tensor], outputs=[output])

# Get the output data
output_data = response.as_numpy("output__0")

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

xgb_data = xgb.DMatrix(input_data)
y_test = model.predict(xgb_data)

# Check that we got the same accuracy as previously
#target = target.to_numpy()
import matplotlib.pyplot as plt

false_pos_rate, true_pos_rate, thresholds = roc_curve(target.to_numpy(), y_test)
auc_result = auc(false_pos_rate, true_pos_rate)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(false_pos_rate, true_pos_rate, lw=3,
        label='AUC = {:.2f}'.format(auc_result))
ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set(
    xlim=(0, 1),
    ylim=(0, 1),
    title="ROC Curve",
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
)
ax.legend(loc='lower right');
plt.show()

# Check that we got the same accuracy as previously
#target = target.to_numpy()
import matplotlib.pyplot as plt

false_pos_rate, true_pos_rate, thresholds = roc_curve(target.to_numpy(), output_data)
auc_result = auc(false_pos_rate, true_pos_rate)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(false_pos_rate, true_pos_rate, lw=3,
        label='AUC = {:.2f}'.format(auc_result))
ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set(
    xlim=(0, 1),
    ylim=(0, 1),
    title="ROC Curve",
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
)
ax.legend(loc='lower right');
plt.show()

!perf_analyzer -m virus_prediction -u "triton:8000"
!perf_analyzer --collect-metrics -m virus_prediction -u "triton:8000" -b 8 --concurrency-range 2:8:2