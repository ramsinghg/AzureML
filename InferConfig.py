# Azure Machine learning inference cofiguration file #

# CREATING WORKSPACE #
WORKSPACE = 'name of workspace'
SUBSCRIPTION_ID = 'paste the subscription id'
RESOURCE_GROUP = 'resource group name'
LOCATION = 'location'
TENATEID = 'paste the tenate id'

# REGISTER MODEL #
MODEL_PATH = "model/model.onnx"
MODEL_NAME = "model"
MODEL_TAGS = {"onnx": "demo"}
MODEL_DESCRIPTION = "scene_classification from ONNX Model"

# CONDA ENVIRONMENT CREATION #
PIP_PACKAGES = ["numpy", "onnxruntime", "azureml-core", "azureml-defaults"]
ENVFILE = "myenv.yml"
ENVNAME = "myenv"

# CONTAINER INSTANCE CREATION #
ACI_CPUCORES = 1
ACI_MEMORY = 1
ACI_DESCRIPTION = 'web service for model'

# MODEL DEPLOYMENT #
ACI_SERVICENAME = 'onnx-demo'

# DELETING WORKSPACE
DEL_WORKSPACE = False


