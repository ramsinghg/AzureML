

# Azure MODEL depoyment 

import InferConfig as infer
from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.webservice import AciWebservice
from random import randint
from azureml.core.authentication import InteractiveLoginAuthentication

# create a workspace

interactive_auth = InteractiveLoginAuthentication(tenant_id=infer.TENATEID, force = True)
ws = Workspace.create(name = infer.WORKSPACE ,                       # provide a name for your workspace
                      subscription_id = infer.SUBSCRIPTION_ID,       # provide your subscription ID
                      resource_group = infer.RESOURCE_GROUP,         # provide a resource group name
                      create_resource_group = True,
                      location = infer.LOCATION,
            		  auth=interactive_auth)                     # For example: 'westeurope' or 'eastus2' or 'westus2' or 'southeastasia'. 

ws.write_config(path='.azureml')

# read workspace from config file
ws = Workspace.from_config()

# register the model 
model = Model.register(model_path = infer.MODEL_PATH,
                       model_name = infer.MODEL_NAME,
                       tags = infer.MODEL_TAGS,
                       description = infer.MODEL_DESCRIPTION,
                       workspace = ws)

# conda environment creation
myenv = CondaDependencies.create(pip_packages = infer.PIP_PACKAGES)

with open(infer.ENVFILE ,"w") as f:
    f.write(myenv.serialize_to_string())

myenv = Environment.from_conda_specification(name = infer.ENVNAME, file_path = infer.ENVFILE)
inference_config = InferenceConfig(entry_script="score.py", environment=myenv)


# deploy the model on azure container instance
aciconfig = AciWebservice.deploy_configuration(cpu_cores = infer.ACI_CPUCORES, 
                                               memory_gb = infer.ACI_MEMORY, 
                                               tags = infer.MODEL_TAGS, 
                                               description = infer.ACI_DESCRIPTION)

aci_service_name = infer.ACI_SERVICENAME
print("Service", aci_service_name)
aci_service = Model.deploy(ws, aci_service_name, [model], inference_config, aciconfig)
aci_service.wait_for_deployment(True)
print(aci_service.state)


if aci_service.state != 'Healthy':
    print(aci_service.get_logs())
    aci_service.delete()


print(aci_service.scoring_uri)

if(infer.DEL_WORKSPACE):
    aci_service.delete() 

