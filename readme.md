# Demonstration of Microsoft Azure Machine Learning and Azure services
   -  In this project, we have demonstrated the azure machine learning capability, for this, we have demonstrated a pipeline, we are doing deployment of the machine learning model and doing inferencing on the cloud. Where we take the camera image data from qcs610 and perform inference and display the output on the device.         

## Dependencies
- Ubuntu System 18.04 or above
- Install Adb tool (Android debugging bridge) on host system
- valid microsoft azure user account
- Install python3.5 or above on the host system 

## python3 requiremnts
 - Following pyhton3 packages are needed, make sure, these libraries are available on the target board
      - idna 2.8
      - certifi 2019.11.2
      - urllib3 1.25.0
      - requests 2.22.0
      - chardet 3.0.4
      - six 1.14.0
  

## Prerequisites
- Camera Environment configuration setup on the target device.
- Install opencv library on board 
- Creating a resource group on azure.


### Camera Environment configuration setup on the target device.
   - To setup the camera environment configuration follow the below document 
'Turbox-C610_Open_Kit_Software_User_Manual_LE1.0_v2.0.pdf' In given url 
“https://www.thundercomm.com/app_en/product/1593776185472315” and Refer section 2.10.1

### Install opencv library on board 
   -   To install opencv library on the target board, the required meta recipes for opencv is already present in folder “poky/meta-openembedded/meta-oe/ recipes-support/opencv/opencv_3.4.5.bb” file. we need to follow the below steps to build.

 - Get into the yocto working directory
    ```sh
      $ cd  <yocto working directory>
    ```

 - Execute source command for environment setting 
   ```sh
     $ source poky/qti-conf/set_bb_env.sh
   ```
   - The pop up menu will be open for available machines in that select “qcs610-odk” and press ok. Then one more pop up window will be open for distribution selection in that we need to select “qti-distro-fullstack-perf”  

  - Run the bitbake command for installing packages.
  ```
    $ bitbake opencv 

  ```  
 - Once the build is complete the shared library and include file will be available in “./tmp-glibc/sysroots-components/armv7ahf-neon/opencv/usr”
Push the opencv shared library to the target board 
 ```
   $ cd  ./tmp-glibc/sysroots-components/armv7ahf-neon/opencv/usr/
   $ adb push lib/  /data/azureml/
 ``` 
 **Note**: 
  - For more reference refer to the “QCS610/QCS410 Linux Platform Development Kit Quick Start Guide document”. 
  - Also make sure install the all the dependency library from the yocto build to the system (ex: libgphoto2, libv4l-utils) 
  - bb recipes of above  library are available inside meta-openembedded layer you can directly run bitbake command


### Setting up Resource group 
   - To create a new resource group on azure please follow the below link      
https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal

 **Note**: once the setup is done , note down the subscription id.

### Deploy the model as service:
   - To deploy the machine learning model on the cloud, first we need to create a workspace and then register the model in the model registry and finally deploy the model in either an azure container instance  or azure kubernetes service. Once we complete the deployment it will generate inference url or rest api using this we can do inference.
  
   - First we need to convert our tensorflow model into onnx model, for this we need to run the script below. First we clone the repository
 
   ```
      $ git clone <source repository
      $ cd  <source repository> /custom_model/
      $ python3 convert_tf_onnx.py --input tfmodel/ --output model/model.onnx
   ```
- In this project we have provided inference pipelines of custom scene classification model. Inference.py and score.py may vary according to the model requirement.
 
- Next to deploy the model, we need to fill the required details in the inferConfig.py file, require details includes  

   - Subscription ID              : provide your subscription id
   - Resource group name          : provide the resource group name
   - Workspace name               : provide the name of workspace you want to create
   - Tenant ID                    : provide your tenant id
   - Model path and model details : provide the local path of model and model details 
   - Conda environment details    : provide conda env details and installation packages, 
   - Instance details             : provide compute service name,   

- **Note** : To run the application, you may require to provide details of subscription id, resource group name and tenant ID,  other details are already present in the config file.

 - After filling configuration details, we need to run the ‘deploy-model.py’ python script on the host system for deploying the model. It may take 5-10 minute for deployment.
    ```  
        $ python3 deploy_model.py
    ``` 
- Once the deployment is completed, it will show the inference url on the screen. open the inference.py file and fill the inference url details.
         
### Steps to run the application: 

 **Step-1** : initialize the target board with root access.
   ```
         $ adb root
         $ adb remount 
         $ adb shell  mount -o remount,rw /
   ```
**Step-2** : Push the application python file and shared library to the target board with adb command.
   ```   $ cd < source repository >
         $ cd  azureml/azure_inference_custom/      
         $ adb push inference.py /data/azureml/
   ```

**Step-3** :   To start the application, run the below commands on the qcs610 board, 
   ```       
         $ adb shell
         /#
   ```
  - To enable wifi connectivity 
    ```
         /# wpa_supplicant -Dnl80211 -iwlan0 -c /etc/misc/wifi/wpa_supplicant.conf -ddddt &
         /# dhcpcd wlan0
    ```      
   - Export the shared library to the LD_LIBRARY_PATH
     ```
         /# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/azure/lib/
         /# cd data/azure/
     ```    
   -  Run the inference script with the command line option as the number of sec applications need to run. 
      ``` 
          python3 inference.py <no of seconds>
        /# python3 inference.py 10
      ```  
  - The python script will capture the video from gstreamer plugin using opencv api. It will do inference on every two seconds and display the inference output on the terminal. Python script closes the application after 10 sec.

  - **Note**:
    - After your work is complete, please makesure that you have deleted all the instance, resource, deployment model, endpoint to avoid unnecessary charge.  
