import json
import time
import sys
import argparse
import os
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime


def init():
    global session
    model = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.onnx')
    session = onnxruntime.InferenceSession(model, None)

def preprocess(input_data_json):
    # convert the JSON data into the tensor input
    img_data = np.array(json.loads(input_data_json)['data']).astype('float32')
    return img_data

def postprocess(result):
    return softmax(np.array(result)).tolist()

def run(input_data_json):
    try:
        start = time.time()
        # load in our data which is expected as NCHW 224x224 image
        input_data = preprocess(input_data_json)
        input_name = session.get_inputs()[0].name  # get the id of the first input of the model   
        result = session.run([], {input_name: input_data})
        end = time.time()     # stop timer
        return {"result": np.array(result).tolist(),
                "time": end - start}
    except Exception as e:
        result = str(e)
        return {"error": result}
