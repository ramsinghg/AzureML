from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
import argparse
import onnx
import keras2onnx

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to tensorflow model")
ap.add_argument("-o", "--output", required=True, help="path of onnx model")
args = vars(ap.parse_args())

onnx_model_name = args['output']

model = load_model(args['input'])
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, onnx_model_name)
