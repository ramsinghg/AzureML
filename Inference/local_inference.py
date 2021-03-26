import json
import sys
import os
import time
import numpy as np
import cv2
import math
import onnx
import onnxruntime
from onnx import numpy_helper
import argparse


label = ['auditorium' , 'bookstore' , 'bowling','classroom','grocerystore','gym','kitchen','livingroom','mall','trainstation']
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

model="../model/model.onnx"

#Preprocess the image
img = cv2.imread(args["image"],-1)

img = cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_AREA)
 
#img = np.moveaxis(img, -1, 0)
img = np.expand_dims(img, axis=0)/255.0
data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

result = session.run([output_name], {input_name: data})

prediction=int(np.argmax(np.array(result).squeeze(), axis=0))
print('index is =',prediction)
print("label is = ", label[prediction])

