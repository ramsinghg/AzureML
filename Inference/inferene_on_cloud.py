import cv2
import argparse
import json
import requests
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

label = ['auditorium' , 'bookstore' , 'bowling','classroom','grocerystore','gym','kitchen','livingroom','mall','trainstation']

deploy_url  = "paste the deploy url"

img = cv2.imread(args["image"],-1)

img = cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_AREA)
 
#img = np.moveaxis(img, -1, 0)
img = np.expand_dims(img, axis=0)/255.0
data = json.dumps({'data': img.tolist()})

headers = {'Content-Type':'application/json'} 
resp = requests.post(deploy_url, data, headers=headers)

print("prediction time (as measured by the scoring container)", json.loads(resp.text)["time"])

response=json.loads(resp.text)
result=np.array(response["result"])

index = np.argmax(result)
print('index is =',index)
print("label is = ", label[index])

