import cv2
import argparse
import json
import requests
import numpy as np

print(cv2.__version__)
cap = cv2.VideoCapture("qtiqmmfsrc ldc=TRUE !video/x-raw, format=NV12, width=1280, height=720, framerate=30/1 ! videoconvert ! appsink", cv2.CAP_GSTREAMER)

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sec", required=True, help="No of sec)
args = vars(ap.parse_args())

deploy_url  = 'Paste the deploy url'
label = ['auditorium' , 'bookstore' , 'bowling','classroom','grocerystore','gym','kitchen','livingroom','mall','trainstation']
output_shape = (224, 224)


i = int(args["sec"])*30

while(i):
	if(i % 60 != 0):
		i = i -1
		continue

	ret ,frame = cap.read()
	cv2.imwrite('image'+str(i)+'.jpg',frame)
	image = cv2.resize(frame, output_shape, interpolation = cv2.INTER_AREA) 

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

cap.release()    

