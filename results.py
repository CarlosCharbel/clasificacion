import numpy as np
from ultralytics import YOLO
import cv2


model = YOLO('./runs/classify/train3/weights/last.pt')  # load a custom model

cap = cv2.VideoCapture('http://192.168.136.239:4747/video')
threshold = 0.8
x, y = 50, 50
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1.0
color = (255, 255, 255)
thickness = 2

while(True):
    ret,frame = cap.read()
    frame = cv2.resize(frame,(640,360))

    cv2.imshow("image",frame)

    results = model.predict(frame,verbose=False)[0]
    names = results.names
    result = results.probs.top1
    conf = float(results.probs.top1conf)
    print(conf)

    if conf > threshold:
        cv2.putText(frame,names.get(result), (x, y), font, fontScale, color, thickness)
        text = names.get(result)
    else: 
        text = "None"

    cv2.putText(frame,text, (x, y), font, fontScale, color, thickness)
    cv2.imshow("image",frame)
    cv2.waitKey(1)
