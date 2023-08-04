import numpy as np
from ultralytics import YOLO
import cv2


model = YOLO('./runs/classify/train3/weights/last.pt')  # load a custom model


cap = cv2.VideoCapture('http://192.168.136.239:4747/video')


ret,frame = cap.read()
frame = cv2.resize(frame,(640,360))

threshold = 0.8

while(True):
    cv2.imshow("image",frame)

    if cv2.waitKey(1) & 0xFF == ord('y'):
        results = model.predict(frame,verbose=False)[0]
        names = results.names
        result = results.probs.top1
        conf = float(results.probs.top1conf)
        print(conf)
        if conf > threshold:
            print('Se encontró: ',names.get(result))
        break


cv2.destroyAllWindows()
cap.release()