import cv2


cap = cv2.VideoCapture('http://192.168.136.239:4747/video')

while True:
    ret,frame =cap.read()
    frame = cv2.resize(frame,(640,360))
    cv2.imshow("image",frame)

    cv2.waitKey(1)
    