import numpy as np 
import cv2

face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)

en_no = input("enter user number")
data_no = 0

while True:
	ret, frame = cam.read()
	gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_detect.detectMultiScale(gray_img, 1.2, 5)

	for (x, y, w, h) in faces:
		data_no += 1
		cv2.imwrite('dataset/data' + str(en_no) + "." + str(data_no) + ".jpg", gray_img[y:y+h, x:x+w] )
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
		cv2.waitKey(100)

	cv2.imshow("frame", frame)
	cv2.waitKey(1)

	if data_no > 10:
		break

cam.release()
cv2.destroyAllWindows()
