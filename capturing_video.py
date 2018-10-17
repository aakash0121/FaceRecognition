import numpy as np 
import cv2

cam = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 320))

while True:
	ret, frame = cam.read()
	gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cv2.imshow("gray_img", gray_img)
	out.write(frame)
	if cv2.waitKey(1):
		break

cam.release()
out.release()
cv2.destroyAllWindows()
