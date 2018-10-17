import numpy as np 
import cv2

cam = cv2.VideoCapture(0)

face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
rec = cv2.createLBPHFaceRecogniser()

rec.load("recogniser/training_data.yml")

en_no = 0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL, 5, 1, 0, 4)

while True:
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_detect.detectMultiScale(gray, 1.3, 5)

	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)

		id, conf = rec.predict(gray[y:y+h, x:x+w])
		if id == 1:
			id = "Aakash"
		cv2.cv.PutText(cv2.cv.fromarray(img), str(id), (x, y+h), 255)
		cv2.imshow("face", img)

		if cv2.waitKey(1) == ord("q"):
			break

cam.release()
cv2.destroyAllWindows()