import numpy as np 
import cv2
import urllib

url = "http://192.168.43.1:8080/shot.jpg"

detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while True:
    img_Resp = urllib.request.urlopen(url)
    imgNP = np.array(bytearray(img_Resp.read()), dtype = np.uint8)
    Img = cv2.imdecode(imgNP, -1)
    ret, img = Img.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
