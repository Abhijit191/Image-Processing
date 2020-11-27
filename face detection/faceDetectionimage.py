import cv2
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('all of them 2.jpg')

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faceCoordinate = trained_face_data.detectMultiScale(gray_image, 1.3)

for (x, y, w, h) in faceCoordinate:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)

cv2.imshow('Friends', img)

cv2.waitKey()
