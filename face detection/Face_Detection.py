import cv2
from random import randrange
print("face detection project")

#loading the pre trained face detcted data form opencv
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect the face 
#image = cv2.imread('all of them.jpg')
webcam=cv2.VideoCapture(0)
while True:
    successfull_frame_read, frame = webcam.read()

    # turning the image into gray that is grayscale
    grayimage=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect image
    faceCorordinates= trained_face_data.detectMultiScale(grayimage)

    #rectangle on face using loop
    for (x, y, w, h) in faceCorordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 5)

    cv2.imshow('Friends cast', frame)
    key=cv2.waitKey(1)
    
    #to exit from the loop
    if key==81 or key==113:
        break
webcam.release()