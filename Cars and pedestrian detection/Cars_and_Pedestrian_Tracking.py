import cv2

# For both cars and pedestrian
video=cv2.VideoCapture("Dashcam predestrian.mp4")

# pre-trained car and predestrian classifiers
car_tracker_haarfile = "cars_haar_data.xml"
predestrian_tracker_haarfile = "pedestrian_haar_data.xml"

# create a classsifier for both c and p
car_tracker = cv2.CascadeClassifier(car_tracker_haarfile)
pedestrian_tracker = cv2.CascadeClassifier(predestrian_tracker_haarfile)

while True:
    (read_successful, frame) = video.read()

    if read_successful:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars
    cars = car_tracker.detectMultiScale(grayscale_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscale_frame)

    # To show the red rectangle around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1,y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)

    # To show the yellow rectangle around the pedestrian
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)


    # Display the cars
    cv2.imshow("Self Driving Car", frame)

    # Wait until any key is being pressed
    key = cv2.waitKey(1)
    
    #to exit from the loop
    if key==81 or key==113:
        break

#Release the VideoCapture object
video.release()


