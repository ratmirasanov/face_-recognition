import numpy as np
import cv2

import config


# Multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier(config.PATH_TO_FILE + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
cap.set(3, 800) # Set Width.
cap.set(4, 600) # Set Height.

while True:

    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

    cv2.imshow("face_detection", img)

    k = cv2.waitKey(30) & 0xff

    if k == 27: # Press "ESC" to quit.

        break

cap.release()
cv2.destroyAllWindows()
