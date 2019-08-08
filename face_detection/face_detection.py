"""Face detection module."""


import os
import cv2


FACE_CASCADE = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(__file__)) +
                                     "/haarcascades/haarcascade_frontalface_default.xml")
CAM = cv2.VideoCapture(0)
CAM.set(3, 800) # Set width.
CAM.set(4, 600) # Set height.

while True:

    RET, IMG = CAM.read()
    IMG = cv2.flip(IMG, 1)
    GRAY = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
    FACES = FACE_CASCADE.detectMultiScale(
        GRAY,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x, y, w, h) in FACES:

        cv2.rectangle(IMG, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = GRAY[y : y + h, x : x + w]
        roi_color = IMG[y : y + h, x : x + w]

    cv2.imshow("face_detection", IMG)
    k = cv2.waitKey(30) & 0xff

    if k == 27: # Press "ESC" to quit.

        break

CAM.release()
cv2.destroyAllWindows()
