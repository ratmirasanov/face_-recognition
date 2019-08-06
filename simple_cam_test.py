import numpy as np
import cv2


cap = cv2.VideoCapture(0)
cap.set(3, 640) # Set Width.
cap.set(4, 480) # Set Height.

while True:

    ret, camera = cap.read()
    camera = cv2.flip(camera, -1) # Flip camera vertically.
    gray = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)

    cv2.imshow("camera", camera)
    cv2.imshow("gray", gray)

    k = cv2.waitKey(30) & 0xff

    if k == 27: # Press "ESC" to quit.

        break

cap.release()
cv2.destroyAllWindows()
