"""Simple camera test."""


import cv2


CAM = cv2.VideoCapture(0)
CAM.set(3, 640) # Set width.
CAM.set(4, 480) # Set height.

while True:

    RET, IMG = CAM.read()
    IMG = cv2.flip(IMG, -1) # Flip camera vertically.
    GRAY = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
    cv2.imshow("camera", IMG)
    cv2.imshow("gray", GRAY)
    k = cv2.waitKey(30) & 0xff

    if k == 27: # Press "ESC" to quit.

        break

CAM.release()
cv2.destroyAllWindows()
