"""Simple camera test."""


import cv2


CAM = cv2.VideoCapture(0)
CAM.set(3, 800)
CAM.set(4, 600)

while True:

    RET, IMG = CAM.read()
    IMG = cv2.flip(IMG, -1)
    GRAY = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
    cv2.imshow("camera", IMG)
    cv2.imshow("gray", GRAY)
    k = cv2.waitKey(30) & 0xff

    if k == 27:

        break

CAM.release()
cv2.destroyAllWindows()
