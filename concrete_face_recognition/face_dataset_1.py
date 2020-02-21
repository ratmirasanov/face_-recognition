"""Face dataset creation module."""


import os
import cv2


FACE_CASCADE = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(__file__)) +
                                     "/haarcascades/haarcascade_frontalface_default.xml")
CAM = cv2.VideoCapture(0)
CAM.set(3, 800)
CAM.set(4, 600)
FACE_ID = input("\n Enter user ID end press 'ENTER': ")
print("\n [INFO] Initializing face capture. Look the camera and wait...")
COUNT = 0

while True:

    RET, IMG = CAM.read()
    IMG = cv2.flip(IMG, 1)
    GRAY = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
    FACES = FACE_CASCADE.detectMultiScale(GRAY, 1.3, 5)

    for (x, y, w, h) in FACES:

        cv2.rectangle(IMG, (x, y), (x + w, y + h), (255, 0, 0), 2)
        COUNT += 1
        cv2.imwrite(os.path.dirname(os.path.abspath(__file__)) +
                    "/dataset/user_" + str(FACE_ID) + "_" + str(COUNT) + ".jpg", GRAY[y:y+h, x:x+w])
        cv2.namedWindow("face_dataset", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("face_dataset", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("face_dataset", IMG)

    k = cv2.waitKey(100) & 0xff

    if k == 27:

        break

    elif COUNT >= 40:

        break

print("\n [INFO] Exiting program and cleanup stuff...")
CAM.release()
cv2.destroyAllWindows()
