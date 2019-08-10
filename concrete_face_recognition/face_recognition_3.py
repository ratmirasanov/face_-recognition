"""Face recognition module."""


import os
import cv2


RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()
RECOGNIZER.read("trainer/trainer.yml")
FACE_CASCADE = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(__file__)) +
                                     "/haarcascades/haarcascade_frontalface_default.xml")
FONT = cv2.FONT_HERSHEY_SIMPLEX
ID_ = 0
NAMES = ["None", "Ratmir", "Melek"]
CAM = cv2.VideoCapture(0)
CAM.set(3, 800)
CAM.set(4, 600)
MINW = 0.1 * CAM.get(3)
MINH = 0.1 * CAM.get(4)

while True:

    RET, IMG = CAM.read()
    IMG = cv2.flip(IMG, 1)
    GRAY = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
    FACES = FACE_CASCADE.detectMultiScale(
        GRAY,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(MINW), int(MINH)),
    )

    for (x, y, w, h) in FACES:

        cv2.rectangle(IMG, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id_, confidence = RECOGNIZER.predict(GRAY[y : y + h, x : x + w])

        if confidence < 100:

            id_ = NAMES[id_]
            confidence = "  {0}%".format(round(100 - confidence))

        else:

            id_ = "Unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(IMG, str(id_), (x + 5, y - 5), FONT, 1, (255, 255, 255), 2)
        cv2.putText(IMG, str(confidence), (x + 5, y + h - 5), FONT, 1, (255, 255, 0), 1)

    cv2.imshow("face_recognition", IMG)
    k = cv2.waitKey(10) & 0xff

    if k == 27:

        break

print("\n [INFO] Exiting program and cleanup stuff...")
CAM.release()
cv2.destroyAllWindows()
