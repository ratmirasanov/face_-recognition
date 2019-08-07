import os
import cv2


face_cascade = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(__file__)) +
                                     "/haarcascades/haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
cam.set(3, 800) # Set width.
cam.set(4, 600) # Set height
# For each person, enter one numeric face ID.
face_id = input("\n Enter user ID end press 'ENTER': ")
print("\n [INFO] Initializing face capture. Look the camera and wait...")
# Initialize individual sampling face count.
count = 0

while True:

    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        # Save the captured image into the "dataset" folder.
        cv2.imwrite("dataset/user_" + str(face_id) + "_" + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow("face_dataset", img)

    k = cv2.waitKey(100) & 0xff

    if k == 27: # Press "ESC" to quit.

        break

    elif count >= 15: # Take 15 face sample and stop video.

        break

# Do a bit of cleanup.
print("\n [INFO] Exiting program and cleanup stuff...")
cam.release()
cv2.destroyAllWindows()
