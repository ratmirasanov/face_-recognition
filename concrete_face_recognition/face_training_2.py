"""Face training module."""


import os
import cv2
import numpy as np
from PIL import Image


def get_images_and_labels(path, detector):
    """Function to get the images and label data."""

    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:

        pil_img = Image.open(image_path).convert("L") # Convert it to grayscale.
        img_numpy = np.array(pil_img, "uint8")
        id_ = int(os.path.split(image_path)[-1].split("_")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x_1, y_1, w_1, h_1) in faces:

            face_samples.append(img_numpy[y_1 : y_1 + h_1, x_1 : x_1 + w_1])
            ids.append(id_)

    return face_samples, ids


RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()
FACE_CASCADE = cv2.CascadeClassifier(os.path.dirname(os.path.abspath(__file__)) +
                                     "/haarcascades/haarcascade_frontalface_default.xml")
print("\n [INFO] Training faces. It will take a few seconds. Wait...")
FACES, IDS = get_images_and_labels("dataset", FACE_CASCADE)
RECOGNIZER.train(FACES, np.array(IDS))
# Save the model into "trainer/trainer.yml".
RECOGNIZER.save("trainer/trainer.yml")
# Print the number of faces trained and end program.
print("\n [INFO] {0} faces trained. Exiting program...".format(len(np.unique(IDS))))
