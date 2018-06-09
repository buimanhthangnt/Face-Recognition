import numpy as np
import cv2 as cv
import imutils


def detect(image_path):
    face_cascade = cv.CascadeClassifier('weights/harr_cascade.xml')
    image = cv.imread(image_path)
    image = imutils.resize(image, width=800)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 10)
    ret = []
    for (x, y, w, h) in faces:
        x, y, w, h = add_padding(x, y, w, h, image.shape[:2])
        ret.append((image[y:y+h, x:x+w], (x, y, w, h)))
    return np.array(ret)


def add_padding(x, y, w, h, image_shape):
    margin_x = w // 5
    margin_y = h // 3
    x = max(x - margin_x, 0)
    y = max(y - margin_y, 0)
    w += 2 * margin_x
    h += 2 * margin_y
    w = min(image_shape[1] - x - 1, w)
    h = min(image_shape[0] - y - 1, h)
    return x, y, w, h