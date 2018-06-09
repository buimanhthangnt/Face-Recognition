import face_detection
import face_recognition
import numpy as np
import pickle
import os


folder = 'images'
for image in os.listdir(folder):
    path = '/'.join([folder, image])
    faces = face_detection.detect(path)
    faces_with_name = face_recognition.recognition(faces, path)