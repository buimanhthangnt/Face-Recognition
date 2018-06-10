import tensorflow as tf
import cv2 as cv
import numpy as np
from tensorflow.python.platform import gfile
import pickle
import imutils
import os

input_image_size = 160
threshold = 1.02

# Load pretrained facenet model
sess = tf.Session()
with gfile.FastGFile('weights/face_recog.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

images_ph = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings_ph = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings_ph.get_shape()[1]


def show_results(faces_and_poses, results, image_path):
    image = cv.imread(image_path)
    image = imutils.resize(image, width=800)
    for idx, pos in enumerate(faces_and_poses[:, 1]):
        x, y, w, h = pos
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv.putText(image, results[idx], (x+12, y+h+32), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if image.shape[0] > 800: image = imutils.resize(image, height=800)
    cv.imshow('result', image)
    cv.waitKey(0)


def recognition(faces_and_poses, image):
    embeddings = get_embeddings(faces_and_poses[:, 0])
    known_people = pickle.load(open('embeddings/known.pkl', 'rb'))
    known_embeddings = known_people[:, 0]
    results = []
    for embedding in embeddings:
        possible = []
        for idx, known in enumerate(known_embeddings):
            distance = np.sqrt(np.sum(np.square(embedding - known)))
            if distance < threshold:
                possible.append((distance, known_people[idx, 1]))
        possible = sorted(possible, key=lambda x: x[0])
        result = 'unknown' if len(possible) == 0 else possible[0][1]
        results.append(result)
    show_results(faces_and_poses, results, image)


def learn(faces_and_poses, label):
    path = 'embeddings/known.pkl'
    embeddings = get_embeddings(faces_and_poses[:, 0])
    if os.path.isfile(path):
        known_people = pickle.load(open(path, 'rb'))
        new_known_people = []
        for person in known_people:
            new_known_people.append(person)
        new_known_people.append((embeddings.flatten(), label))
        pickle.dump(np.array(new_known_people), open(path, 'wb'), pickle.HIGHEST_PROTOCOL)
    else:
        known_people = np.array([((embeddings.flatten()), label)])
        pickle.dump(known_people, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)
        


def get_embeddings(faces):
    # Preprocessing
    new_faces = []
    for face in faces:
        face = cv.resize(face, (input_image_size, input_image_size), interpolation=cv.INTER_CUBIC)
        face = prewhiten(face)
        new_faces.append(face)

    reshaped = np.array(new_faces).reshape(-1, input_image_size, input_image_size, 3)
    feed_dict = {images_ph: reshaped, phase_train_placeholder: False}
    embeddings = sess.run(embeddings_ph, feed_dict=feed_dict)
    return embeddings


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  
