from __future__ import absolute_import, division, print_function

import tensorflow as tf
from keras.models import Sequential
import argparse
import facenet
import imutils
import os
from datetime import datetime
import pickle
import detect_face 
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import time
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime
import csv
from gtts import gTTS
import playsound

# Initialize Firebase
json_file_path = "C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\FaceRecog-facenet\\face-attendance-manage-firebase-adminsdk-lc8o9-4c94711f6b.json"
cred = credentials.Certificate(json_file_path)
firebase_config = {
    'databaseURL': "https://face-attendance-manage-default-rtdb.firebaseio.com/",
    'storageBucket': "face-attendance-manage.appspot.com"
}
firebase_app = firebase_admin.initialize_app(cred, firebase_config)

bucket = storage.bucket()
modeType = 0
counter = 0
COL_NAMES = ['MSSV', 'NAME', 'TIME-IN', 'ADDRESS-IN', 'TIME-OUT', 'ADDRESS-OUT', 'PHONE']
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = 'temp.mp3'
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

def read_csv(file_path):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

def update_or_add_data(data, mssv, name, in_time, in_address, out_time, out_address, phone):
    updated = False
    for row in data:
        if row[0] == mssv:
            row[4] = out_time
            row[5] = out_address
            updated = True
            break
    if not updated:
        new_row = [mssv, name, in_time, in_address, out_time, out_address, phone]
        data.append(new_row)

def write_csv(file_path, data):
    with open(file_path, 'w', newline='\n', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', help='Path to video file', default='video/video1.mp4')
    args = parser.parse_args()

    # Load the classifier, class names, MTCNN model, and TensorFlow session
    classifier_path = 'C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\FaceRecog-facenet\\Models\\facemodel.pkl'
    facenet_model_path = 'C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\FaceRecog-facenet\\Models\\20180402-114759.pb'

    with open(classifier_path, 'rb') as infile:
        (model, class_names) = pickle.load(infile)
    print("Classifier loaded")

    with tf.compat.v1.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        with sess.as_default():
            facenet.load_model(facenet_model_path)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            video_capture = cv2.VideoCapture(args.video_path)
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    print("Unable to fetch frame")
                    break
                pnet, rnet, onet = detect_face.create_mtcnn(sess, "C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\FaceRecog-facenet\\src")
                if len(faces) > 0:
                    # Process each face detected
                    for face in faces:
                        x1, y1, x2, y2 = [int(pos) for pos in face[:4]]
                        cropped = frame[y1:y2, x1:x2]
                        scaled = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)
                        feed_dict = {images_placeholder: [scaled], phase_train_placeholder: False}
                        embedding = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(embedding)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                        # Check if the prediction is confident enough
                        if best_class_probabilities > 0.5:
                            text = f'{class_names[best_class_indices[0]]}: {best_class_probabilities[0]:.2%}'
                            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 255, 0), 1)
                            print("Detected:", text)
                cv2.imshow('Video', frame)

                # Press 'q' to quit the video stream
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
