from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from imutils.video import VideoStream
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
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime
import csv
from win32com.client import Dispatch

# cred = credentials.Certificate("face-attendance-manage-firebase-adminsdk-lc8o9-4c94711f6b.json")
# firebase_admin.initialize_app(cred,{
#     'databaseURL':"https://face-attendance-manage-default-rtdb.firebaseio.com/",
#     'storageBucket': "face-attendance-manage.appspot.com"
# })
cred = credentials.Certificate("C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\face-detect-71d15-firebase-adminsdk-du0ib-d96ef84311.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://face-detect-71d15-default-rtdb.firebaseio.com/",
    'storageBucket': "gs://face-detect-71d15.appspot.com"
})
 
bucket = storage.bucket()
modeType = 0
counter = 0
COL_NAMES = ['MSSV','NAME', 'TIME-IN','ADDRESS-IN','TIME-OUT','ADDRESS-OUT','PHONE']
ts=time.time()
date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)
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
        if row[0] == mssv:  # So sánh theo MSSV hoặc thuộc tính duy nhất khác
            # Nếu người đã có trong dữ liệu, cập nhật thông tin
            # row[2] = {in_time}
            #row[3] = in_address
            row[4] = out_time
            row[5] = out_address
            # row[6] = phone
            updated = True
            break
    if not updated:
        # Nếu người chưa có trong dữ liệu, thêm thông tin mới
        new_row = [mssv, name, in_time, in_address, out_time, out_address, phone]
        data.append(new_row)
def write_csv(file_path, data):
    with open(file_path, 'w', newline='\n', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Models\\facemodel5.pkl'
    #VIDEO_PATH = args.path
   #VIDEO_PATH = 'C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\FaceRecog-facenet\\video\\video5\\ronaldo.mp4'
    FACENET_MODEL_PATH = 'C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Models\\20180402-114759.pb'
    Attendance_path = 'C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Attendance'
    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        # Cai dat GPU neu co
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
        
            pnet, rnet, onet = detect_face.create_mtcnn(sess, "C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\src")

            people_detected = set()
            person_detected = collections.Counter()

            cap  = VideoStream(src=0).start()
            

            while (True):
                frame = cap.read()
                frame = imutils.resize(frame, width=600)
                frame = cv2.flip(frame, 1)

                bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 1:
                        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                    elif faces_found == 1:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            #print(bb[i][3]-bb[i][1])
                            #print(frame.shape[0])
                            #print((bb[i][3]-bb[i][1])/frame.shape[0])
                            if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25:
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled = facenet.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))



                                if best_class_probabilities > 0.8:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    ts=time.time()
                                    date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                                    timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
                                    name = class_names[best_class_indices[0]]
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    studentInfo = db.reference(f'Students/{name}').get()   
                                    datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S") 
                                    secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                                    csv_file_path = "C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Attendance\\Attendance_" +str(date) +"check-in.csv"
                                    print("time:",secondsElapsed)
                                    if secondsElapsed > 7200:           # lớn hơn 2 tiếng
                                        ref = db.reference(f'Students/{name}')                                    
                                        ref.update({'total_attendance': studentInfo['total_attendance'],
                                        'last_attendance_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                                        k1='changes'
                                        
                                        change=db.reference(f'Students/{k1}')
                                        change.update({'name':studentInfo['name'], 'status': '1' })
                                        attendance=[str(studentInfo['id']),str(studentInfo['name']), str(timestamp),str(studentInfo['addrSchool']),str(None),str(None), str(studentInfo['phone'] )]
                                        print(studentInfo)
                                        print(attendance)

                                        exist=os.path.isfile("C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Attendance\\Attendance_" + date + "check-in" ".csv")

                                        if exist:
                                            datacsv = read_csv(csv_file_path)
                                            if  str(studentInfo['id']) not in set(row['MSSV'] for row in datacsv):                                                
                                                with open("C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Attendance\\Attendance_" + date + "check-in" + ".csv", "a",newline='\n') as csvfile:
                                                    print("ten hoc hinh da duoc ghi nhan ")
                                                    writer=csv.writer(csvfile)
                                                    writer.writerow(attendance)
                                                csvfile.close()
                                        else:
                                            print("tao file ghi du lieu moi")   
                                            with open("C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Attendance\\Attendance_" + date + "check-in" + ".csv", "a",newline='\n') as csvfile:
                                                writer=csv.writer(csvfile)
                                                writer.writerow(COL_NAMES)
                                                writer.writerow(attendance)
                                            csvfile.close()
                                    elif 3600 > secondsElapsed > 20:
                                        ref = db.reference(f'Students/{name}')                                    
                                        studentInfo['total_attendance'] += 1
                                        ref.update({'total_attendance': studentInfo['total_attendance'],
                                            'last_attendance_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                                        #attendance=[str(studentInfo['id']),str(studentInfo['name']), str(timestamp),str(studentInfo['address']), str(studentInfo['phone'] )]
                                        attendance=[str(studentInfo['id']),str(studentInfo['name']), str(timestamp),str(studentInfo['addrSchool']),str(None),str(None), str(studentInfo['phone'] )]
                                        
                                        print(studentInfo)
                                        print(attendance)

                                            # Đọc dữ liệu hiện có từ tệp CSV
                                        datacsv = read_csv(csv_file_path)

                                            # Gọi hàm để cập nhật hoặc thêm dữ liệu mới
                                        update_or_add_data(datacsv ,str(studentInfo['id']),str(studentInfo['name']), str(timestamp),str(studentInfo['addrSchool']),str(timestamp),str(studentInfo['address']), str(studentInfo['phone']))
                                        write_csv(csv_file_path, datacsv)



                                                       
                                    person_detected[best_name] += 1
                                else:
                                    name = "Unknown"
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (2, 255, 2), thickness=1, lineType=2)
                                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)

                except:
                    pass
                frame = cv2.resize(frame, (0, 0), None, 4, 4)
                

                frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # cap.release()
            cv2.destroyAllWindows()


main()
