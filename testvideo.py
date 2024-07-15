from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import facenet
import os
import sys
import math
import pickle
import detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import ctypes
import time

# # Äá»‹nh nghÄ©a cÃ¡c kiá»ƒu dá»¯ liá»‡u cáº§n thiáº¿t
# c_void_p = ctypes.c_void_p
# c_size_t = ctypes.c_size_t
# c_int = ctypes.c_int
# c_off_t = ctypes.c_long

# # Load thÆ° viá»‡n C
# libc = ctypes.CDLL(None)

# # Äá»‹nh nghÄ©a prototype cho hÃ m mmap
# mmap_prototype = ctypes.CFUNCTYPE(c_void_p, c_void_p, c_size_t, c_int, c_int, c_int, c_off_t)

# # Äá»‹nh nghÄ©a hÃ m mmap tá»« thÆ° viá»‡n C
# mmap_func = mmap_prototype(('mmap', libc), (
#     (1, 'addr'),
#     (1, 'length'),
#     (1, 'prot'),
#     (1, 'flags'),
#     (1, 'fd'),
#     (1, 'offset')
# ))

# def mmap(address, length, protect, flags, filedes, offset):
#     # Gá»i hÃ m mmap
#     memory_map = mmap_func(address, length, protect, flags, filedes, offset)

#     # Kiá»ƒm tra káº¿t quáº£
#     if memory_map != -1:
#         print("Memory mapped successfully.")
#     else:
#         print("Failed to memory map.")
    
#     return memory_map




def main():
    frame_count = 0
    start_time = time.time()

    # address_virtual_base = None
    # length_virtual_base = 1024
    # protect_virtual_base = 1
    # flags_virtual_base = 1
    # filedes_virtual_base = 0  # Äáº·t file descriptor tÃ¹y thuá»™c vÃ o há»‡ thá»‘ng cá»§a báº¡n
    # offset_virtual_base = 0
    # memory_map_virtual_base = mmap(address_virtual_base, length_virtual_base, protect_virtual_base, flags_virtual_base, filedes_virtual_base, offset_virtual_base)



    parser = argparse.ArgumentParser()
    parser.add_argument( '--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()
    
    # Cai dat cac tham so can thiet
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\FaceRecog-facenet\\Models\\facemodel3.pkl'
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = 'C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\FaceRecog-facenet\\Models\\20180402-114759.pb'

    # Load model da train de nhan dien khuon mat - thuc chat la classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        # Cai dat GPU neu co
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load model MTCNN phat hien khuon mat
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Lay tensor input va output
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Cai dat cac mang con
            pnet, rnet, onet = detect_face.create_mtcnn(sess, "C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\FaceRecog-facenet\\src")

            people_detected = set()
            person_detected = collections.Counter()

            # Lay hinh anh tu file video
            cap = cv2.VideoCapture(VIDEO_PATH)

            while (cap.isOpened()):
                # Doc tung frame
                print("Before reading frame")
                ret, frame = cap.read()
                if ret:
                    # TÄƒng biáº¿n Ä‘áº¿m sá»‘ khung hÃ¬nh Ä‘Ã£ xá»­ lÃ½
                    frame_count += 1

                    # Hiá»ƒn thá»‹ khung hÃ¬nh

                    # TÃ­nh thá»i gian káº¿t thÃºc
                    end_time = time.time()

                    # TÃ­nh toÃ¡n FPS
                    elapsed_time = end_time - start_time
                    fps = frame_count / elapsed_time

                    # Reset biáº¿n Ä‘áº¿m vÃ  thá»i Ä‘iá»ƒm báº¯t Ä‘áº§u náº¿u Ä‘Ã£ qua 1 giÃ¢y
                    if elapsed_time > 1:
                        frame_count = 0
                        start_time = time.time()

                    # Hiá»ƒn thá»‹ FPS lÃªn giao diá»‡n ngÆ°á»i dÃ¹ng
                    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Hiá»ƒn thá»‹ khung hÃ¬nh
                    cv2.imshow('Face Recognition', frame)


                # Phat hien khuon mat, tra ve vi tri trong bounding_boxes
                bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    # Neu co it nhat 1 khuon mat trong frame
                    if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # Cat phan khuon mat tim duoc
                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                            
                            # Dua vao model de classifier
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices]
                            
                            # Lay ra ten va ty le % cua class co ty le cao nhat
                            best_name = class_names[best_class_indices[0]]
                            print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                            # Ve khung mau xanh quanh khuon mat
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20

                            # Neu ty le nhan dang > 0.5 thi hien thi ten
                            if best_class_probabilities > 0.5:
                                name = class_names[best_class_indices[0]]
                            else:
                                # Con neu <=0.5 thi hien thi Unknow
                                name = "Unknown"
                                
                            # Viet text len tren frame    
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                            person_detected[best_name] += 1
                except:
                    pass

                # Hien thi frame len man hinh
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


main()
