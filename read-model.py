import pickle
import numpy as np
import pickle

# Đọc đối tượng từ file pickle
with open('C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\FaceRecog-facenet\\Models\\facemodel.pkl', 'rb') as f:
    loaded_object = pickle.load(f)

# In ra thông số của mô hình SVC
print("Model Parameters:")
print(loaded_object[0])

# In ra danh sách tên người
print("\nList of Names:")
print(loaded_object[1])