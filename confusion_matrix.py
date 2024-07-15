import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle
import glob
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn đến tệp facemodel.pkl và thư mục kiểm tra
MODEL_PATH = 'C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Models\\facemodel6.pkl'
TEST_DIR = 'C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Dataset\\facetest\\processed6'
FACENET_MODEL_PATH = 'C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Models\\20180402-114759.pb'

# Hàm để tải mô hình phân loại và danh sách tên lớp
def load_classifier_and_classnames(classifier_path):
    with open(classifier_path, 'rb') as file:
        model, class_names = pickle.load(file)
    return model, class_names

# Hàm để tải và tiền xử lý hình ảnh kiểm tra
def load_and_preprocess_image(image_path, image_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype('float32') / 255.0
    return img

# Tạo một phiên TensorFlow
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.Session()

print('Loading feature extraction model')
# Đọc mô hình FaceNet từ tệp pb
with gfile.FastGFile(FACENET_MODEL_PATH, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.compat.v1.import_graph_def(graph_def, name='')

# Lấy các biến cần thiết từ mô hình
images_placeholder = sess.graph.get_tensor_by_name('input:0')
embeddings = sess.graph.get_tensor_by_name('embeddings:0')
phase_train_placeholder = sess.graph.get_tensor_by_name('phase_train:0')

# Tải mô hình phân loại và danh sách tên lớp từ tệp facemodel.pkl
model, class_names = load_classifier_and_classnames(MODEL_PATH)

# Tải danh sách đường dẫn của hình ảnh kiểm tra
test_paths = glob.glob(os.path.join(TEST_DIR, '**/*.png'), recursive=True)

X_test = []  # Danh sách chứa các hình ảnh kiểm tra
y_test = []  # Danh sách chứa nhãn tương ứng của các hình ảnh

# Lặp qua các hình ảnh kiểm tra và tạo dữ liệu kiểm tra
for path in test_paths:
    label = os.path.basename(os.path.dirname(path))  # Nhãn là tên thư mục chứa hình ảnh
    img = load_and_preprocess_image(path, image_size=160)
    X_test.append(img)
    y_test.append(label)

# Chuyển đổi danh sách hình ảnh kiểm tra thành mảng numpy
X_test = np.array(X_test)
y_test = np.array(y_test)

# Trích xuất các vectơ nhúng từ dữ liệu kiểm tra bằng mô hình FaceNet
X_test_embeddings = []
for image in X_test:
    # Tiền xử lý ảnh và trích xuất vectơ nhúng
    scaled_reshape = image.reshape(-1, 160, 160, 3)
    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
    emb_array = sess.run(embeddings, feed_dict=feed_dict)
    X_test_embeddings.append(emb_array.flatten())

# Chuyển đổi danh sách vectơ nhúng thành mảng numpy
X_test_embeddings = np.array(X_test_embeddings)

# Dự đoán với mô hình SVC
y_pred = model.predict(X_test_embeddings)

# Tạo một đối tượng LabelEncoder và sử dụng nó để chuyển đổi các nhãn chuỗi thành các nhãn số
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

# Tính ma trận nhầm lẫn và độ chính xác
confusion = confusion_matrix(y_test_encoded, y_pred)
accuracy = accuracy_score(y_test_encoded, y_pred)

# Chuyển đổi ma trận nhầm lẫn sang tỷ lệ phần trăm
confusion_percent = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

# Ma trận nhầm lẫn và độ chính xác
print("Confusion Matrix (Percentage):")
print(confusion_percent)
print("Accuracy:", accuracy)

# Hiển thị thông tin bổ sung như precision, recall, và f1-score
report = classification_report(y_test_encoded, y_pred, target_names=class_names)
print(report)

# Ma trận nhầm lẫn dưới dạng biểu đồ
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.2)  # Font size
sns.heatmap(confusion_percent, annot=True, fmt='.2f', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Percentage)")
plt.show()

# Đóng phiên TensorFlow sau khi hoàn thành công việc
sess.close()
