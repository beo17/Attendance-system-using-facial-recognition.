import os
import matplotlib.pyplot as plt

#base_folder = "Dataset\\facedata"
base_folder = "C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\Dataset\\facedata"

# Hàm để đếm số ảnh trong thư mục của mỗi học sinh
def count_images(folder):
    student_image_count = {}
    for student_folder in os.listdir(folder):
        student_path = os.path.join(folder, student_folder)
        if os.path.isdir(student_path):
            images = [f for f in os.listdir(student_path) if f.endswith('.jpg') or f.endswith('.png')]
            student_image_count[student_folder] = len(images)
    return student_image_count

# Đếm số ảnh trong thư mục processed
processed_folder = os.path.join(base_folder, 'processed6')
processed_counts = count_images(processed_folder)

# Đếm số ảnh trong thư mục raw
raw_folder = os.path.join(base_folder, 'raw')
raw_counts = count_images(raw_folder)

# Tổng hợp số ảnh từ cả hai thư mục
all_counts = {student: processed_counts.get(student, 0) + raw_counts.get(student, 0) for student in set(processed_counts) | set(raw_counts)}

# Sắp xếp theo số lượng ảnh giảm dần
sorted_counts = dict(sorted(all_counts.items(), key=lambda item: item[1], reverse=True))

# Vẽ đồ thị
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.4
index = range(len(sorted_counts))

bars = ax.bar(index, sorted_counts.values(), bar_width, label='Image Count')

ax.set_xlabel('People')
ax.set_ylabel('Image Count')
ax.set_title('Number of Images per People in FaceData')
ax.set_xticks(index)
ax.set_xticklabels(sorted_counts.keys(), rotation='vertical')
ax.legend()

plt.show()
