import cv2
import os

def create_grayscale_images(input_folder, output_folder, max_images=300):
    # Kiểm tra xem thư mục đầu ra có tồn tại chưa, nếu không thì tạo mới
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Duyệt qua tất cả các tệp trong thư mục đầu vào
    count = 0
    for filename in os.listdir(input_folder):
        # Thực hiện chỉ đối với các tệp hình ảnh (có thể cần kiểm tra định dạng tệp)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jfif") :
            # Đường dẫn đầy đủ đến tệp hình ảnh đầu vào
            input_path = os.path.join(input_folder, filename)

            # Đọc ảnh màu
            color_image = cv2.imread(input_path)

            # Chuyển đổi sang ảnh xám
            grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Tạo tên cho ảnh xám thứ nhất và thứ hai
            output_path_1 = os.path.join(output_folder, f"{filename[:-4]}_1.jpg")

            # Lưu ảnh xám thứ nhất
            cv2.imwrite(output_path_1, grayscale_image)

          

            count += 1
            if count >= max_images:
                break

# Gọi hàm để tạo ảnh xám và lưu vào thư mục đầu ra
#create_grayscale_images("Dataset\\facedata\\raw\\Brie Larson", "Dataset\\facedata\\raw\\Brie Larson1", max_images=50)

create_grayscale_images("C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\FaceRecog-facenet\\Dataset\\facedata\\raw\\Thinh", "C:\\Users\\DELL\\Desktop\\FaceRecog-facenet\\FaceRecog-facenet\\Dataset\\facedata\\raw\\Thinh1", max_images=300)
