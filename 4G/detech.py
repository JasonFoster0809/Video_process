from ultralytics import YOLO
import cv2

# Load model YOLOv8 (ví dụ dùng YOLOv8n là phiên bản nhỏ và nhanh nhất)
model = YOLO('yolov8m.pt')

def detect_objects(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)

    # Sử dụng YOLOv8 để phát hiện đối tượng
    results = model(img)  # Dùng ảnh hoặc đường dẫn đến ảnh

    # Trích xuất thông tin kết quả
    annotated_img = results[0].plot()  # Tạo ảnh với các bounding boxes
    cv2.imwrite(image_path.replace('.jpg', '_result.jpg'), annotated_img)  # Lưu ảnh với bounding box
