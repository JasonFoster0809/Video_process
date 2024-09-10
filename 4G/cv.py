import cv2
import os

def extract_frames(video_path, output_folder):
    # Kiểm tra nếu folder output không tồn tại, tạo mới
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Đọc video
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    while success:
        # Lưu frame dưới dạng ảnh
        cv2.imwrite(f"{output_folder}/frame{count}.jpg", image)
        success, image = vidcap.read()
        count += 1

    print(f"Đã lưu {count} frame trong thư mục {output_folder}")

# Ví dụ sử dụng
extract_frames('path/to/video.mp4', 'output_frames')
