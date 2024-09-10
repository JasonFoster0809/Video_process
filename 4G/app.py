from flask import Flask, render_template, request, redirect, url_for
import os
from ultralytics import YOLO  # Dùng YOLOv8
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'

# Khởi tạo YOLOv8 model (dùng YOLOv8m)
model = YOLO('yolov8m.pt')  # Thay đổi ở đây nếu bạn muốn dùng YOLOv8m

def is_similar_frame(frame1, frame2, threshold=0.9):
    """
    Hàm so sánh sự giống nhau giữa hai frame bằng cách so sánh histogram.
    Nếu giá trị trả về cao hơn ngưỡng threshold, coi như hai frame giống nhau.
    """
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    hist1 = cv2.normalize(hist1, hist1)
    hist2 = cv2.normalize(hist2, hist2)
    
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity > threshold

def extract_frames(video_path, result_folder, video_id, video_name, frame_interval=15, similarity_threshold=0.9):
    """
    Hàm tách frame từ video, mỗi frame_interval giây lấy một frame.
    Chỉ lấy frame nếu không giống frame trước đó dựa trên so sánh histogram.
    Tên tệp sẽ bao gồm tên video gốc kèm số frame.
    """
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    processed_frame_count = 1  # Bộ đếm riêng để đánh số cho frame đã xử lý
    frame_paths = []
    success, previous_frame = video.read()  # Đọc frame đầu tiên
    if not success:
        return f"{video_name}: Failed to read video"

    base_video_name = os.path.splitext(video_name)[0]  # Lấy phần tên tệp không có phần mở rộng

    while success:
        success, frame = video.read()
        if not success:
            break

        # Chỉ lưu frame nếu frame_count chia hết cho frame_interval và không giống frame trước đó
        if frame_count % frame_interval == 0 and not is_similar_frame(previous_frame, frame, similarity_threshold):
            # Đặt tên tệp theo định dạng {video_name}_frame{processed_frame_count:02d}.jpg
            frame_path = os.path.join(result_folder, f"{base_video_name}_frame{processed_frame_count:02d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            previous_frame = frame  # Cập nhật frame trước đó để tiếp tục so sánh
            processed_frame_count += 1  # Tăng bộ đếm sau khi xử lý frame

        frame_count += 1
    video.release()

    # Xử lý các frame với YOLOv8 song song
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(detect_objects, path) for path in frame_paths]
        for future in as_completed(futures):
            result_path = future.result()
            print(f"{video_name} - Processed frame: {result_path}")

    return f"{video_name}: {processed_frame_count - 1} frames processed"


def detect_objects(image_path):
    """
    Hàm sử dụng YOLOv8 để phát hiện đối tượng trên một ảnh và lưu kết quả.
    """
    img = cv2.imread(image_path)
    results = model(img)  # Dùng YOLOv8 để phát hiện đối tượng
    annotated_img = results[0].plot()  # Tạo bounding box trên ảnh
    result_path = image_path.replace('uploads', 'results')  # Đổi đường dẫn lưu kết quả
    cv2.imwrite(result_path, annotated_img)  # Lưu ảnh với bounding box
    return result_path

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Trang chính của ứng dụng. Nhận video từ người dùng, xử lý và trả về kết quả.
    """
    if request.method == 'POST':
        videos = request.files.getlist('videos')  # Nhận danh sách nhiều video
        if videos:
            with ThreadPoolExecutor(max_workers=len(videos)) as executor:
                futures = []
                for video in videos:
                    video_id = request.form.get('video_id', '01')  # Lấy số thứ tự video từ form hoặc mặc định là 01
                    video_name = video.filename  # Lưu tên của video
                    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
                    video.save(video_path)
                    futures.append(executor.submit(extract_frames, video_path, app.config['RESULT_FOLDER'], video_id, video_name, frame_interval=15, similarity_threshold=0.9))

                for future in as_completed(futures):
                    result = future.result()
                    print(result)  # Hiển thị kết quả kèm tên video

            return "All videos processed and objects detected!"
    return render_template('index.html')



if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    if not os.path.exists(app.config['RESULT_FOLDER']):
        os.makedirs(app.config['RESULT_FOLDER'])

    app.run(debug=True)
