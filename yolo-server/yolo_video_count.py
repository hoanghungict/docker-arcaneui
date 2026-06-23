from ultralytics import YOLO
import cv2
from collections import defaultdict
import sys

# ================== CẤU HÌNH ==================
model = YOLO("yolo11s.pt")        # thay bằng best.pt nếu có model custom
input_video = "video_test.mp4"    # ← ĐƯỜNG DẪN VIDEO CỦA BẠN
output_video = "ketqua_" + input_video

conf_threshold = 0.5
count_class = "person"            # thay thành "car", "all" nếu muốn đếm tất cả
# ============================================

cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

track_history = defaultdict(list)
seen_ids = set()
total_count = 0

print("Đang xử lý video...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Chạy YOLO với tracking
    results = model.track(frame, persist=True, conf=conf_threshold, iou=0.45)

    for result in results:
        if result.boxes.id is not None:
            for box in result.boxes:
                cls_id = int(box.cls)
                track_id = int(box.id)
                class_name = result.names[cls_id]

                # Vẽ box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"{class_name} #{track_id}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Đếm
                if class_name == count_class or count_class == "all":
                    if track_id not in seen_ids:
                        seen_ids.add(track_id)
                        total_count += 1

    # Hiển thị số đếm trên video
    cv2.putText(frame, f"{count_class.capitalize()}: {total_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    out.write(frame)

print(f"✅ Hoàn thành!")
print(f"   Tổng {count_class} đếm được: {total_count}")
print(f"   Video kết quả: {output_video}")

cap.release()
out.release()