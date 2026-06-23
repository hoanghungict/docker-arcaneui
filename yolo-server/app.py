from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
import cv2
import tempfile
import shutil
from pathlib import Path
import uuid

app = FastAPI(title="YOLOv11 Video Server")

model = YOLO("yolo11s.pt")   # thay bằng best.pt nếu bạn có model custom

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    # ... (giữ nguyên code detect ảnh cũ của bạn)
    pass  # bạn có thể giữ lại nếu cần

@app.post("/detect_video")
async def detect_video(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return JSONResponse({"error": "Chỉ hỗ trợ video mp4, avi, mov, mkv"}, status_code=400)

    # Tạo file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_in:
        shutil.copyfileobj(file.file, tmp_in)
        tmp_in_path = tmp_in.name

    output_path = f"results/{uuid.uuid4()}.mp4"
    Path("results").mkdir(exist_ok=True)

    cap = cv2.VideoCapture(tmp_in_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    seen_ids = set()
    total_person = 0

    print(f"Đang xử lý video: {file.filename}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, conf=0.5, iou=0.45)

        for result in results:
            if result.boxes.id is not None:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    track_id = int(box.id)
                    class_name = result.names[cls_id]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, f"{class_name} #{track_id}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if class_name == "person" and track_id not in seen_ids:
                        seen_ids.add(track_id)
                        total_person += 1

        cv2.putText(frame, f"Persons: {total_person}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        out.write(frame)

    cap.release()
    out.release()

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"ketqua_{file.filename}"
    )