from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI(title="YOLOv11 Server")

# Load model (có thể đổi thành best.pt của bạn)
model = YOLO("yolo11s.pt")   # n/s/m/l/x tùy máy

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    results = model(image, conf=0.5, iou=0.45)
    
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": r.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
            })
    
    return JSONResponse({
        "success": True,
        "num_objects": len(detections),
        "detections": detections
    })

# Chạy: uvicorn app:app --host 0.0.0.0 --port 8000