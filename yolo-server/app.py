from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
import mediapipe as mp
import asyncio

app = FastAPI(title="Finger Count Live")
templates = Jinja2Templates(directory="templates")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def count_fingers(hand_landmarks):
    landmarks = hand_landmarks.landmark
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    count = 0
    
    # Ngón cái
    if landmarks[4].x < landmarks[3].x:
        count += 1
    # Các ngón còn lại
    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1
    return count

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        total = 0
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                fingers = count_fingers(hand_lms)
                total += fingers
                
                # Hiển thị trên tay
                h, w, _ = frame.shape
                x = int(hand_lms.landmark[0].x * w)
                y = int(hand_lms.landmark[0].y * h)
                cv2.putText(frame, str(fingers), (x-30, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        cv2.putText(frame, f"Total Fingers: {total}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), 
                           media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)