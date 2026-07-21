import os
import cv2
import torch
import difflib
import numpy as np
import time
from threading import Thread, Lock
from queue import Queue
from PIL import Image, ImageEnhance
from flask import Flask, render_template, Response, request, jsonify, send_file
from flask_socketio import SocketIO
import yt_dlp
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ==========================================
# 1. מילון קבוצות FRC בישראל
# ==========================================
ALLOWED_FRC_TEAMS = [
    "1574", "1577", "1657", "1690", "1937", "1943", "1954", "2096", 
    "2212", "2230", "2231", "2630", "3065", "3075", "3085", "3211", 
    "3316", "3339", "3388", "3835", "4319", "4320", "4338", "4416", 
    "4586", "4590", "4661", "4744", "5135", "5251", "5554", "5614", 
    "5635", "5654", "5715", "5928", "5951", "5987", "6104", "6168", 
    "6230", "6738", "6740", "6741", "7039", "7067", "7112", "7177", 
    "7845", "8175", "8222", "8223", "9303", "9304", "9739", "9740"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

trocr_available = False
image_processor = None
tokenizer = None
trocr_model = None

try:
    from transformers import TrOCRImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
    print("טוען את מודל TrOCR...")
    image_processor = TrOCRImageProcessor.from_pretrained("microsoft/trocr-base-printed")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-printed", use_fast=False)
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)
    trocr_available = True
    print("TrOCR נטען בהצלחה!")
except Exception as e:
    print(f"⚠️ אזהרה: TrOCR לא נטען. רץ במצב YOLO בלבד. שגיאה: {e}")


class PrecisionHomography:
    def __init__(self):
        self.H = None

    def update_normalized(self, src_norm, dst_norm, frame_w, frame_h, map_w, map_h):
        if len(src_norm) == 4 and len(dst_norm) == 4:
            src_real = np.array([[p[0] * frame_w, p[1] * frame_h] for p in src_norm], dtype=np.float32)
            dst_real = np.array([[p[0] * map_w, p[1] * map_h] for p in dst_norm], dtype=np.float32)
            self.H, _ = cv2.findHomography(src_real, dst_real)

    def transform_point(self, x, y):
        if self.H is None:
            return x, y
        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        return float(transformed[0][0][0]), float(transformed[0][0][1])


class TrOCRStreamEngine:
    def __init__(self, model_path="best.pt"):
        print(f"Loading YOLO Model from {model_path}...")
        self.yolo_model = YOLO(model_path)

        self.cap = None
        self.is_running = False
        self.is_paused = False
        
        self.cap_lock = Lock()
        self.frame_lock = Lock()
        
        self.raw_frame = None
        self.latest_encoded_frame = None
        self.homography = PrecisionHomography()

        self.frame_w = 1280
        self.frame_h = 720
        
        if os.path.exists('map.png'):
            map_img = cv2.imread('map.png')
            self.map_h, self.map_w = map_img.shape[:2]
        else:
            self.map_w, self.map_h = 1000, 600

        self.current_telemetry = []
        self.robot_team_cache = {} # {track_id: "1690"}
        
        # תור משימות אסינכרוני ל-OCR
        self.ocr_queue = Queue()
        
        Thread(target=self._yolo_loop, daemon=True).start()
        Thread(target=self._ocr_worker_loop, daemon=True).start()

    def start_source(self, url_or_path):
        with self.cap_lock:
            self.is_running = False
            time.sleep(0.05)
            if self.cap:
                self.cap.release()

            if url_or_path.startswith("http"):
                ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url_or_path, download=False)
                    stream_url = info.get('url')
            else:
                stream_url = url_or_path

            self.cap = cv2.VideoCapture(stream_url)
            if not self.cap.isOpened():
                return False, "לא ניתן לפתוח את הזרם"

            self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

            self.is_running = True
            self.is_paused = False

        Thread(target=self._fast_video_loop, daemon=True).start()
        return True, "הצלחה"

    def set_pause(self, state):
        self.is_paused = state

    def seek_seconds(self, seconds_offset):
        with self.cap_lock:
            if self.cap and self.cap.isOpened():
                current_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                target_frame = max(0, current_frame + (seconds_offset * current_fps))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    def _fast_video_loop(self):
        target_delay = 1.0 / 30.0
        while self.is_running:
            start_t = time.time()
            if self.is_paused:
                time.sleep(0.05)
                continue

            with self.cap_lock:
                if not self.cap or not self.cap.isOpened():
                    break
                success, frame = self.cap.read()
                if not success or frame is None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                cur_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                tot_ms = (self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / (self.cap.get(cv2.CAP_PROP_FPS) or 30)) * 1000

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                with self.frame_lock:
                    self.raw_frame = frame.copy()
                    self.latest_encoded_frame = buffer.tobytes()

            socketio.emit('telemetry_update', {
                'objects': self.current_telemetry,
                'current_time': cur_ms / 1000.0,
                'total_time': tot_ms / 1000.0,
                'map_w': self.map_w,
                'map_h': self.map_h
            })

            elapsed = time.time() - start_t
            time.sleep(max(0.001, target_delay - elapsed))

    def _preprocess_bumper_crop(self, crop_bgr):
        """עיבוד מקדים לתמונת הבאמפר להבלטת המספרים עבור ה-OCR"""
        # המרה לגווני אפור + העלאת קונטרסט
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        # הבלטת טקסט בהיר על רקע כהה
        contrast = cv2.equalizeHist(gray)
        pil_img = Image.fromarray(contrast).convert("RGB")
        return pil_img

    def _ocr_worker_loop(self):
        """תהליך ברקע שמפענח מספרי באמפרים ללא התקעות בלייב"""
        while True:
            track_id, crop_bgr = self.ocr_queue.get()
            if not trocr_available or crop_bgr is None:
                self.ocr_queue.task_done()
                continue

            try:
                pil_crop = self._preprocess_bumper_crop(crop_bgr)
                pixel_values = image_processor(images=pil_crop, return_tensors="pt").pixel_values.to(device)
                
                with torch.no_grad():
                    generated_ids = trocr_model.generate(pixel_values)
                
                raw_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                clean_digits = "".join([c for c in raw_text if c.isdigit()])

                if clean_digits:
                    matches = difflib.get_close_matches(clean_digits, ALLOWED_FRC_TEAMS, n=1, cutoff=0.25)
                    if matches:
                        self.robot_team_cache[track_id] = matches[0]
                        print(f"🎯 Successfully Identified Team #{matches[0]} for Robot ID {track_id}")
            except Exception as e:
                pass

            self.ocr_queue.task_done()

    def _yolo_loop(self):
        while True:
            if not self.is_running or self.is_paused or self.raw_frame is None:
                time.sleep(0.03)
                continue

            with self.frame_lock:
                frame_to_process = self.raw_frame.copy()

            height, width, _ = frame_to_process.shape

            results = self.yolo_model.track(
                frame_to_process, persist=True, tracker="bytetrack.yaml",
                conf=0.12, imgsz=416, verbose=False
            )

            telemetry = []

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    cls_name = self.yolo_model.names.get(cls_id, f"Class_{cls_id}")
                    track_id = int(box.id[0]) if box.id is not None else 0

                    cname_upper = cls_name.upper()
                    is_fuel = "FUEL" in cname_upper or "BALL" in cname_upper or "NOTE" in cname_upper

                    # אם זה רובוט וטרם פוענח המספר שלו - נשלח לתור ה-OCR
                    if not is_fuel and track_id not in self.robot_team_cache:
                        x1_p, y1_p = max(0, int(x1) - 10), max(0, int(y1) - 10)
                        x2_p, y2_p = min(width, int(x2) + 10), min(height, int(y2) + 10)
                        
                        bumper_crop = frame_to_process[y1_p:y2_p, x1_p:x2_p]
                        if bumper_crop.size > 0 and self.ocr_queue.qsize() < 3:
                            self.ocr_queue.put((track_id, bumper_crop))

                    team_number = self.robot_team_cache.get(track_id, f"#{track_id}")

                    bot_x = (x1 + x2) / 2.0
                    bot_y = float(y2)

                    real_map_x, real_map_y = self.homography.transform_point(bot_x, bot_y)
                    norm_x = real_map_x / self.map_w
                    norm_y = real_map_y / self.map_h

                    telemetry.append({
                        'id': f"{team_number}",
                        'cls': cls_name,
                        'number': team_number,
                        'norm_x': norm_x,
                        'norm_y': norm_y,
                        'is_fuel': is_fuel
                    })

            self.current_telemetry = telemetry
            time.sleep(0.02)

engine = TrOCRStreamEngine(model_path="best.pt")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map.png')
def get_map():
    return send_file('map.png', mimetype='image/png')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with engine.frame_lock:
                frame_bytes = engine.latest_encoded_frame
                
            if frame_bytes is None:
                time.sleep(0.03)
                continue

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start():
    data = request.json
    url = data.get('url')
    success, msg = engine.start_source(url)
    return jsonify({'status': 'success' if success else 'error', 'message': msg})

@app.route('/api/control', methods=['POST'])
def control():
    action = request.json.get('action')
    if action == 'play':
        engine.set_pause(False)
    elif action == 'pause':
        engine.set_pause(True)
    elif action == 'seek_back':
        engine.seek_seconds(-10)
    elif action == 'seek_forward':
        engine.seek_seconds(10)
    return jsonify({'status': 'success'})

@app.route('/api/update_calibration', methods=['POST'])
def update_calibration():
    data = request.json
    src_norm = data.get('src_norm')
    dst_norm = data.get('dst_norm')
    
    engine.homography.update_normalized(
        src_norm, dst_norm, 
        engine.frame_w, engine.frame_h, 
        engine.map_w, engine.map_h
    )
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)