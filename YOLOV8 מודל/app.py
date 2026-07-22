import os
import cv2
import torch
import difflib
import numpy as np
import time
import json
from collections import deque
from threading import Thread, Lock
from queue import Queue
from PIL import Image
from flask import Flask, render_template, Response, request, jsonify, send_file
from flask_socketio import SocketIO
import yt_dlp
from ultralytics import YOLO

# ==========================================
# חישוב נתיבים מוחלטים למניעת שגיאות טעינה
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
BALL_MODEL_PATH = os.path.join(BASE_DIR, 'ball.pt')

# יצירת תיקיות דיבאג ושמירת סרטונים
CROPS_DIR = os.path.join(BASE_DIR, 'debug_crops')
CLIPS_DIR = os.path.join(BASE_DIR, 'saved_clips')
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ==========================================
# 1. מילון קבוצות FRC בישראל (מעודכן לכל הקבוצות והרוקיז)
# ==========================================
ALLOWED_FRC_TEAMS = [
    "1574", "1576", "1577", "1580", "1657", "1690", "1937", "1942", "1943", "1954", 
    "2096", "2212", "2230", "2231", "2630", "2679", "3065", "3075", "3083", "3085", 
    "3211", "3316", "3339", "3388", "3835", "4319", "4320", "4338", "4416", "4586", 
    "4590", "4661", "4744", "5135", "5251", "5291", "5554", "5614", "5635", "5654", 
    "5715", "5928", "5951", "5987", "5990", "6104", "6168", "6230", "6738", "6740", 
    "6741", "7039", "7067", "7112", "7177", "7845", "8175", "8222", "8223", "9303", 
    "9304", "9738", "9739", "9740", "10139", "10695", "10935", "10986", "11070", "11329", 
    "11332", "11390", "11471", "11478", "11480"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

trocr_available = False
processor = None
trocr_model = None

try:
    from transformers import TrOCRProcessor, RobertaTokenizer, ViTImageProcessor, VisionEncoderDecoderModel
    print("טוען את מודל TrOCR...")
    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/trocr-base-printed")
    image_processor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-printed")
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)
    
    trocr_available = True
    print("✅ TrOCR נטען בהצלחה!")
except Exception as e:
    print(f"⚠️ אזהרה: TrOCR לא נטען. שגיאה: {e}")

def sanitize_telemetry(data):
    """פונקציית עזר להמרת טיפוסי NumPy ל-Python פרימיטיבי עבור JSON"""
    if isinstance(data, list):
        return [sanitize_telemetry(item) for item in data]
    elif isinstance(data, dict):
        return {k: sanitize_telemetry(v) for k, v in data.items()}
    elif isinstance(data, (np.floating, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.integer, np.int32, np.int64)):
        return int(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data


class PrecisionHomography:
    def __init__(self):
        self.H = None

    def update_from_points(self, video_pts, map_pts, frame_w, frame_h, map_w, map_h):
        if len(video_pts) == 4 and len(map_pts) == 4:
            src_real = np.array([[float(p['x']), float(p['y'])] for p in video_pts], dtype=np.float32)
            dst_real = np.array([[float(p['x']), float(p['y'])] for p in map_pts], dtype=np.float32)
            self.H, _ = cv2.findHomography(src_real, dst_real)

    def transform_point(self, x, y):
        if self.H is None:
            return float(x), float(y)
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        return float(transformed[0][0][0]), float(transformed[0][0][1])


class TrOCRStreamEngine:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, ball_model_path=BALL_MODEL_PATH):
        print(f"Loading YOLO Model from {model_path}...")
        if os.path.exists(model_path):
            self.yolo_model = YOLO(model_path)
            print("✅ מודל YOLO ראשי נטען בהצלחה!")
        else:
            print(f"⚠️ אזהרה: הקובץ {model_path} לא נמצא. טוען yolo11n.pt ברירת מחדל...")
            self.yolo_model = YOLO("yolo11n.pt")

        print(f"Loading Ball YOLO Model from {ball_model_path}...")
        if os.path.exists(ball_model_path):
            self.ball_model = YOLO(ball_model_path)
            print("✅ מודל הכדורים (BALL) נטען בהצלחה!")
        else:
            print(f"⚠️ אזהרה: קובץ מודל הכדורים {ball_model_path} לא נמצא. זיהוי כדורים לא יופעל.")
            self.ball_model = None

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
        self.fps = 30
        
        self.map_w, self.map_h = 600, 400
        map_path = os.path.join(BASE_DIR, 'static', 'map.png')
        if os.path.exists(map_path):
            map_img = cv2.imread(map_path)
            if map_img is not None:
                self.map_h, self.map_w = map_img.shape[:2]

        self.current_telemetry = []
        self.current_scores = {'blue': 0, 'red': 0}
        
        self.accumulated_blue_score = 0
        self.accumulated_red_score = 0
        
        self.robot_team_cache = {}
        self.ocr_queue = Queue(maxsize=50)

        self.rolling_buffer = deque(maxlen=30 * 20)
        self.is_recording_clip = False
        self.clip_frames = []
        self.clip_start_time = 0
        self.saw_zero_time = False
        
        Thread(target=self._yolo_loop, daemon=True).start()
        Thread(target=self._ocr_worker_loop, daemon=True).start()

    def reset_scores(self):
        self.accumulated_blue_score = 0
        self.accumulated_red_score = 0
        self.current_scores = {'blue': 0, 'red': 0}

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
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

            self.reset_scores()

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
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
                tot_ms = (total_frames / fps) * 1000

            self.rolling_buffer.append(frame.copy())

            if self.is_recording_clip:
                self.clip_frames.append(frame.copy())
                elapsed_clip_time = time.time() - self.clip_start_time
                if elapsed_clip_time >= 160.0:
                    self.is_recording_clip = False
                    if self.saw_zero_time:
                        self._save_recorded_clip()
                    self.clip_frames = []

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                with self.frame_lock:
                    self.raw_frame = frame.copy()
                    self.latest_encoded_frame = buffer.tobytes()

            cur_sec = int(cur_ms / 1000)
            tot_sec = int(tot_ms / 1000)
            time_str = f"{cur_sec//60:02d}:{cur_sec%60:02d} / {tot_sec//60:02d}:{tot_sec%60:02d}"

            clean_objects = sanitize_telemetry(self.current_telemetry)

            socketio.emit('telemetry_update', {
                'objects': clean_objects,
                'scores': self.current_scores,
                'time': time_str,
                'current_time': float(cur_ms / 1000.0),
                'total_time': float(tot_ms / 1000.0)
            })

            elapsed = time.time() - start_t
            time.sleep(max(0.001, target_delay - elapsed))

    def _save_recorded_clip(self):
        try:
            timestamp_str = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(CLIPS_DIR, f"match_clip_{timestamp_str}.mp4")
            h, w, _ = self.clip_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, self.fps, (w, h))
            for f in self.clip_frames:
                out.write(f)
            out.release()
        except Exception:
            pass

    def _preprocess_bumper_crop(self, crop_bgr):
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        filtered = cv2.bilateralFilter(resized, 9, 75, 75)
        return Image.fromarray(filtered).convert("RGB")

    def _ocr_worker_loop(self):
        while True:
            track_id, crop_bgr = self.ocr_queue.get()
            if not trocr_available or crop_bgr is None:
                self.ocr_queue.task_done()
                continue

            try:
                pil_crop = self._preprocess_bumper_crop(crop_bgr)
                pixel_values = processor(images=pil_crop, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    generated_ids = trocr_model.generate(pixel_values, max_new_tokens=6)
                raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                clean_digits = "".join([c for c in raw_text if c.isdigit()])

                if len(clean_digits) >= 3:
                    matches = difflib.get_close_matches(clean_digits, ALLOWED_FRC_TEAMS, n=1, cutoff=0.5)
                    if matches:
                        self.robot_team_cache[track_id] = str(matches[0])
            except Exception:
                pass

            self.ocr_queue.task_done()

    def _yolo_loop(self):
        while True:
            if not self.is_running or self.is_paused or self.raw_frame is None:
                time.sleep(0.01)
                continue

            with self.frame_lock:
                frame_to_process = self.raw_frame.copy()

            height, width, _ = frame_to_process.shape
            telemetry = []

            # 1. הרצת מודל הרובוטים והשעון (best.pt)
            results = self.yolo_model.track(
                frame_to_process, persist=True, tracker="bytetrack.yaml",
                conf=0.05, imgsz=640, verbose=False
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
                    track_id = int(box.id[0].item()) if box.id is not None else None
                    cls_id = int(box.cls[0].item()) if box.cls is not None else 0
                    cls_name = str(self.yolo_model.names.get(cls_id, f"Class_{cls_id}")).upper()

                    if cls_name in ['TIME', 'CLOCK']:
                        x1_p, y1_p = max(0, int(x1)), max(0, int(y1))
                        x2_p, y2_p = min(width, int(x2)), min(height, int(y2))
                        time_crop = frame_to_process[y1_p:y2_p, x1_p:x2_p]

                        if time_crop.size > 0 and trocr_available:
                            try:
                                pil_crop = self._preprocess_bumper_crop(time_crop)
                                pixel_values = processor(images=pil_crop, return_tensors="pt").pixel_values.to(device)
                                with torch.no_grad():
                                    generated_ids = trocr_model.generate(pixel_values, max_new_tokens=6)
                                raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                                
                                if "2:20" in raw_text or "2.20" in raw_text:
                                    if not self.is_recording_clip:
                                        self.is_recording_clip = True
                                        self.clip_start_time = time.time()
                                        self.saw_zero_time = False
                                        self.clip_frames = list(self.rolling_buffer)

                                if "0:00" in raw_text or "0.00" in raw_text or "00:00" in raw_text:
                                    if self.is_recording_clip:
                                        self.saw_zero_time = True
                            except Exception:
                                pass

                    elif cls_name in ['SR', 'SB', 'SCORE_RED', 'SCORE_BLUE']:
                        x1_p, y1_p = max(0, int(x1)), max(0, int(y1))
                        x2_p, y2_p = min(width, int(x2)), min(height, int(y2))
                        score_crop = frame_to_process[y1_p:y2_p, x1_p:x2_p]

                        if score_crop.size > 0 and trocr_available:
                            try:
                                pil_crop = self._preprocess_bumper_crop(score_crop)
                                pixel_values = processor(images=pil_crop, return_tensors="pt").pixel_values.to(device)
                                with torch.no_grad():
                                    generated_ids = trocr_model.generate(pixel_values, max_new_tokens=4)
                                raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                                digits_only = "".join([c for c in raw_text if c.isdigit()])
                                if digits_only:
                                    val = int(digits_only)
                                    if cls_name in ['SR', 'SCORE_RED']:
                                        self.accumulated_red_score = val
                                    elif cls_name in ['SB', 'SCORE_BLUE']:
                                        self.accumulated_blue_score = val
                            except Exception:
                                pass
                    else:
                        if track_id is not None and track_id not in self.robot_team_cache:
                            x1_p, y1_p = max(0, int(x1) - 10), max(0, int(y1) - 10)
                            x2_p, y2_p = min(width, int(x2) + 10), min(height, int(y2) + 10)
                            bumper_crop = frame_to_process[y1_p:y2_p, x1_p:x2_p]
                            
                            if bumper_crop.size > 0 and self.ocr_queue.qsize() < 30:
                                try:
                                    self.ocr_queue.put_nowait((track_id, bumper_crop))
                                except Exception:
                                    pass

                        team_number = str(self.robot_team_cache.get(track_id, f"Robot-{track_id}" if track_id else "Robot"))

                        bot_x = float((x1 + x2) / 2.0)
                        bot_y = float(y2)
                        
                        real_map_x, real_map_y = self.homography.transform_point(bot_x, bot_y)

                        telemetry.append({
                            'id': team_number,
                            'type': 'robot',
                            'alliance': 'red' if (track_id or 0) % 2 == 0 else 'blue',
                            'x': float(real_map_x),
                            'y': float(real_map_y)
                        })

            # 2. הרצת מודל הכדורים (BALL)
            if self.ball_model is not None:
                ball_results = self.ball_model(frame_to_process, conf=0.12, imgsz=640, verbose=False)
                if ball_results and len(ball_results) > 0 and ball_results[0].boxes is not None:
                    ball_boxes = ball_results[0].boxes
                    for bbox in ball_boxes:
                        bx1, by1, bx2, by2 = [float(v) for v in bbox.xyxy[0].cpu().numpy()]
                        b_cls_id = int(bbox.cls[0].item()) if bbox.cls is not None else 0
                        b_cls_name = str(self.ball_model.names.get(b_cls_id, f"Class_{b_cls_id}")).lower()

                        if b_cls_name == 'fuel':
                            ball_center_x = float((bx1 + bx2) / 2.0)
                            ball_center_y = float((by1 + by2) / 2.0)

                            map_ball_x, map_ball_y = self.homography.transform_point(ball_center_x, ball_center_y)

                            telemetry.append({
                                'id': 'fuel',
                                'type': 'ball',
                                'alliance': 'neutral',
                                'x': float(map_ball_x),
                                'y': float(map_ball_y)
                            })

            self.current_telemetry = telemetry
            self.current_scores = {
                'blue': self.accumulated_blue_score, 
                'red': self.accumulated_red_score
            }
            time.sleep(0.005)

engine = TrOCRStreamEngine(model_path=DEFAULT_MODEL_PATH, ball_model_path=BALL_MODEL_PATH)

# ==========================================
# Routes & API Endpoints
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/matches')
def matches():
    return render_template('matches.html')

@app.route('/war-room')
def war_room():
    return render_template('war_room.html')


@app.route('/static/map.png')
def get_map():
    map_path = os.path.join(BASE_DIR, 'static', 'map.png')
    if os.path.exists(map_path):
        return send_file(map_path, mimetype='image/png')
    return jsonify({'error': 'Map image not found'}), 404

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

@app.route('/api/start_stream', methods=['POST'])
def start_stream():
    data = request.json or {}
    url = data.get('url', '')
    engine.reset_scores()
    success, msg = engine.start_source(url)
    return jsonify({'status': 'success' if success else 'error', 'message': msg})

@app.route('/api/control', methods=['POST'])
def control():
    data = request.json or {}
    action = data.get('action')
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
    data = request.json or {}
    video_pts = data.get('video', [])
    map_pts = data.get('map', [])
    
    engine.homography.update_from_points(
        video_pts, map_pts, 
        engine.frame_w, engine.frame_h, 
        engine.map_w, engine.map_h
    )
    return jsonify({'status': 'success'})

# --- ניהול נתוני War Room ---
MATCHES_DIR = os.path.join(BASE_DIR, "cached_matches")
os.makedirs(MATCHES_DIR, exist_ok=True)

@app.route('/api/matches', methods=['GET'])
def get_matches():
    matches_list = []
    for filename in os.listdir(MATCHES_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(MATCHES_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    matches_list.append({
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "date": data.get("date"),
                        "duration": data.get("duration"),
                        "red_score": data.get("red_score"),
                        "blue_score": data.get("blue_score")
                    })
            except Exception:
                pass
    return jsonify({"matches": matches_list})

@app.route('/api/heatmap/<match_id>')
def get_heatmap(match_id):
    robot_filter = request.args.get('robot', 'all')
    filepath = os.path.join(MATCHES_DIR, f"{match_id}.json")
    if not os.path.exists(filepath):
        return jsonify({"points": [], "robots": []})
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            match = json.load(f)
    except Exception:
        return jsonify({"points": [], "robots": []})
    
    points = []
    robot_ids = set()
    
    if "frames" in match:
        for frame in match.get("frames", []):
            for obj in frame.get("objects", []):
                if obj.get("type") == "robot":
                    r_id = str(obj.get("id", obj.get("alliance", "unknown")))
                    robot_ids.add(r_id)
                    if robot_filter == 'all' or robot_filter == r_id:
                        points.append({"x": obj["x"], "y": obj["y"]})
    else:
        points = match.get("points", [])
        robot_ids = {"red_alliance", "blue_alliance"}

    return jsonify({
        "points": points,
        "robots": sorted(list(robot_ids))
    })

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)