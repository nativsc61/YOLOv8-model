import os
import cv2
import torch
import difflib
import numpy as np
import time
import json
import re
import subprocess
from collections import deque
from threading import Thread, Lock
from queue import Queue
from PIL import Image
from flask import Flask, render_template, Response, request, jsonify, send_file
from flask_socketio import SocketIO
import yt_dlp
from ultralytics import YOLO
from pypdf import PdfReader

# ==========================================
# חישוב נתיבים מוחלטים למניעת שגיאות טעינה
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
BALL_MODEL_PATH = os.path.join(BASE_DIR, 'ball.pt')

# יצירת תיקיות דיבאג ושמירת סרטונים
CROPS_DIR = os.path.join(BASE_DIR, 'debug_crops')
CLIPS_DIR = os.path.join(BASE_DIR, 'saved_clips')
MATCHES_DIR = os.path.join(BASE_DIR, "cached_matches")
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(MATCHES_DIR, exist_ok=True)

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
            self.yolo_model = YOLO(model_path).to(device)
            print("✅ מודל YOLO ראשי נטען בהצלחה על כרטיס המסך!")
        else:
            print(f"⚠️ אזהרה: הקובץ {model_path} לא נמצא. טוען yolo11n.pt ברירת מחדל...")
            self.yolo_model = YOLO("yolo11n.pt").to(device)

        print(f"Loading Ball YOLO Model from {ball_model_path}...")
        if os.path.exists(ball_model_path):
            self.ball_model = YOLO(ball_model_path).to(device)
            print("✅ מודל הכדורים (BALL) נטען בהצלחה על כרטיס המסך!")
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
        
        self.last_ocr_time = 0
        
        # היסטוריית מיקומים לצורך מפת חום
        self.historical_points = {
            'all': [],
            'red_alliance': [],
            'blue_alliance': [],
            'robots': {} # יישמר לפי מספר רובוט
        }
        
        Thread(target=self._yolo_loop, daemon=True).start()
        Thread(target=self._ocr_worker_loop, daemon=True).start()

    def reset_scores(self):
        self.accumulated_blue_score = 0
        self.accumulated_red_score = 0
        self.current_scores = {'blue': 0, 'red': 0}
        self.historical_points = {
            'all': [],
            'red_alliance': [],
            'blue_alliance': [],
            'robots': {}
        }

    def _get_youtube_hd_stream_url(self, youtube_url):
        """שליפת כתובת הסטרימינג הישירה של יוטיוב באיכות HD (עד 1080p) באמצעות yt-dlp"""
        command = [
            'yt-dlp',
            '-f', 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
            '-g',
            youtube_url
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            urls = result.stdout.strip().split('\n')
            return urls[0] if urls else None
        except Exception as e:
            print(f"⚠️ שגיאה בשליפת קישור HD מיוטיוב: {e}")
            return None

    def start_source(self, url_or_path):
        with self.cap_lock:
            self.is_running = False
            time.sleep(0.05)
            if self.cap:
                self.cap.release()

            stream_url = url_or_path
            if url_or_path.startswith("http"):
                print("מחלץ קישור HD ישיר מיוטיוב עבור הלייב...")
                hd_url = self._get_youtube_hd_stream_url(url_or_path)
                if hd_url:
                    stream_url = hd_url
                    print("✅ קישור ה-HD לחילוץ בהצלחה!")
                else:
                    print("⚠️ נכשל חילוץ ה-HD, מנסה פביליות רגילות של yt-dlp...")
                    ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url_or_path, download=False)
                        stream_url = info.get('url')

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

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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

            results = self.yolo_model.track(
                frame_to_process, persist=True, tracker="bytetrack.yaml",
                conf=0.1, imgsz=1280, verbose=False
            )

            can_run_ocr = (time.time() - self.last_ocr_time) >= 1.0

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
                    track_id = int(box.id[0].item()) if box.id is not None else None
                    cls_id = int(box.cls[0].item()) if box.cls is not None else 0
                    cls_name = str(self.yolo_model.names.get(cls_id, f"Class_{cls_id}")).upper()

                    if cls_name in ['TIME', 'CLOCK'] and can_run_ocr:
                        self.last_ocr_time = time.time()
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

                    elif cls_name in ['SR', 'SB', 'SCORE_RED', 'SCORE_BLUE'] and can_run_ocr:
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
                        alliance_color = 'red' if (track_id or 0) % 2 == 0 else 'blue'

                        bot_x = float((x1 + x2) / 2.0)
                        bot_y = float(y2)
                        
                        real_map_x, real_map_y = self.homography.transform_point(bot_x, bot_y)

                        pt_entry = {'x': float(real_map_x), 'y': float(real_map_y), 'robot_id': team_number}
                        
                        # שמירה להיסטוריית מפת החום
                        self.historical_points['all'].append(pt_entry)
                        if alliance_color == 'red':
                            self.historical_points['red_alliance'].append(pt_entry)
                        else:
                            self.historical_points['blue_alliance'].append(pt_entry)
                            
                        if team_number not in self.historical_points['robots']:
                            self.historical_points['robots'][team_number] = []
                        self.historical_points['robots'][team_number].append(pt_entry)

                        telemetry.append({
                            'id': team_number,
                            'type': 'robot',
                            'alliance': alliance_color,
                            'x': float(real_map_x),
                            'y': float(real_map_y)
                        })

            if self.ball_model is not None:
                ball_results = self.ball_model(frame_to_process, conf=0.12, imgsz=1280, verbose=False)
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

# --- ניהול נתוני War Room והעלאת לוחות מקצים (PDF, JSON, CSV) ---

@app.route('/api/upload_schedule', methods=['POST'])
def upload_schedule():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
    
    file = request.files['file']
    filename = file.filename.lower()
    
    try:
        if filename.endswith('.pdf'):
            reader = PdfReader(file)
            full_text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    full_text += extracted + "\n"
            
            parsed_matches = []
            
            lines = full_text.split('\n')
            current_match_num = None
            current_teams = []
            
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                
                match_num_search = re.search(r'\b(?:Qual|Qualification|Match|Q)\s*#?\s*(\d+)\b', line_str, re.IGNORECASE)
                
                if match_num_search:
                    if current_match_num and len(current_teams) >= 6:
                        parsed_matches.append({
                            "id": f"qual_{current_match_num}",
                            "name": f"Qualification {current_match_num}",
                            "date": "ISR District Event",
                            "duration": "02:30",
                            "blue_alliance": current_teams[0:3],
                            "red_alliance": current_teams[3:6],
                            "red_score": 0,
                            "blue_score": 0,
                            "points": []
                        })
                    current_match_num = match_num_search.group(1)
                    current_teams = []
                
                teams_in_line = re.findall(r'\b([1-9]\d{2,4})\b', line_str)
                for t in teams_in_line:
                    if len(t) >= 3 and t not in current_teams:
                        current_teams.append(t)
            
            if current_match_num and len(current_teams) >= 6:
                parsed_matches.append({
                    "id": f"qual_{current_match_num}",
                    "name": f"Qualification {current_match_num}",
                    "date": "ISR District Event",
                    "duration": "02:30",
                    "blue_alliance": current_teams[0:3],
                    "red_alliance": current_teams[3:6],
                    "red_score": 0,
                    "blue_score": 0,
                    "points": []
                })

            if not parsed_matches:
                all_teams = re.findall(r'\b([1-9]\d{2,4})\b', full_text)
                for i in range(0, len(all_teams) - 5, 6):
                    match_idx = (i // 6) + 1
                    chunk = all_teams[i:i+6]
                    parsed_matches.append({
                        "id": f"qual_{match_idx}",
                        "name": f"Qualification {match_idx}",
                        "date": "ISR District Event",
                        "duration": "02:30",
                        "blue_alliance": chunk[0:3],
                        "red_alliance": chunk[3:6],
                        "red_score": 0,
                        "blue_score": 0,
                        "points": []
                    })

            schedule_id = "schedule_" + time.strftime("%Y%m%d_%H%M%S")
            master_schedule_data = {
                "id": schedule_id,
                "name": file.filename,
                "matches": parsed_matches
            }
            
            filepath = os.path.join(MATCHES_DIR, f"{schedule_id}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(master_schedule_data, f, ensure_ascii=False, indent=4)
                
            return jsonify({
                'status': 'success', 
                'message': f'PDF processed successfully. Extracted {len(parsed_matches)} matches.'
            })
            
        elif filename.endswith('.json'):
            data = json.load(file)
            match_id = data.get("id", "match_" + time.strftime("%Y%m%d_%H%M%S"))
            filepath = os.path.join(MATCHES_DIR, f"{match_id}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            return jsonify({'status': 'success'})
            
        elif filename.endswith('.csv'):
            file_content = file.read().decode('utf-8', errors='ignore')
            match_data = {
                "id": "csv_schedule_" + time.strftime("%Y%m%d_%H%M%S"),
                "name": file.filename,
                "date": "יום התחרות",
                "duration": "02:30",
                "red_score": 0,
                "blue_score": 0,
                "raw_content": file_content
            }
            filepath = os.path.join(MATCHES_DIR, f"{match_data['id']}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(match_data, f, ensure_ascii=False, indent=4)
            return jsonify({'status': 'success'})
            
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported file format'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/matches', methods=['GET'])
def get_matches():
    matches_list = []
    for filename in os.listdir(MATCHES_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(MATCHES_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "matches" in data and isinstance(data["matches"], list):
                        for m in data["matches"]:
                            matches_list.append({
                                "id": m.get("id"),
                                "name": m.get("name"),
                                "date": m.get("date"),
                                "duration": m.get("duration"),
                                "red_score": m.get("red_score", 0),
                                "blue_score": m.get("blue_score", 0)
                            })
                    else:
                        matches_list.append({
                            "id": data.get("id"),
                            "name": data.get("name"),
                            "date": data.get("date"),
                            "duration": data.get("duration"),
                            "red_score": data.get("red_score", 0),
                            "blue_score": data.get("blue_score", 0)
                        })
            except Exception:
                pass
    return jsonify({"matches": matches_list})

def get_match_by_id(match_id):
    """פונקציית עזר המאותרת ושולפת מקץ ספציפי לפי מזהה מתוך כל קבצי ה-JSON בתיקייה"""
    for filename in os.listdir(MATCHES_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(MATCHES_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("id") == match_id:
                        return {
                            "red_teams": data.get("red_alliance", []),
                            "blue_teams": data.get("blue_alliance", [])
                        }
                    if "matches" in data and isinstance(data["matches"], list):
                        for m in data["matches"]:
                            if m.get("id") == match_id:
                                return {
                                    "red_teams": m.get("red_alliance", []),
                                    "blue_teams": m.get("blue_alliance", [])
                                }
            except Exception:
                pass
    return None

@app.route('/api/war_room/head_to_head/<match_id>')
def head_to_head_analysis(match_id):
    match_data = get_match_by_id(match_id)
    if not match_data:
        return jsonify({"status": "error", "message": "Match not found"}), 404

    red_teams = match_data.get("red_teams", [])
    blue_teams = match_data.get("blue_teams", [])

    def get_team_stats(team_number):
        return {
            "team": team_number,
            "avg_auto": 15.5,
            "avg_teleop": 45.0,
            "avg_endgame": 12.0,
            "primary_role": "Scorer"
        }

    red_stats = [get_team_stats(t) for t in red_teams]
    blue_stats = [get_team_stats(t) for t in blue_teams]

    red_total_auto = sum(s["avg_auto"] for s in red_stats)
    blue_total_auto = sum(s["avg_auto"] for s in blue_stats)
    
    red_total_power = sum(s["avg_teleop"] + s["avg_endgame"] for s in red_stats)
    blue_total_power = sum(s["avg_teleop"] + s["avg_endgame"] for s in blue_stats)

    analysis = {
        "red": {"teams": red_stats, "total_auto": red_total_auto, "total_power": red_total_power},
        "blue": {"teams": blue_stats, "total_auto": blue_total_auto, "total_power": blue_total_power},
        "autonomous_edge": "Red" if red_total_auto > blue_total_auto else "Blue"
    }

    return jsonify({"status": "success", "analysis": analysis})

@app.route('/api/matches/<match_id>', methods=['DELETE'])
def delete_match(match_id):
    filepath = os.path.join(MATCHES_DIR, f"{match_id}.json")
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return jsonify({'status': 'success', 'message': 'Match deleted successfully'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    for filename in os.listdir(MATCHES_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(MATCHES_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "matches" in data:
                    original_len = len(data["matches"])
                    data["matches"] = [m for m in data["matches"] if m.get("id") != match_id]
                    if len(data["matches"]) < original_len:
                        with open(filepath, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                        return jsonify({'status': 'success', 'message': 'Match deleted from schedule successfully'})
            except Exception:
                pass

    return jsonify({'status': 'error', 'message': 'Match not found'}), 404

@app.route('/api/heatmap/<match_id>')
def get_heatmap(match_id):
    robot_filter = request.args.get('robot', 'all')
    
    # שליפת רשימת הרובוטים הפעילים מתוך מנוע הריצה
    active_robots = list(engine.historical_points['robots'].keys())
    
    points = []
    if robot_filter == 'all':
        points = engine.historical_points['all']
    elif robot_filter == 'red_alliance':
        points = engine.historical_points['red_alliance']
    elif robot_filter == 'blue_alliance':
        points = engine.historical_points['blue_alliance']
    else:
        # סינון לפי מספר קבוצה/רובוט ספציפי
        points = engine.historical_points['robots'].get(str(robot_filter), [])
        
    return jsonify({
        'robots': active_robots,
        'points': points
    })
    
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
