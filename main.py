import cv2
import time
import os
import urllib.request
import zipfile
import ctypes
import math
import threading
import json
import difflib
from collections import deque
import numpy as np
import sounddevice as sd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pynput.mouse import Controller

# Import the custom One Euro Filter module
from one_euro_filter import OneEuroFilter

# =====================================================================
# CONFIGURATION CONSTANTS
# =====================================================================
# Active zone margins (normalized range: 0.0 to 0.45)
# 0.2 means a 20% margin is cropped from the edges of the camera frame.
ACTIVE_MARGIN_X = 0.2  # Horizontal margin from left/right edges
ACTIVE_MARGIN_Y = 0.2  # Vertical margin from top/bottom edges

# =====================================================================
# HAND DESIGNATIONS
# =====================================================================
# Note: MediaPipe handedness assumes a mirrored (selfie) view.
# - If you find the roles of your hands are swapped, swap these two values.
PRIMARY_HAND_TYPE = "Right"    # Hand that controls the cursor and clicks
SECONDARY_HAND_TYPE = "Left"  # Hand that controls scrolling and right-click modifier

# =====================================================================
# MEDIAPIPE TRACKING PARAMETERS
# =====================================================================
# Tune these parameters to control how easily the system detects and tracks your hand:
# - DETECTION_CONFIDENCE: Min confidence for initial hand detection. Lower = easier to detect.
# - PRESENCE_CONFIDENCE: Min confidence to verify hand is still present. Lower = fewer tracking drops.
# - TRACKING_CONFIDENCE: Min confidence to track movements frame-to-frame. Lower = more persistent.
# - MODEL_COMPLEXITY: 0 for Lite (fastest), 1 for Full (more accurate but slightly more compute).
DETECTION_CONFIDENCE = 0.3
PRESENCE_CONFIDENCE = 0.3
TRACKING_CONFIDENCE = 0.3
MODEL_COMPLEXITY = 0

# =====================================================================
# ONE EURO FILTER TUNABLE PARAMETERS
# =====================================================================
# Adjust these parameters to tune the smoothness vs. latency of the cursor:
# - min_cutoff: Minimum cutoff frequency (Hz). Lower values reduce jitter at rest.
# - beta: Speed coefficient. Higher values reduce lag during fast movement.
# - d_cutoff: Cutoff frequency for derivative (Hz). Usually left at 1.0.
FILTER_MIN_CUTOFF = 1.0
FILTER_BETA = 0.05
FILTER_D_CUTOFF = 1.0

# =====================================================================
# PINCH-TO-CLICK CONFIGURATION
# =====================================================================
# Settings for simulated mouse clicks via pinching (thumb + index fingertips):
# - PINCH_THRESHOLD: Normalized distance threshold. Lower = tighter pinch required.
# - PINCH_DEBOUNCE_MS: Time (in ms) the pinch must be held to register a click.
PINCH_THRESHOLD = 0.045
PINCH_DEBOUNCE_MS = 100

# =====================================================================
# SCROLLING CONFIGURATION
# =====================================================================
# Settings for secondary-hand pinch scroll control:
# - SCROLL_THRESHOLD: Normalized distance below which scrolling is active.
# - SCROLL_SENSITIVITY: Maximum scrolling speed multiplier.
# - SCROLL_DEADZONE: Vertical offset required to register a scroll direction.
SCROLL_THRESHOLD = 0.08
SCROLL_SENSITIVITY = 40
SCROLL_DEADZONE = 0.02

# =====================================================================
# PREDICTIVE POSITIONING CONFIGURATION
# =====================================================================
# Settings to compensate for pipeline lag (capture + inference + display latency):
# - PREDICTION_HORIZON_MS: Extrapolates position ahead by this many milliseconds.
PREDICTION_HORIZON_MS = 60.0

# =====================================================================
# WAKE-WORD DETECTION CONFIGURATION (openWakeWord)
# =====================================================================
# Options: "alexa", "hey_mycroft", "hey_jarvis", "hey_rhasspy"
WAKE_WORD_MODEL = "alexa"
WAKE_WORD_THRESHOLD = 0.5

# =====================================================================
# VOICE COMMAND CONFIGURATION (Vosk)
# =====================================================================
# Fixed dictionary of supported voice commands
VOICE_COMMANDS = [
    "click", "right click", "double click", 
    "scroll up", "scroll down", "screenshot", 
    "stop listening", "start listening"
]
VOSK_MODEL_ZIP = "vosk-model-small-en-us-0.15.zip"
VOSK_MODEL_DIR = "vosk-model-small-en-us-0.15"
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

# =====================================================================
# OPTIMIZATION PARAMETERS
# =====================================================================
# Downscale input frame size specifically for MediaPipe inference (e.g. 320x240).
INFERENCE_WIDTH = 320
INFERENCE_HEIGHT = 240

# =====================================================================
# MODEL CONFIGURATION
# =====================================================================
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Standard hand landmarks connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                              # Palm base connection
]

# Win32 Mouse Event Flags
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800

# Global Vosk Model reference
vosk_model = None

class PinchState:
    IDLE = "IDLE"
    PINCH_STARTED = "PINCH_STARTED"
    PINCH_CONFIRMED = "PINCH_CONFIRMED"
    RELEASED = "RELEASED"

class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.new_frame_available = False

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
                    self.new_frame_available = True
            time.sleep(0.001)

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
            self.new_frame_available = False
        return grabbed, frame

    def release(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()

class AudioProcessor:
    def __init__(self, model_name="alexa", threshold=0.5, state_dict=None):
        self.model_name = model_name
        self.threshold = threshold
        self.model = None
        self.wake_word_detected_time = 0.0
        self.state_dict = state_dict if state_dict is not None else {}
        
        # Audio recording buffer variables
        self.recording_active = False
        self.recording_start_time = 0.0
        self.recorded_chunks = []

    def initialize_model(self):
        import openwakeword
        from openwakeword.model import Model
        
        print("Checking openwakeword models...")
        openwakeword.utils.download_models()
        self.model = Model(wakeword_models=[self.model_name])
        print(f"openWakeWord engine successfully loaded for: '{self.model_name}'")

    def play_beep_async(self, freq=1000, duration=150):
        def beep():
            try:
                import winsound
                winsound.Beep(freq, duration)
            except Exception:
                pass
        threading.Thread(target=beep, daemon=True).start()

    def callback(self, indata, frames, time_info, status):
        if status:
            print(f"[Audio Thread] Warning: {status}", flush=True)

        # 1. Convert float32 input buffer in range [-1.0, 1.0] to 16-bit PCM (mono)
        pcm_data = (indata * 32767).astype(np.int16).squeeze()

        # 2. Compute RMS volume level for stats
        rms = np.sqrt(np.mean(indata**2)) if len(indata) > 0 else 0.0

        current_time = time.time()

        # 3. Buffer audio if voice command recording is active
        if self.recording_active:
            self.recorded_chunks.append(pcm_data)
            
            # Check if 2 seconds have elapsed
            if current_time - self.recording_start_time >= 2.0:
                self.recording_active = False
                print("[Audio Thread] Done recording 2 seconds. Processing speech...", flush=True)
                audio_data = np.concatenate(self.recorded_chunks)
                
                # Process speech transcription in background thread to prevent buffer overflows
                threading.Thread(
                    target=process_voice_command,
                    args=(audio_data, self.state_dict, self.play_beep_async),
                    daemon=True
                ).start()

        # 4. Perform wake word classification if model is initialized and not currently recording
        elif self.model is not None:
            predictions = self.model.predict(pcm_data)
            
            for key, val in predictions.items():
                if self.model_name in key and val > self.threshold:
                    print(f"\n*** [WAKE WORD DETECTED: {key} (confidence: {val:.2f})] ***\n", flush=True)
                    self.wake_word_detected_time = current_time
                    self.state_dict['wake_word_detected_time'] = current_time
                    self.play_beep_async(1000, 150)
                    
                    # Start 2-second voice command recording
                    self.recording_active = True
                    self.recording_start_time = current_time
                    self.recorded_chunks = [pcm_data]
                    print("[Audio Thread] Recording 2 seconds of command speech...", flush=True)

        print(f"[Audio Thread] RMS Volume: {rms:.4f}", flush=True)

def download_vosk_model():
    if not os.path.exists(VOSK_MODEL_DIR):
        if not os.path.exists(VOSK_MODEL_ZIP):
            print("Downloading small Vosk English model (~40MB)... This may take a few moments.")
            urllib.request.urlretrieve(VOSK_MODEL_URL, VOSK_MODEL_ZIP)
            print("Vosk model download completed.")
        print("Extracting Vosk model...")
        with zipfile.ZipFile(VOSK_MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Vosk model extracted successfully.")

def load_vosk_model_async():
    global vosk_model
    try:
        download_vosk_model()
        from vosk import Model
        vosk_model = Model(VOSK_MODEL_DIR)
        print("Vosk speech recognition engine loaded successfully in background.")
    except Exception as e:
        print(f"Warning: Failed to load Vosk engine ({e}). Voice commands will not function.")

def process_voice_command(audio_data, state_dict, beep_callback):
    global vosk_model
    if vosk_model is None:
        print("[Voice Command] Error: Vosk model not loaded yet. Please wait.", flush=True)
        return

    try:
        from vosk import KaldiRecognizer
        rec = KaldiRecognizer(vosk_model, 16000)
        
        # Run audio buffers through Vosk
        rec.AcceptWaveform(audio_data.tobytes())
        result_json = rec.Result()
        
        result_dict = json.loads(result_json)
        transcription = result_dict.get("text", "").strip().lower()
        
        if not transcription:
            print("[Voice Command] Recognized: '' -> command not recognized", flush=True)
            state_dict['command_match_time'] = time.time()
            state_dict['command_match_text'] = "COMMAND NOT RECOGNIZED"
            state_dict['command_match_success'] = False
            return

        # Perform fuzzy matching using difflib
        matches = difflib.get_close_matches(transcription, VOICE_COMMANDS, n=1, cutoff=0.6)
        if matches:
            matched_cmd = matches[0]
            print(f"[Voice Command] Recognized: '{transcription}' -> Matched: '{matched_cmd}'", flush=True)
            
            # Update state dictionary for visual overlay
            state_dict['command_match_time'] = time.time()
            state_dict['command_match_text'] = f"COMMAND MATCHED: {matched_cmd.upper()}"
            state_dict['command_match_success'] = True
            
            # Play a double success beep
            def double_beep():
                beep_callback(1200, 100)
                time.sleep(0.08)
                beep_callback(1200, 100)
            threading.Thread(target=double_beep, daemon=True).start()
        else:
            print(f"[Voice Command] Recognized: '{transcription}' -> command not recognized", flush=True)
            state_dict['command_match_time'] = time.time()
            state_dict['command_match_text'] = "COMMAND NOT RECOGNIZED"
            state_dict['command_match_success'] = False

    except Exception as e:
        print(f"[Voice Command] Error: {e}", flush=True)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task model file...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download completed successfully.")

def get_screen_size():
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        try:
            import tkinter as tk
            root = tk.Tk()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return width, height
        except Exception:
            return 1920, 1080

def run_performance_benchmark():
    print("--------------------------------------------------")
    print("Running cursor-update latency benchmark (1000 iterations each)...")
    
    try:
        pynput_mouse = Controller()
        start_time = time.perf_counter()
        for i in range(1000):
            pynput_mouse.position = (i % 500, i % 500)
        end_time = time.perf_counter()
        pynput_avg_us = ((end_time - start_time) / 1000.0) * 1e6
        print(f"pynput average update latency: {pynput_avg_us:.3f} microseconds")
    except Exception as e:
        print(f"pynput benchmark failed: {e}")
        pynput_avg_us = None

    try:
        user32 = ctypes.windll.user32
        start_time = time.perf_counter()
        for i in range(1000):
            user32.SetCursorPos(i % 500, i % 500)
        end_time = time.perf_counter()
        ctypes_avg_us = ((end_time - start_time) / 1000.0) * 1e6
        print(f"ctypes SetCursorPos average update latency: {ctypes_avg_us:.3f} microseconds")
    except Exception as e:
        print(f"ctypes benchmark failed: {e}")
        ctypes_avg_us = None

    if pynput_avg_us and ctypes_avg_us:
        speedup = pynput_avg_us / ctypes_avg_us
        pct_improvement = ((pynput_avg_us - ctypes_avg_us) / pynput_avg_us) * 100
        print(f"ctypes SetCursorPos is {speedup:.2f}x faster ({pct_improvement:.2f}% latency reduction)")
    print("--------------------------------------------------\n")

def main():
    # Ensure the MediaPipe model file is available
    download_model()

    # Load Vosk speech recognition model in a background thread at startup
    threading.Thread(target=load_vosk_model_async, daemon=True).start()

    # Get screen size dynamically
    screen_width, screen_height = get_screen_size()
    print(f"Detected Screen Resolution: {screen_width}x{screen_height}")

    # Run the micro-benchmark
    run_performance_benchmark()

    # Load Windows User32 DLL for low-level cursor movement
    user32 = ctypes.windll.user32

    # Initialize separate One Euro Filters for X and Y coordinates
    filter_x = OneEuroFilter(min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA, d_cutoff=FILTER_D_CUTOFF)
    filter_y = OneEuroFilter(min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA, d_cutoff=FILTER_D_CUTOFF)

    # Deques for storing coordinate history (used for debug trails)
    trail_maxlen = 30
    raw_trail = deque(maxlen=trail_maxlen)
    smooth_trail = deque(maxlen=trail_maxlen)
    predict_trail = deque(maxlen=trail_maxlen)

    # Sliding history for velocity prediction: stores (timestamp, x, y)
    prediction_history = deque(maxlen=5)

    # FSM state variables for primary hand click
    pinch_state = PinchState.IDLE
    pinch_start_time = 0.0
    active_click_button = "left"

    # Scrolling state variables for secondary hand
    scroll_active = False
    scroll_start_y = 0.0

    # UI and toggle variables
    debug_overlay = False
    enable_prediction = True

    # Shared thread-safe state dictionary
    shared_state = {
        'wake_word_detected_time': 0.0,
        'command_match_time': 0.0,
        'command_match_text': "",
        'command_match_success': False
    }

    # Profiling Accumulators (in seconds)
    acc_capture = 0.0
    acc_inference = 0.0
    acc_filtering = 0.0
    acc_actuation = 0.0
    frame_count = 0

    # Configure Hand Landmarker Options for 2 hands
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_hand_presence_confidence=PRESENCE_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    )

    # Initialize Hand Landmarker
    detector = vision.HandLandmarker.create_from_options(options)

    # Initialize Audio Processor and Input Stream (runs in background thread)
    audio_stream = None
    audio_processor = AudioProcessor(model_name=WAKE_WORD_MODEL, threshold=WAKE_WORD_THRESHOLD, state_dict=shared_state)
    try:
        audio_processor.initialize_model()
        audio_stream = sd.InputStream(
            channels=1,
            samplerate=16000,
            blocksize=1280,  # Exactly 1280 frames (80ms) for openWakeWord
            callback=audio_processor.callback
        )
        audio_stream.start()
        print("Audio capture and wake-word thread started successfully (16kHz, mono).")
    except Exception as e:
        print(f"Warning: Could not start audio input thread ({e}). Running without audio.")

    # Initialize Threaded Camera
    camera = ThreadedCamera(0).start()
    
    # Give the threads a brief moment to initialize
    time.sleep(0.5)

    print("Webcam thread started. Controls:")
    print("  - Press 'q' to quit.")
    print("  - Press 'd' to toggle debug overlay trails.")
    print("  - Press 'p' to toggle predictive positioning.")

    prev_time = time.time()

    while True:
        # --- PHASE 1: FRAME CAPTURE (Asynchronous/Threaded wait) ---
        t_start = time.perf_counter()
        
        while not camera.new_frame_available:
            time.sleep(0.001)

        ret, frame = camera.read()
        
        t_capture_end = time.perf_counter()
        acc_capture += (t_capture_end - t_start)

        if not ret or frame is None:
            print("Error: Failed to grab frame.")
            break

        h, w, _ = frame.shape

        # --- PHASE 2: MEDIAPIPE INFERENCE ---
        t_inf_start = time.perf_counter()

        small_frame = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        results = detector.detect_for_video(mp_image, timestamp_ms)

        t_inf_end = time.perf_counter()
        acc_inference += (t_inf_end - t_inf_start)

        # --- PHASE 3: COORDINATE FILTERING & GESTURE INTERPRETATION ---
        t_filt_start = time.perf_counter()

        x_min = ACTIVE_MARGIN_X
        x_max = 1.0 - ACTIVE_MARGIN_X
        y_min = ACTIVE_MARGIN_Y
        y_max = 1.0 - ACTIVE_MARGIN_Y

        rect_x1, rect_y1 = int(x_min * w), int(y_min * h)
        rect_x2, rect_y2 = int(x_max * w), int(y_max * h)

        # Draw active zone rectangle (Cyan)
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 0), 2)
        cv2.putText(frame, "Active Tracking Zone", (rect_x1 + 5, rect_y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # Separate hands by classification roles
        primary_hand_landmarks = None
        secondary_hand_landmarks = None

        if results.hand_landmarks and results.handedness:
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                handedness_list = results.handedness[idx]
                category = handedness_list[0]
                hand_type = category.category_name # "Left" or "Right"

                if hand_type == PRIMARY_HAND_TYPE:
                    primary_hand_landmarks = hand_landmarks
                elif hand_type == SECONDARY_HAND_TYPE:
                    secondary_hand_landmarks = hand_landmarks

        # -----------------------------------------------------------------
        # SECTION A: PROCESS SECONDARY HAND (Scroll / Right-Click Modifier)
        # -----------------------------------------------------------------
        sec_is_pinching = False
        
        if secondary_hand_landmarks:
            sec_pixel_landmarks = []
            for lm in secondary_hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                sec_pixel_landmarks.append((cx, cy))
                
            for conn in HAND_CONNECTIONS:
                pt1 = sec_pixel_landmarks[conn[0]]
                pt2 = sec_pixel_landmarks[conn[1]]
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # Blue Connections
            
            for cx, cy in sec_pixel_landmarks:
                cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)  # Cyan Landmarks

            if len(secondary_hand_landmarks) > 8:
                sec_thumb = secondary_hand_landmarks[4]
                sec_index = secondary_hand_landmarks[8]
                
                sec_pinch_dist = math.sqrt((sec_thumb.x - sec_index.x)**2 + (sec_thumb.y - sec_index.y)**2)
                sec_is_pinching = sec_pinch_dist < PINCH_THRESHOLD
                
                if sec_pinch_dist < SCROLL_THRESHOLD:
                    if not scroll_active:
                        scroll_active = True
                        scroll_start_y = sec_index.y
                    
                    dy = scroll_start_y - sec_index.y
                    
                    if abs(dy) > SCROLL_DEADZONE:
                        direction = 1 if dy > 0 else -1
                        tightness = (SCROLL_THRESHOLD - sec_pinch_dist) / SCROLL_THRESHOLD
                        tightness = max(0.0, min(1.0, tightness))
                        scroll_amount = int(direction * SCROLL_SENSITIVITY * tightness)
                        
                        t_act_start = time.perf_counter()
                        user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, scroll_amount, 0)
                        acc_actuation += (time.perf_counter() - t_act_start)
                else:
                    scroll_active = False

                sec_thumb_px = sec_pixel_landmarks[4]
                sec_index_px = sec_pixel_landmarks[8]
                
                if scroll_active and abs(scroll_start_y - sec_index.y) > SCROLL_DEADZONE:
                    sec_line_color = (255, 255, 255)
                elif sec_is_pinching:
                    sec_line_color = (0, 255, 255)
                else:
                    sec_line_color = (0, 0, 255)
                
                cv2.line(frame, sec_thumb_px, sec_index_px, sec_line_color, 2)
                cv2.circle(frame, sec_thumb_px, 6, sec_line_color, -1)
                cv2.circle(frame, sec_index_px, 6, sec_line_color, -1)
                
                cv2.putText(frame, f"Sec Pinch: {sec_pinch_dist:.3f}", (w - 220, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Scroll Active: {'Yes' if scroll_active else 'No'}", (w - 220, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            scroll_active = False

        # -----------------------------------------------------------------
        # SECTION B: PROCESS PRIMARY HAND (Cursor Movement & Clicks)
        # -----------------------------------------------------------------
        if primary_hand_landmarks:
            pixel_landmarks = []
            for lm in primary_hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                pixel_landmarks.append((cx, cy))
                
            for conn in HAND_CONNECTIONS:
                pt1 = pixel_landmarks[conn[0]]
                pt2 = pixel_landmarks[conn[1]]
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            for cx, cy in pixel_landmarks:
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            if len(primary_hand_landmarks) > 8:
                thumb_tip = primary_hand_landmarks[4]
                index_tip = primary_hand_landmarks[8]
                
                current_time = time.time()
                smooth_x = filter_x.filter(index_tip.x, current_time)
                smooth_y = filter_y.filter(index_tip.y, current_time)

                prediction_history.append((current_time, smooth_x, smooth_y))

                target_x = smooth_x
                target_y = smooth_y

                if enable_prediction and len(prediction_history) >= 2:
                    t_first, x_first, y_first = prediction_history[0]
                    t_last, x_last, y_last = prediction_history[-1]
                    dt = t_last - t_first
                    if dt > 0.0:
                        v_x = (x_last - x_first) / dt
                        v_y = (y_last - y_first) / dt

                        horizon_sec = PREDICTION_HORIZON_MS / 1000.0
                        target_x = smooth_x + v_x * horizon_sec
                        target_y = smooth_y + v_y * horizon_sec

                        target_x = max(0.0, min(1.0, target_x))
                        target_y = max(0.0, min(1.0, target_y))

                raw_pixel = (int(index_tip.x * w), int(index_tip.y * h))
                smooth_pixel = (int(smooth_x * w), int(smooth_y * h))
                predict_pixel = (int(target_x * w), int(target_y * h))

                raw_trail.append(raw_pixel)
                smooth_trail.append(smooth_pixel)
                predict_trail.append(predict_pixel)

                t_x = (target_x - x_min) / (x_max - x_min) if x_max > x_min else 0.5
                t_x = max(0.0, min(1.0, t_x))
                
                t_y = (target_y - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                t_y = max(0.0, min(1.0, t_y))
                
                screen_x = int((1.0 - t_x) * screen_width)
                screen_y = int(t_y * screen_height)
                
                screen_x = max(0, min(screen_x, screen_width - 1))
                screen_y = max(0, min(screen_y, screen_height - 1))
                
                t_act_start = time.perf_counter()
                user32.SetCursorPos(screen_x, screen_y)
                acc_actuation += (time.perf_counter() - t_act_start)

                pinch_dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
                is_pinching = pinch_dist < PINCH_THRESHOLD
                current_time_ms = time.time() * 1000

                if pinch_state == PinchState.IDLE:
                    if is_pinching:
                        pinch_state = PinchState.PINCH_STARTED
                        pinch_start_time = current_time_ms

                elif pinch_state == PinchState.PINCH_STARTED:
                    if is_pinching:
                        if current_time_ms - pinch_start_time >= PINCH_DEBOUNCE_MS:
                            pinch_state = PinchState.PINCH_CONFIRMED
                            active_click_button = "right" if sec_is_pinching else "left"
                            
                            t_act_start = time.perf_counter()
                            if active_click_button == "right":
                                user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                                print("Pinch Click: Right Press Triggered")
                            else:
                                user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                                print("Pinch Click: Left Press Triggered")
                            acc_actuation += (time.perf_counter() - t_act_start)
                    else:
                        pinch_state = PinchState.IDLE

                elif pinch_state == PinchState.PINCH_CONFIRMED:
                    if not is_pinching:
                        pinch_state = PinchState.RELEASED
                        
                        t_act_start = time.perf_counter()
                        if active_click_button == "right":
                            user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                            print("Pinch Click: Right Release Triggered")
                        else:
                            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                            print("Pinch Click: Left Release Triggered")
                        acc_actuation += (time.perf_counter() - t_act_start)

                elif pinch_state == PinchState.RELEASED:
                    pinch_state = PinchState.IDLE

                thumb_pixel = pixel_landmarks[4]
                index_pixel = pixel_landmarks[8]
                
                if pinch_state == PinchState.PINCH_CONFIRMED:
                    line_color = (0, 255, 0)
                elif pinch_state == PinchState.PINCH_STARTED:
                    line_color = (0, 165, 255)
                else:
                    line_color = (0, 0, 255)
                
                cv2.line(frame, thumb_pixel, index_pixel, line_color, 2)
                cv2.circle(frame, thumb_pixel, 6, line_color, -1)
                cv2.circle(frame, index_pixel, 6, line_color, -1)
                
                click_type_str = f" ({active_click_button.upper()})" if pinch_state == PinchState.PINCH_CONFIRMED else ""
                cv2.putText(frame, f"Primary State: {pinch_state}{click_type_str}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Pinch Distance: {pinch_dist:.3f}", (10, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            if pinch_state == PinchState.PINCH_CONFIRMED:
                t_act_start = time.perf_counter()
                if active_click_button == "right":
                    user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                else:
                    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                acc_actuation += (time.perf_counter() - t_act_start)
                print(f"Pinch Click: Safety Release ({active_click_button.capitalize()} Lost)")
            
            pinch_state = PinchState.IDLE
            filter_x.x_prev = None
            filter_x.t_prev = None
            filter_y.x_prev = None
            filter_y.t_prev = None
            raw_trail.clear()
            smooth_trail.clear()
            predict_trail.clear()
            prediction_history.clear()

        t_filt_end = time.perf_counter()
        acc_filtering += (t_filt_end - t_filt_start)

        # Draw trails if debug overlay is active
        if debug_overlay:
            for i in range(1, len(raw_trail)):
                thickness = int(1 + 3 * (i / len(raw_trail)))
                cv2.line(frame, raw_trail[i - 1], raw_trail[i], (0, 255, 255), thickness)
            if len(raw_trail) > 0:
                cv2.circle(frame, raw_trail[-1], 6, (0, 255, 255), -1)

            for i in range(1, len(smooth_trail)):
                thickness = int(1 + 4 * (i / len(smooth_trail)))
                cv2.line(frame, smooth_trail[i - 1], smooth_trail[i], (255, 0, 255), thickness)
            if len(smooth_trail) > 0:
                cv2.circle(frame, smooth_trail[-1], 8, (255, 0, 255), -1)

            if enable_prediction:
                for i in range(1, len(predict_trail)):
                    thickness = int(1 + 4 * (i / len(predict_trail)))
                    cv2.line(frame, predict_trail[i - 1], predict_trail[i], (255, 255, 255), thickness)
                if len(predict_trail) > 0:
                    cv2.circle(frame, predict_trail[-1], 8, (255, 255, 255), -1)
                
            cv2.putText(frame, "DEBUG: ON (Raw=Yellow, Smooth=Magenta, Predict=White)", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw command matching overlay (flashes for 1.5 seconds) or wake-word banner (flashes for 1.0 second)
        wake_detected_time = shared_state.get('wake_word_detected_time', 0.0)
        cmd_match_time = shared_state.get('command_match_time', 0.0)
        
        if time.time() - cmd_match_time < 1.5:
            cmd_text = shared_state.get('command_match_text', "")
            success = shared_state.get('command_match_success', False)
            color = (0, 200, 0) if success else (0, 0, 200)  # Green for match, Red for unrecognized
            cv2.rectangle(frame, (0, 0), (w, 55), color, -1)
            cv2.putText(frame, cmd_text, (w // 2 - 180, 36), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        elif time.time() - wake_detected_time < 1.0:
            cv2.rectangle(frame, (0, 0), (w, 55), (0, 200, 0), -1)
            cv2.putText(frame, "WAKE WORD DETECTED!", (w // 2 - 140, 36), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        # Draw FPS on frame
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw prediction state text
        pred_status = f"Prediction: {'ON' if enable_prediction else 'OFF'} ({PREDICTION_HORIZON_MS}ms)"
        cv2.putText(frame, pred_status, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (0, 255, 255) if enable_prediction else (128, 128, 128), 1, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("Gesture Cursor Feed", frame)

        # Print profiling breakdown every 60 frames
        frame_count += 1
        if frame_count >= 60:
            print("\n=============== PER-FRAME PROFILING BREAKDOWN ===============")
            print(f"Average over last 60 frames:")
            print(f"  [1] Frame Capture (Thread Wait): {acc_capture / 60.0 * 1000:.3f} ms")
            print(f"  [2] MediaPipe Inference:         {acc_inference / 60.0 * 1000:.3f} ms")
            print(f"  [3] Coordinate Filtering:        {acc_filtering / 60.0 * 1000:.3f} ms")
            print(f"  [4] Cursor Actuation:            {acc_actuation / 60.0 * 1000:.3f} ms")
            print(f"  Total Active Loop Processing:    {(acc_capture + acc_inference + acc_filtering + acc_actuation) / 60.0 * 1000:.3f} ms")
            print("=============================================================")
            acc_capture = 0.0
            acc_inference = 0.0
            acc_filtering = 0.0
            acc_actuation = 0.0
            frame_count = 0

        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_overlay = not debug_overlay
            print(f"Debug overlay toggled: {'ON' if debug_overlay else 'OFF'}")
        elif key == ord('p'):
            enable_prediction = not enable_prediction
            print(f"Predictive positioning toggled: {'ON' if enable_prediction else 'OFF'}")

    # Clean up resources
    camera.release()
    if audio_stream is not None:
        try:
            audio_stream.stop()
            audio_stream.close()
        except Exception:
            pass
    detector.close()
    cv2.destroyAllWindows()
    print("Cleaned up and exited.")

if __name__ == "__main__":
    main()
