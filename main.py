import cv2
import time
import os
import urllib.request
import ctypes
import math
import threading
from collections import deque
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
# PREDICTIVE POSITIONING CONFIGURATION
# =====================================================================
# Settings to compensate for pipeline lag (capture + inference + display latency):
# - PREDICTION_HORIZON_MS: Extrapolates position ahead by this many milliseconds.
#   (Typical range: 40ms to 90ms. Higher values feel faster/snappier but may overshoot).
PREDICTION_HORIZON_MS = 60.0

# =====================================================================
# OPTIMIZATION PARAMETERS
# =====================================================================
# Downscale input frame size specifically for MediaPipe inference (e.g. 320x240).
# Saves CPU time during image wrapper copying and model processing.
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

class PinchState:
    IDLE = "IDLE"
    PINCH_STARTED = "PINCH_STARTED"
    PINCH_CONFIRMED = "PINCH_CONFIRMED"
    RELEASED = "RELEASED"

class ThreadedCamera:
    """
    Asynchronously reads camera frames in a background thread
    to remove blocking IO bottlenecks from the main thread.
    """
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
            time.sleep(0.001)  # Yield thread

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
    
    # Benchmark 1: pynput.mouse.Controller
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

    # Benchmark 2: ctypes user32.SetCursorPos (Windows)
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
    # Ensure the model file is available
    download_model()

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

    # Finite State Machine state variables
    pinch_state = PinchState.IDLE
    pinch_start_time = 0.0

    # UI and toggle variables
    debug_overlay = False
    enable_prediction = True

    # Profiling Accumulators (in seconds)
    acc_capture = 0.0
    acc_inference = 0.0
    acc_filtering = 0.0
    acc_actuation = 0.0
    frame_count = 0

    # Configure Hand Landmarker Options
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_hand_presence_confidence=PRESENCE_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    )

    # Initialize Hand Landmarker
    detector = vision.HandLandmarker.create_from_options(options)

    # Initialize Threaded Camera
    camera = ThreadedCamera(0).start()
    
    # Give the thread a brief moment to initialize
    time.sleep(0.5)

    print("Webcam thread started. Controls:")
    print("  - Press 'q' to quit.")
    print("  - Press 'd' to toggle debug overlay trails.")
    print("  - Press 'p' to toggle predictive positioning.")

    prev_time = time.time()

    while True:
        # --- PHASE 1: FRAME CAPTURE (Asynchronous/Threaded wait) ---
        t_start = time.perf_counter()
        
        # Wait until the background thread fetches a new frame
        while not camera.new_frame_available:
            time.sleep(0.001)

        # Retrieve the frame from camera object
        ret, frame = camera.read()
        
        t_capture_end = time.perf_counter()
        acc_capture += (t_capture_end - t_start)

        if not ret or frame is None:
            print("Error: Failed to grab frame.")
            break

        h, w, _ = frame.shape

        # --- PHASE 2: MEDIAPIPE INFERENCE ---
        t_inf_start = time.perf_counter()

        # Downscale the frame specifically for MediaPipe to accelerate inference
        small_frame = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Generate monotonic timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)
        
        # Run detection synchronously
        results = detector.detect_for_video(mp_image, timestamp_ms)

        t_inf_end = time.perf_counter()
        acc_inference += (t_inf_end - t_inf_start)

        # --- PHASE 3: COORDINATE FILTERING ---
        t_filt_start = time.perf_counter()

        # Define active zone boundaries
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

        # Process hand landmarks
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Convert normalized landmarks to pixel coordinates on the HIGH-RES frame
                pixel_landmarks = []
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pixel_landmarks.append((cx, cy))
                
                # Draw connection lines (Green)
                for conn in HAND_CONNECTIONS:
                    pt1 = pixel_landmarks[conn[0]]
                    pt2 = pixel_landmarks[conn[1]]
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                
                # Draw landmark circles (Red)
                for cx, cy in pixel_landmarks:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Tracking and gesture control logic
                if len(hand_landmarks) > 8:
                    thumb_tip = hand_landmarks[4]
                    index_tip = hand_landmarks[8]
                    
                    # 1. Apply the One Euro Filter to raw index fingertip coordinates
                    current_time = time.time()
                    smooth_x = filter_x.filter(index_tip.x, current_time)
                    smooth_y = filter_y.filter(index_tip.y, current_time)

                    # 2. Append smoothed coordinates to sliding history for velocity calculation
                    prediction_history.append((current_time, smooth_x, smooth_y))

                    # 3. Calculate velocity and extrapolate cursor position if prediction is enabled
                    target_x = smooth_x
                    target_y = smooth_y

                    if enable_prediction and len(prediction_history) >= 2:
                        t_first, x_first, y_first = prediction_history[0]
                        t_last, x_last, y_last = prediction_history[-1]
                        dt = t_last - t_first
                        if dt > 0.0:
                            v_x = (x_last - x_first) / dt
                            v_y = (y_last - y_first) / dt

                            # Extrapolate position ahead by the prediction horizon (seconds)
                            horizon_sec = PREDICTION_HORIZON_MS / 1000.0
                            target_x = smooth_x + v_x * horizon_sec
                            target_y = smooth_y + v_y * horizon_sec

                            # Clamp to prevent predicted position from sliding off-limits
                            target_x = max(0.0, min(1.0, target_x))
                            target_y = max(0.0, min(1.0, target_y))

                    # 4. Append coordinates to history for trails
                    raw_pixel = (int(index_tip.x * w), int(index_tip.y * h))
                    smooth_pixel = (int(smooth_x * w), int(smooth_y * h))
                    predict_pixel = (int(target_x * w), int(target_y * h))

                    raw_trail.append(raw_pixel)
                    smooth_trail.append(smooth_pixel)
                    predict_trail.append(predict_pixel)

                    # 5. Interpolate target position within active zone [x_min, x_max]
                    t_x = (target_x - x_min) / (x_max - x_min) if x_max > x_min else 0.5
                    t_x = max(0.0, min(1.0, t_x)) # Clamp
                    
                    t_y = (target_y - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                    t_y = max(0.0, min(1.0, t_y)) # Clamp
                    
                    # Flip X coordinate for natural mirrored control
                    screen_x = int((1.0 - t_x) * screen_width)
                    screen_y = int(t_y * screen_height)
                    
                    # Ensure coordinates are within screen boundaries
                    screen_x = max(0, min(screen_x, screen_width - 1))
                    screen_y = max(0, min(screen_y, screen_height - 1))

                    # 6. Calculate pinch distance in 2D normalized space
                    pinch_dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
                    is_pinching = pinch_dist < PINCH_THRESHOLD
                    current_time_ms = time.time() * 1000

                    # 7. Finite State Machine for click actions
                    if pinch_state == PinchState.IDLE:
                        if is_pinching:
                            pinch_state = PinchState.PINCH_STARTED
                            pinch_start_time = current_time_ms

                    elif pinch_state == PinchState.PINCH_STARTED:
                        if is_pinching:
                            if current_time_ms - pinch_start_time >= PINCH_DEBOUNCE_MS:
                                pinch_state = PinchState.PINCH_CONFIRMED
                                
                                # --- PHASE 4 (Sub): CURSOR ACTUATION ---
                                t_act_start = time.perf_counter()
                                user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                                acc_actuation += (time.perf_counter() - t_act_start)
                        else:
                            pinch_state = PinchState.IDLE

                    elif pinch_state == PinchState.PINCH_CONFIRMED:
                        if not is_pinching:
                            pinch_state = PinchState.RELEASED
                            
                            # --- PHASE 4 (Sub): CURSOR ACTUATION ---
                            t_act_start = time.perf_counter()
                            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                            acc_actuation += (time.perf_counter() - t_act_start)

                    elif pinch_state == PinchState.RELEASED:
                        pinch_state = PinchState.IDLE

                    # Draw pinch-to-click visual guide on high-res frame
                    thumb_pixel = pixel_landmarks[4]
                    index_pixel = pixel_landmarks[8]
                    
                    if pinch_state == PinchState.PINCH_CONFIRMED:
                        line_color = (0, 255, 0)  # Green for active click
                    elif pinch_state == PinchState.PINCH_STARTED:
                        line_color = (0, 165, 255)  # Orange for debouncing
                    else:
                        line_color = (0, 0, 255)  # Red for no pinch
                    
                    cv2.line(frame, thumb_pixel, index_pixel, line_color, 2)
                    cv2.circle(frame, thumb_pixel, 6, line_color, -1)
                    cv2.circle(frame, index_pixel, 6, line_color, -1)
                    
                    # Draw pinch feedback text
                    cv2.putText(frame, f"Click State: {pinch_state}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Pinch Distance: {pinch_dist:.3f}", (10, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    t_filt_end = time.perf_counter()
                    acc_filtering += (t_filt_end - t_filt_start)

                    # --- PHASE 4: CURSOR ACTUATION (Position setting) ---
                    t_act_start = time.perf_counter()
                    user32.SetCursorPos(screen_x, screen_y)
                    t_act_end = time.perf_counter()
                    acc_actuation += (t_act_end - t_act_start)
        else:
            t_filt_start = time.perf_counter()
            # Safety checks & resets when hand is lost
            if pinch_state == PinchState.PINCH_CONFIRMED:
                t_act_start = time.perf_counter()
                user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                acc_actuation += (time.perf_counter() - t_act_start)
                print("Pinch Click: Safety Release (Hand Lost)")
            
            pinch_state = PinchState.IDLE
            filter_x.x_prev = None
            filter_x.t_prev = None
            filter_y.x_prev = None
            filter_y.t_prev = None
            raw_trail.clear()
            smooth_trail.clear()
            predict_trail.clear()
            prediction_history.clear()
            acc_filtering += (time.perf_counter() - t_filt_start)

        # Draw trails if debug overlay is active
        if debug_overlay:
            # Draw raw trail (Yellow)
            for i in range(1, len(raw_trail)):
                thickness = int(1 + 3 * (i / len(raw_trail)))
                cv2.line(frame, raw_trail[i - 1], raw_trail[i], (0, 255, 255), thickness)
            if len(raw_trail) > 0:
                cv2.circle(frame, raw_trail[-1], 6, (0, 255, 255), -1)

            # Draw smooth trail (Magenta)
            for i in range(1, len(smooth_trail)):
                thickness = int(1 + 4 * (i / len(smooth_trail)))
                cv2.line(frame, smooth_trail[i - 1], smooth_trail[i], (255, 0, 255), thickness)
            if len(smooth_trail) > 0:
                cv2.circle(frame, smooth_trail[-1], 8, (255, 0, 255), -1)

            # Draw prediction trail (White) if active
            if enable_prediction:
                for i in range(1, len(predict_trail)):
                    thickness = int(1 + 4 * (i / len(predict_trail)))
                    cv2.line(frame, predict_trail[i - 1], predict_trail[i], (255, 255, 255), thickness)
                if len(predict_trail) > 0:
                    cv2.circle(frame, predict_trail[-1], 8, (255, 255, 255), -1)
                
            # Draw overlay status text
            cv2.putText(frame, "DEBUG MODE: ON (Raw=Yellow, Smooth=Magenta, Predict=White)", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

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
            # Reset counters
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
    detector.close()
    cv2.destroyAllWindows()
    print("Cleaned up and exited.")

if __name__ == "__main__":
    main()
