import cv2
import time
import os
import urllib.request
import ctypes
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

    # State variables
    debug_overlay = False

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

    # Initialize webcam (default index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        detector.close()
        return

    prev_time = time.time()
    print("Webcam started. Press 'q' to quit, 'd' to toggle debug overlay trails.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        h, w, _ = frame.shape

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

        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Generate monotonic timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)
        
        # Run detection synchronously
        results = detector.detect_for_video(mp_image, timestamp_ms)

        # Draw landmarks, print coordinates, and move cursor
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Convert normalized landmarks to pixel coordinates
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
                
                # Move cursor based on index fingertip (landmark 8)
                if len(hand_landmarks) > 8:
                    index_tip = hand_landmarks[8]
                    
                    # 1. Apply the One Euro Filter to raw coordinates in seconds time-domain
                    current_time = time.time()
                    smooth_x = filter_x.filter(index_tip.x, current_time)
                    smooth_y = filter_y.filter(index_tip.y, current_time)

                    # 2. Append coordinates to history for trails
                    raw_pixel = (int(index_tip.x * w), int(index_tip.y * h))
                    smooth_pixel = (int(smooth_x * w), int(smooth_y * h))
                    raw_trail.append(raw_pixel)
                    smooth_trail.append(smooth_pixel)

                    # 3. Interpolate smoothed fingertip position within active zone [x_min, x_max]
                    t_x = (smooth_x - x_min) / (x_max - x_min) if x_max > x_min else 0.5
                    t_x = max(0.0, min(1.0, t_x)) # Clamp
                    
                    t_y = (smooth_y - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                    t_y = max(0.0, min(1.0, t_y)) # Clamp
                    
                    # Flip X coordinate for natural mirrored control
                    screen_x = int((1.0 - t_x) * screen_width)
                    screen_y = int(t_y * screen_height)
                    
                    # Ensure coordinates are within screen boundaries
                    screen_x = max(0, min(screen_x, screen_width - 1))
                    screen_y = max(0, min(screen_y, screen_height - 1))
                    
                    # Update mouse position using low-level Win32 API SetCursorPos
                    user32.SetCursorPos(screen_x, screen_y)
        else:
            # Reset filters and trails if hand is lost to prevent jump spikes on reappearance
            filter_x.x_prev = None
            filter_x.t_prev = None
            filter_y.x_prev = None
            filter_y.t_prev = None
            raw_trail.clear()
            smooth_trail.clear()

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
                
            # Draw overlay status text
            cv2.putText(frame, "DEBUG MODE: ON (Raw = Yellow, Smoothed = Magenta)", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        # Draw FPS on frame
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("Gesture Cursor Feed", frame)

        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_overlay = not debug_overlay
            print(f"Debug overlay toggled: {'ON' if debug_overlay else 'OFF'}")

    # Clean up resources
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Cleaned up and exited.")

if __name__ == "__main__":
    main()
