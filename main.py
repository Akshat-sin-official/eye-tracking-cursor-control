"""
Eye-Tracking Cursor Control System - GUI Dashboard

A professional desktop application built using CustomTkinter that controls 
the mouse cursor using webcam eye tracking via MediaPipe Face Mesh.
Features:
- Premium dark-themed dashboard.
- Live video feed embedded directly inside the GUI application window.
- Dropdown camera selector with background scanning.
- Dynamic parameter tuning (cursor smoothing & wink click threshold).
- Visual HUD overlay indicating active tracking status, coordinates, and click states.
- Thread-safe architecture separating camera frame processing from GUI loop.
- Controls: 'q'/ESC to quit, 'r' to recalibrate center, 's' to pause/resume tracking.
"""

import cv2
import mediapipe as mp
import pyautogui
import math
import time
import threading
import queue
import customtkinter as ctk
from PIL import Image, ImageTk

# System configurations
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

# Standard landmark indices for EAR calculations (MediaPipe indexes)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [263, 385, 387, 362, 380, 373]


def calculate_ear(eye_landmarks, landmarks):
    """Calculate the Eye Aspect Ratio (EAR) for blink/wink detection."""
    v1 = math.dist([landmarks[eye_landmarks[1]].x, landmarks[eye_landmarks[1]].y],
                   [landmarks[eye_landmarks[5]].x, landmarks[eye_landmarks[5]].y])
    v2 = math.dist([landmarks[eye_landmarks[2]].x, landmarks[eye_landmarks[2]].y],
                   [landmarks[eye_landmarks[4]].x, landmarks[eye_landmarks[4]].y])
    h = math.dist([landmarks[eye_landmarks[0]].x, landmarks[eye_landmarks[0]].y],
                  [landmarks[eye_landmarks[3]].x, landmarks[eye_landmarks[3]].y])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


class EyeTrackerThread(threading.Thread):
    """Background thread to capture webcam, run MediaPipe face mesh, and track eyes."""
    def __init__(self, frame_queue, app_state):
        super().__init__()
        self.frame_queue = frame_queue
        self.app_state = app_state
        self.running = True
        
        self.cam_index_change = queue.Queue()
        self.current_cam_idx = 0
        self.cam = None
        
        # Initialize face mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Dynamic eye-tracking calibration bounds
        self.min_x, self.max_x = 0.47, 0.53
        self.min_y, self.max_y = 0.47, 0.53
        self.padding = 0.008
        self.smoothed_x, self.smoothed_y = None, None

        self.last_click_time = 0.0
        self.click_cooldown = 0.8  # seconds

    def run(self):
        # Open initial camera
        self.cam = cv2.VideoCapture(self.current_cam_idx, cv2.CAP_DSHOW)
        
        while self.running:
            # Check if camera switch was requested
            if not self.cam_index_change.empty():
                try:
                    new_idx = self.cam_index_change.get_nowait()
                    if self.cam is not None:
                        self.cam.release()
                    self.cam = cv2.VideoCapture(new_idx, cv2.CAP_DSHOW)
                    self.current_cam_idx = new_idx
                    self.reset_calibration()
                except Exception as e:
                    print(f"Error switching camera: {e}")

            if self.cam is None or not self.cam.isOpened():
                time.sleep(0.1)
                continue

            success, frame = self.cam.read()
            if not success:
                time.sleep(0.03)
                continue

            # Process frame
            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = self.face_mesh.process(rgb_frame)
            landmark_points = output.multi_face_landmarks

            # Clear click feedback indicator after expiration
            if time.time() > self.app_state['feedback_expiry']:
                self.app_state['feedback_text'] = ""

            if landmark_points:
                landmarks = landmark_points[0].landmark

                # Draw eye outline points
                for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                    x = int(landmarks[idx].x * frame_w)
                    y = int(landmarks[idx].y * frame_h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

                # Calculate pupil position
                iris_x = sum([landmarks[i].x for i in range(474, 478)]) / 4.0
                iris_y = sum([landmarks[i].y for i in range(474, 478)]) / 4.0

                # Highlight pupil
                cv2.circle(frame, (int(iris_x * frame_w), int(iris_y * frame_h)), 4, (0, 255, 0), -1)

                # Dynamically expand calibration boundaries
                if iris_x < self.min_x: self.min_x = iris_x
                if iris_x > self.max_x: self.max_x = iris_x
                if iris_y < self.min_y: self.min_y = iris_y
                if iris_y > self.max_y: self.max_y = iris_y

                # Map eye position to screen coordinates
                range_x = max(self.max_x - self.min_x, 0.01)
                range_y = max(self.max_y - self.min_y, 0.01)

                norm_x = (iris_x - (self.min_x + self.padding)) / (range_x - 2 * self.padding)
                norm_y = (iris_y - (self.min_y + self.padding)) / (range_y - 2 * self.padding)

                norm_x = max(0.0, min(1.0, norm_x))
                norm_y = max(0.0, min(1.0, norm_y))

                target_x = norm_x * self.app_state['screen_w']
                target_y = norm_y * self.app_state['screen_h']

                # Smooth movements
                alpha = self.app_state['alpha']
                if self.smoothed_x is None:
                    self.smoothed_x = target_x
                    self.smoothed_y = target_y
                else:
                    self.smoothed_x = alpha * target_x + (1 - alpha) * self.smoothed_x
                    self.smoothed_y = alpha * target_y + (1 - alpha) * self.smoothed_y

                # Move cursor if active
                if self.app_state['tracking_active']:
                    try:
                        pyautogui.moveTo(int(self.smoothed_x), int(self.smoothed_y))
                    except pyautogui.FailSafeException:
                        self.app_state['tracking_active'] = False
                        self.app_state['feedback_text'] = "FAILSAFE TRIGGERED"
                        self.app_state['feedback_expiry'] = time.time() + 2.0

                # Click detection
                left_ear = calculate_ear(LEFT_EYE_INDICES, landmarks)
                right_ear = calculate_ear(RIGHT_EYE_INDICES, landmarks)
                
                ear_closed_threshold = self.app_state['ear_threshold']
                ear_open_threshold = ear_closed_threshold + 0.05
                current_time = time.time()

                if current_time - self.last_click_time > self.click_cooldown:
                    # Left wink -> Left Click
                    if left_ear < ear_closed_threshold and right_ear > ear_open_threshold:
                        pyautogui.click(button='left')
                        self.last_click_time = current_time
                        self.app_state['feedback_text'] = "LEFT CLICK!"
                        self.app_state['feedback_expiry'] = current_time + 0.5
                    # Right wink -> Right Click
                    elif right_ear < ear_closed_threshold and left_ear > ear_open_threshold:
                        pyautogui.click(button='right')
                        self.last_click_time = current_time
                        self.app_state['feedback_text'] = "RIGHT CLICK!"
                        self.app_state['feedback_expiry'] = current_time + 0.5

            # Render feedback status directly on frame
            if self.app_state['feedback_text']:
                cv2.putText(frame, self.app_state['feedback_text'], (frame_w // 2 - 80, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

            # Send frame to queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            self.frame_queue.put(frame)

        # Cleanup
        if self.cam is not None:
            self.cam.release()

    def switch_camera(self, idx):
        self.cam_index_change.put(idx)

    def reset_calibration(self):
        # Reset eye bounding box limits
        self.min_x, self.max_x = 0.47, 0.53
        self.min_y, self.max_y = 0.47, 0.53
        self.smoothed_x, self.smoothed_y = None, None
        self.app_state['feedback_text'] = "RE-CALIBRATED!"
        self.app_state['feedback_expiry'] = time.time() + 0.8
        print("Calibrated eye centers.")


class EyeTrackerApp(ctk.CTk):
    """Main CustomTkinter Dashboard Application."""
    def __init__(self):
        super().__init__()

        # Appearance Settings
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Eye-Tracking Control Dashboard")
        self.geometry("980x640")
        self.resizable(False, False)

        # Thread state shared data
        self.app_state = {
            'screen_w': pyautogui.size()[0],
            'screen_h': pyautogui.size()[1],
            'tracking_active': False,  # Pause at start to prevent immediate cursor stealing
            'alpha': 0.20,
            'ear_threshold': 0.18,
            'feedback_text': "",
            'feedback_expiry': 0.0
        }

        # Comm Queue
        self.frame_queue = queue.Queue()
        self.tracker_thread = EyeTrackerThread(self.frame_queue, self.app_state)
        self.tracker_thread.start()

        # Keyboard Bindings
        self.bind("<Key>", self.handle_keypress)

        # Create Layout
        self.grid_columnconfigure(0, weight=0)  # Sidebar
        self.grid_columnconfigure(1, weight=1)  # Frame Window
        self.grid_rowconfigure(0, weight=1)

        self.create_sidebar()
        self.create_main_content()

        # Start dynamic scanning for cameras
        self.cameras = [0]  # Default starting list
        self.scan_cameras_in_background()

        # Frame loop
        self.update_video_loop()

    def create_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        # App Title
        title_label = ctk.CTkLabel(sidebar, text="EYE TRACKING HUB", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        # --- CAMERA CONFIG ---
        cam_frame = ctk.CTkFrame(sidebar)
        cam_frame.grid(row=1, column=0, padx=15, pady=10, sticky="ew")
        
        cam_section_label = ctk.CTkLabel(cam_frame, text="CAMERA SOURCE", font=ctk.CTkFont(size=12, weight="bold"))
        cam_section_label.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="w")

        self.cam_selector = ctk.CTkOptionMenu(cam_frame, values=["Camera 0"], command=self.on_camera_select)
        self.cam_selector.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="ew")

        # --- SYSTEM CONTROLS ---
        ctrl_frame = ctk.CTkFrame(sidebar)
        ctrl_frame.grid(row=2, column=0, padx=15, pady=10, sticky="ew")

        ctrl_section_label = ctk.CTkLabel(ctrl_frame, text="SYSTEM CONTROLS", font=ctk.CTkFont(size=12, weight="bold"))
        ctrl_section_label.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="w")

        self.btn_toggle = ctk.CTkButton(ctrl_frame, text="Start Tracking [S]", fg_color="green", hover_color="#006400", command=self.toggle_tracking)
        self.btn_toggle.grid(row=1, column=0, padx=15, pady=5, sticky="ew")

        btn_recalib = ctk.CTkButton(ctrl_frame, text="Recalibrate Center [R]", command=self.recalibrate)
        btn_recalib.grid(row=2, column=0, padx=15, pady=(5, 15), sticky="ew")

        # --- TUNING SETTINGS ---
        tune_frame = ctk.CTkFrame(sidebar)
        tune_frame.grid(row=3, column=0, padx=15, pady=10, sticky="ew")

        tune_section_label = ctk.CTkLabel(tune_frame, text="SENSITIVITY PARAMETERS", font=ctk.CTkFont(size=12, weight="bold"))
        tune_section_label.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="w")

        # Smoothing Slider
        self.lbl_smoothing = ctk.CTkLabel(tune_frame, text=f"Cursor Smoothing: {self.app_state['alpha']:.2f}")
        self.lbl_smoothing.grid(row=1, column=0, padx=15, pady=(5, 0), sticky="w")
        self.slider_smoothing = ctk.CTkSlider(tune_frame, from_=0.05, to=0.8, number_of_steps=75, command=self.on_smoothing_change)
        self.slider_smoothing.set(self.app_state['alpha'])
        self.slider_smoothing.grid(row=2, column=0, padx=15, pady=(0, 10), sticky="ew")

        # Wink Slider
        self.lbl_wink = ctk.CTkLabel(tune_frame, text=f"Wink Sensitivity: {self.app_state['ear_threshold']:.2f}")
        self.lbl_wink.grid(row=3, column=0, padx=15, pady=(5, 0), sticky="w")
        self.slider_wink = ctk.CTkSlider(tune_frame, from_=0.10, to=0.25, number_of_steps=30, command=self.on_wink_change)
        self.slider_wink.set(self.app_state['ear_threshold'])
        self.slider_wink.grid(row=4, column=0, padx=15, pady=(0, 15), sticky="ew")

    def create_main_content(self):
        main_area = ctk.CTkFrame(self)
        main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        main_area.grid_columnconfigure(0, weight=1)
        main_area.grid_rowconfigure(1, weight=1)

        # Video feed frame holder
        self.lbl_video = ctk.CTkLabel(main_area, text="Starting webcam feed...", fg_color="black")
        self.lbl_video.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        # Visual Dashboard Status Bar
        self.lbl_status = ctk.CTkLabel(
            main_area, 
            text="STATUS: PAUSED", 
            fg_color="#D2143A", 
            text_color="white", 
            font=ctk.CTkFont(size=14, weight="bold"),
            height=35
        )
        self.lbl_status.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="ew")

        # Quick tip instructions label at bottom
        self.lbl_help = ctk.CTkLabel(
            main_area, 
            text="Tip: Wink with Left Eye to Left Click, Wink with Right Eye to Right Click.\nHold mouse in screen corner to trigger Failsafe emergency stop.",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color="gray"
        )
        self.lbl_help.grid(row=2, column=0, padx=20, pady=(5, 20))

    def on_camera_select(self, choice):
        idx = int(choice.split(" ")[-1])
        self.tracker_thread.switch_camera(idx)
        print(f"Switched camera feed to Camera Index {idx}.")

    def scan_cameras_in_background(self):
        def scan_job():
            open_cams = []
            for i in range(5):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        open_cams.append(i)
                    cap.release()
            
            if open_cams:
                self.cameras = open_cams
                # Update selector options
                menu_values = [f"Camera {c}" for c in open_cams]
                self.cam_selector.configure(values=menu_values)
                print(f"Found active camera sources: {open_cams}")

        # Start thread
        threading.Thread(target=scan_job, daemon=True).start()

    def toggle_tracking(self):
        self.app_state['tracking_active'] = not self.app_state['tracking_active']
        self.update_control_ui_state()

    def recalibrate(self):
        self.tracker_thread.reset_calibration()

    def on_smoothing_change(self, val):
        self.app_state['alpha'] = val
        self.lbl_smoothing.configure(text=f"Cursor Smoothing: {val:.2f}")

    def on_wink_change(self, val):
        self.app_state['ear_threshold'] = val
        self.lbl_wink.configure(text=f"Wink Sensitivity: {val:.2f}")

    def update_control_ui_state(self):
        if self.app_state['tracking_active']:
            self.lbl_status.configure(text="STATUS: EYE TRACKING ACTIVE", fg_color="green")
            self.btn_toggle.configure(text="Pause Tracking [S]", fg_color="red", hover_color="#8B0000")
        else:
            self.lbl_status.configure(text="STATUS: PAUSED", fg_color="#D2143A")
            self.btn_toggle.configure(text="Start Tracking [S]", fg_color="green", hover_color="#006400")

    def handle_keypress(self, event):
        key = event.keysym.lower()
        if key == 's':
            self.toggle_tracking()
        elif key == 'r':
            self.recalibrate()
        elif key == 'q' or key == 'escape':
            self.quit_app()

    def update_video_loop(self):
        # Update active tracking button state dynamically if changed by failsafe
        if not self.app_state['tracking_active'] and self.btn_toggle.cget("text").startswith("Pause"):
            self.update_control_ui_state()

        if not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
                # Convert BGR frame from opencv to RGB PIL Image
                img = Image.fromarray(frame)
                
                # Resize image slightly if needed to fit the frame holder cleanly
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                
                img_tk = ImageTk.PhotoImage(image=img)
                self.lbl_video.configure(image=img_tk, text="")
                self.lbl_video.image = img_tk  # Keep reference
            except Exception as e:
                print(f"Frame rendering error: {e}")

        # Polling rate of 30 ms
        self.after(30, self.update_video_loop)

    def quit_app(self):
        print("Shutting down dashboard...")
        self.tracker_thread.running = False
        self.tracker_thread.join(timeout=1.0)
        self.destroy()


def main():
    app = EyeTrackerApp()
    app.mainloop()


if __name__ == "__main__":
    main()


