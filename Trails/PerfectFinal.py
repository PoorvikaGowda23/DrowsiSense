import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils
from tensorflow.keras.models import load_model   # type: ignore
import threading
import os
from collections import deque
import pygame
import streamlit as st
import tempfile
import time

# Initialize pygame mixer for audio control
pygame.mixer.init()

# Streamlit page configuration
st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="😴")

# Sidebar for configuration
st.sidebar.title("Drowsiness Detection Settings")

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Improved function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(mouth):
    # Vertical distances
    vertical1 = np.linalg.norm(mouth[2] - mouth[10])  # top to bottom
    vertical2 = np.linalg.norm(mouth[4] - mouth[8])   # top to bottom
    
    # Horizontal distance
    horizontal = np.linalg.norm(mouth[0] - mouth[6])  # left to right
    
    # Calculate MAR using average of vertical distances divided by horizontal
    mar = (vertical1 + vertical2) / (2.0 * horizontal)
    return mar

# Alarm sound control
alarm_sound = None
alarm_playing = False
last_alarm_time = 0
alarm_cooldown = 5  # seconds

def play_alarm():
    global alarm_playing, alarm_sound, last_alarm_time
    current_time = time.time()
    
    # Check if alarm is not already playing and cooldown has passed
    if not alarm_playing and (current_time - last_alarm_time) > alarm_cooldown:
        alarm_path = r"C:\Users\user\Desktop\VSC Folder\SEM6\DL Project\DD_Final\alarm.wav"
        
        if os.path.exists(alarm_path):
            try:
                alarm_sound = pygame.mixer.Sound(alarm_path)
                alarm_sound.play(-1)  # Loop indefinitely until stopped
                alarm_playing = True
                last_alarm_time = current_time
            except Exception as e:
                st.warning(f"Error playing alarm: {e}")
        else:
            st.warning(f"Error: Alarm sound file not found at {alarm_path}")

def stop_alarm():
    global alarm_playing, alarm_sound
    if alarm_playing and alarm_sound is not None:
        alarm_sound.stop()
        alarm_playing = False

# Streamlit-specific configuration
st.sidebar.header("Thresholds")
FRAME_THRESHOLD = st.sidebar.slider("Frame Threshold", 5, 30, 20)
FRAME_CONSECUTIVE_THRESHOLD = st.sidebar.slider("Consecutive Frames", 1, 15, 10)
CNN_PROB_THRESHOLD = st.sidebar.slider("CNN Probability Threshold", 0.5, 1.0, 0.7)
EAR_THRESHOLD = st.sidebar.slider("Eye Aspect Ratio Threshold", 0.1, 0.5, 0.2)
EAR_TIME_THRESHOLD = st.sidebar.slider("Eye Closure Duration (seconds)", 1, 5, 3)
MAR_THRESHOLD = st.sidebar.slider("Mouth Aspect Ratio Threshold", 0.5, 1.0, 0.75)
YAWN_FRAME_THRESHOLD = st.sidebar.slider("Yawn Frame Threshold", 5, 20, 10)

# Load models and classifiers
@st.cache_resource
def load_models():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    model = load_model(r"C:\Users\user\Desktop\VSC Folder\SEM6\DL Project\DD_Final\drowsiness_cnn_model.h5")
    return face_cascade, predictor, model

face_cascade, predictor, model = load_models()

class DrowsinessDetector:
    def __init__(self, fps=30):
        self.fps = fps
        self.eye_closure_start_time = None
        self.ear_buffer = deque(maxlen=10)
        self.is_drowsy = False
        self.drowsy_start_time = None
        self.yawn_frames = 0
        self.yawn_detected = False

    def detect_drowsiness(self, frame):
        # Core drowsiness detection logic
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        drowsy_detected = False
        yawn_detected = False

        for (x, y, w, h) in faces:
            rect = dlib.rectangle(x, y, x + w, y + h)
            
            # Detect facial landmarks
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Extract eye and mouth landmarks
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            mouth = shape[48:68]

            # Compute EAR & MAR
            ear_left = calculate_ear(left_eye)
            ear_right = calculate_ear(right_eye)
            ear = (ear_left + ear_right) / 2.0
            mar = calculate_mar(mouth)

            # Add to EAR buffer for smoothing
            self.ear_buffer.append(ear)
            smoothed_ear = np.mean(self.ear_buffer)

            # Draw eye and mouth landmarks
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)
            cv2.polylines(frame, [mouth], True, (0, 0, 255), 2)

            # Eye closure detection with time-based approach
            current_time = time.time()

            if smoothed_ear < EAR_THRESHOLD:
                # First time detecting low EAR
                if self.eye_closure_start_time is None:
                    self.eye_closure_start_time = current_time
                
                # Check duration of eye closure
                closure_duration = current_time - self.eye_closure_start_time
                if closure_duration >= EAR_TIME_THRESHOLD:
                    drowsy_detected = True
                    cv2.putText(frame, f"DROWSY (Eyes Closed {closure_duration:.1f}s)", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Reset eye closure timer if eyes are open
                self.eye_closure_start_time = None

            # Yawn detection logic with frame persistence
            if mar > MAR_THRESHOLD:
                self.yawn_frames += 1
                if self.yawn_frames >= YAWN_FRAME_THRESHOLD:
                    yawn_detected = True
                    cv2.putText(frame, "YAWN DETECTED!", (x, y - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                self.yawn_frames = max(0, self.yawn_frames - 1)

            # Convert ROI to 3 channels for CNN
            roi = frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, (64, 64))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = roi / 255.0
            roi = np.expand_dims(roi, axis=0)

            # CNN Prediction (optional additional verification)
            prediction = model.predict(roi, verbose=0)[0]
            drowsy_prob = prediction[1]  # Assuming class 1 is drowsy

            # Combine CNN prediction with other indicators
            if drowsy_prob > CNN_PROB_THRESHOLD:
                drowsy_detected = True
                cv2.putText(frame, "CNN: DROWSY", (x, y - 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return frame, drowsy_detected, yawn_detected

def main():
    st.title("🚗 Driver Drowsiness Detection System")
    
    # Video source selection
    video_source = st.sidebar.selectbox("Select Video Source", 
        ["Webcam", "Upload Video"])
    
    if video_source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi"])
        if uploaded_file is not None:
            # Save uploaded file to temporary location
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            st.warning("Please upload a video file.")
            return

    # Get FPS for time-based calculations
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # default to 30 if can't be determined

    # Initialize drowsiness detector
    detector = DrowsinessDetector(fps)

    # Streamlit video display
    frame_placeholder = st.empty()
    stop_button = st.sidebar.button("Stop")

    # Metrics
    drowsy_count = 0
    yawn_count = 0
    total_frames = 0

    # Status indicators
    status_placeholder = st.sidebar.empty()
    drowsy_count_placeholder = st.sidebar.empty()
    yawn_count_placeholder = st.sidebar.empty()

    # Main processing loop
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        frame = imutils.resize(frame, width=640)
        processed_frame, is_drowsy, is_yawn = detector.detect_drowsiness(frame)

        # Update metrics
        total_frames += 1
        if is_drowsy:
            drowsy_count += 1
            # Trigger alarm for drowsiness
            play_alarm()
        if is_yawn:
            yawn_count += 1
            # Also trigger alarm for yawn
            play_alarm()

        # Update status
        status_text = "🟢 Normal" if not (is_drowsy or is_yawn) else "🔴 Alert!"
        status_placeholder.markdown(f"**Status:** {status_text}")
        drowsy_count_placeholder.markdown(f"**Drowsy Frames:** {drowsy_count}")
        yawn_count_placeholder.markdown(f"**Yawn Frames:** {yawn_count}")

        # Display processed frame
        frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), 
                                channels="RGB")

        # Stop alarm if no drowsiness or yawn detected for a while
        if not is_drowsy and not is_yawn:
            stop_alarm()

        # Break loop if stop button pressed
        if stop_button:
            break

    # Cleanup
    cap.release()
    pygame.mixer.quit()
    st.success("Detection Stopped")

if __name__ == "__main__":
    main()