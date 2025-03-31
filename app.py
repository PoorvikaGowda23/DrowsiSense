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
import matplotlib.pyplot as plt

# Initialize pygame mixer for audio control
pygame.mixer.init()

# Streamlit page configuration
st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="😴", layout="wide")

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
        alarm_path = r"C:\Users\user\Desktop\GITHUB\Driver Drowsiness Detection system\alarm.wav"
        
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
    model = load_model(r"C:\Users\user\Desktop\GITHUB\Driver Drowsiness Detection system\drowsiness_cnn_model.h5")
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
        self.ear_history = []
        self.mar_history = []
        self.drowsy_history = []
        self.yawn_history = []
        # Initialize with some default values to prevent empty graphs
        self.ear_history.append(0.3)
        self.mar_history.append(0.5)
        self.drowsy_history.append(0)
        self.yawn_history.append(0)

    def reset(self):
        """Reset all detection history"""
        self.eye_closure_start_time = None
        self.ear_buffer = deque(maxlen=10)
        self.is_drowsy = False
        self.drowsy_start_time = None
        self.yawn_frames = 0
        self.yawn_detected = False
        self.ear_history = []
        self.mar_history = []
        self.drowsy_history = []
        self.yawn_history = []
        # Initialize with some default values to prevent empty graphs
        self.ear_history.append(0.3)
        self.mar_history.append(0.5)
        self.drowsy_history.append(0)
        self.yawn_history.append(0)

    def detect_drowsiness(self, frame):
        # Core drowsiness detection logic
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        drowsy_detected = False
        yawn_detected = False
        current_ear = 0.3  # Default value when no face detected
        current_mar = 0.5  # Default value when no face detected

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
            current_ear = ear
            current_mar = mar

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

        # Store values for plotting
        self.ear_history.append(current_ear)
        self.mar_history.append(current_mar)
        self.drowsy_history.append(1 if drowsy_detected else 0)
        self.yawn_history.append(1 if yawn_detected else 0)

        return frame, drowsy_detected, yawn_detected

    def create_ratio_plot(self):
        # Create plot for EAR and MAR ratios
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot EAR and MAR values
        if len(self.ear_history) > 0:
            ax.plot(self.ear_history, label='EAR', color='blue')
        if len(self.mar_history) > 0:
            ax.plot(self.mar_history, label='MAR', color='green')
        
        # Add threshold lines
        ax.axhline(y=EAR_THRESHOLD, color='blue', linestyle='--', label='EAR Threshold')
        ax.axhline(y=MAR_THRESHOLD, color='green', linestyle='--', label='MAR Threshold')
        
        ax.set_title('EAR and MAR Ratios Over Time')
        ax.set_ylabel('Ratio Value')
        ax.set_xlabel('Frame Number')
        ax.legend()
        plt.tight_layout()
        return fig

    def create_detection_plot(self):
        # Create plot for detection events
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot detection events
        if len(self.drowsy_history) > 0:
            ax.plot(self.drowsy_history, label='Eye Closure', color='red')
        if len(self.yawn_history) > 0:
            ax.plot(self.yawn_history, label='Yawn', color='orange')
        
        ax.set_title('Detection Events Over Time')
        ax.set_ylabel('Detection (1=Detected)')
        ax.set_xlabel('Frame Number')
        ax.legend()
        plt.tight_layout()
        return fig

def main():
    st.title("🚗 Driver Drowsiness Detection System")
    
    # Initialize session state for restart functionality
    if 'restart' not in st.session_state:
        st.session_state.restart = False
    if 'detector' not in st.session_state:
        st.session_state.detector = None

    # Video source selection
    video_source = st.sidebar.selectbox("Select Video Source", 
        ["Webcam", "Upload Video"])
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    stop_button = col1.button("Stop")
    restart_button = col2.button("Restart")

    if restart_button:
        st.session_state.restart = True
        st.experimental_rerun()

    if st.session_state.restart:
        st.session_state.restart = False
        if st.session_state.detector:
            st.session_state.detector.reset()
        # Clear all the existing elements
        st.empty()
        # Rerun to start fresh
        st.experimental_rerun()

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
    if st.session_state.detector is None:
        st.session_state.detector = DrowsinessDetector(fps)
    detector = st.session_state.detector

    # Streamlit layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        frame_placeholder = st.empty()
    with col2:
        status_placeholder = st.empty()
        drowsy_count_placeholder = st.empty()
        yawn_count_placeholder = st.empty()

    # Placeholder for plots (will appear below the video)
    st.subheader("Performance Metrics")
    plot_placeholder1 = st.empty()
    plot_placeholder2 = st.empty()

    # Metrics
    drowsy_count = 0
    yawn_count = 0
    total_frames = 0

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
            play_alarm()
        if is_yawn:
            yawn_count += 1
            play_alarm()

        # Update status
        status_text = "🟢 Normal" if not (is_drowsy or is_yawn) else "🔴 Alert!"
        status_placeholder.markdown(f"**Status:** {status_text}")
        drowsy_count_placeholder.markdown(f"**Drowsy Frames:** {drowsy_count}")
        yawn_count_placeholder.markdown(f"**Yawn Frames:** {yawn_count}")

        # Display processed frame
        frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), 
                                channels="RGB")

        # Update plots periodically (every 10 frames to reduce computation)
        if total_frames % 10 == 0 or total_frames == 1:  # Also update on first frame
            ratio_plot = detector.create_ratio_plot()
            detection_plot = detector.create_detection_plot()
            plot_placeholder1.pyplot(ratio_plot)
            plot_placeholder2.pyplot(detection_plot)

        # Stop alarm if no drowsiness
        if not is_drowsy and not is_yawn:
            stop_alarm()

        # Break loop if stop button pressed
        if stop_button:
            break

    # Final plots after stopping
    ratio_plot = detector.create_ratio_plot()
    detection_plot = detector.create_detection_plot()
    plot_placeholder1.pyplot(ratio_plot)
    plot_placeholder2.pyplot(detection_plot)
    
    # Cleanup
    cap.release()
    pygame.mixer.quit()
    
    if stop_button:
        st.success("Detection Stopped")
        st.button("Start Again", on_click=lambda: st.session_state.update(restart=True))

if __name__ == "__main__":
    main()