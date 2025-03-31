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

# Initialize pygame mixer for audio control
pygame.mixer.init()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load dlib's shape predictor for facial landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load pre-trained CNN model for classification
model = load_model(r"C:\Users\user\Desktop\VSC Folder\SEM6\DL Project\DD_Final\drowsiness_cnn_model.h5")

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

# Function to play alarm sound with stoppable control
alarm_sound = None
alarm_playing = False

def play_alarm():
    global alarm_playing, alarm_sound
    alarm_path = r"C:\Users\user\Desktop\VSC Folder\SEM6\DL Project\DD_Final\alarm.wav"
    
    if os.path.exists(alarm_path):
        if not alarm_playing:
            alarm_sound = pygame.mixer.Sound(alarm_path)
            alarm_sound.play(-1)  # Loop indefinitely until stopped
            alarm_playing = True
    else:
        print(f"Error: Alarm sound file not found at {alarm_path}")

def stop_alarm():
    global alarm_playing, alarm_sound
    if alarm_playing and alarm_sound is not None:
        alarm_sound.stop()
        alarm_playing = False

# Drowsiness Detection Thresholds
FRAME_THRESHOLD = 15  # Number of frames to trigger alarm
FRAME_CONSECUTIVE_THRESHOLD = 5  # Consecutive frames needed for drowsiness
CNN_PROB_THRESHOLD = 0.7  # Lowered from 0.9 for better yawn detection
EAR_THRESHOLD = 0.3  # Default until baseline
MAR_THRESHOLD = 0.75  # Threshold for mouth opening
YAWN_FRAME_THRESHOLD = 10  # Frames mouth must be open to count as yawn

# Initialize video capture
cap = cv2.VideoCapture(0)
frame_count = 0
consecutive_drowsy_frames = 0
yawn_frame_count = 0

# Moving average buffers for EAR and MAR smoothing
ear_buffer = deque(maxlen=7)
mar_buffer = deque(maxlen=15)

# Baseline EAR/MAR Calculation
baseline_ear = 0
baseline_mar = 0
frame_count_baseline = 50
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face using Haar cascade
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

        # Add to buffers and compute moving average
        ear_buffer.append(ear)
        mar_buffer.append(mar)
        smoothed_ear = np.mean(ear_buffer)
        smoothed_mar = np.mean(mar_buffer)

        # Baseline EAR/MAR Calculation
        frame_counter += 1
        if frame_counter <= frame_count_baseline:
            baseline_ear += ear
            baseline_mar += mar
            if frame_counter == frame_count_baseline:
                baseline_ear /= frame_count_baseline
                baseline_mar /= frame_count_baseline
                EAR_THRESHOLD = max(baseline_ear * 0.8, 0.15)
                MAR_THRESHOLD = min(baseline_mar * 1.5, 0.8)
                print(f"Baseline EAR: {baseline_ear:.2f}, MAR: {baseline_mar:.2f}")
                print(f"Adjusted EAR_THRESHOLD: {EAR_THRESHOLD:.2f}, MAR_THRESHOLD: {MAR_THRESHOLD:.2f}")

        # Draw eye and mouth landmarks
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)
        cv2.polylines(frame, [mouth], True, (0, 0, 255), 2)

        # Convert ROI to 3 channels for CNN
        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = roi / 255.0
        roi = np.expand_dims(roi, axis=0)

        # CNN Prediction
        prediction = model.predict(roi)[0]
        drowsy_prob = prediction[1]  # Assuming class 1 is drowsy
        class_index = 1 if drowsy_prob > CNN_PROB_THRESHOLD else 0

        # Debug info
        print(f"Smoothed EAR: {smoothed_ear:.2f} (Threshold: {EAR_THRESHOLD:.2f}), Smoothed MAR: {smoothed_mar:.2f} (Threshold: {MAR_THRESHOLD:.2f}), CNN Drowsy Prob: {drowsy_prob:.2f}")

        # Yawn detection logic
        if smoothed_mar > MAR_THRESHOLD:
            yawn_frame_count += 1
            if yawn_frame_count >= YAWN_FRAME_THRESHOLD:
                yawn_detected = True
                cv2.putText(frame, "YAWN DETECTED!", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # Trigger alarm immediately when yawn is detected
                if not alarm_playing:
                    threading.Thread(target=play_alarm, daemon=True).start()
        else:
            yawn_frame_count = max(0, yawn_frame_count - 1)  # Decrement but don't go below 0
            yawn_detected = False

        # Eye closure detection logic
        if smoothed_ear < EAR_THRESHOLD:
            consecutive_drowsy_frames += 1
            if consecutive_drowsy_frames >= FRAME_CONSECUTIVE_THRESHOLD:
                drowsy_detected = True
                frame_count += 1
                cv2.putText(frame, "DROWSY (EYES)!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if frame_count >= FRAME_THRESHOLD:
                    if not alarm_playing:
                        threading.Thread(target=play_alarm, daemon=True).start()
        else:
            consecutive_drowsy_frames = 0
            frame_count = 0

    # Stop alarm if no drowsiness or yawn is detected
    if not drowsy_detected and not yawn_detected:
        stop_alarm()

    # Display output
    cv2.imshow('Driver Drowsiness Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()