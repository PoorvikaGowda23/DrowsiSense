import cv2
import dlib
import imutils
from imutils import face_utils

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load dlib's shape predictor for facial landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Open the webcam
cap = cv2.VideoCapture(0)

# Set fixed FPS to avoid issues
fps = 20  
frame_width = 600  # Resized frame width
frame_height = int(cap.get(4) * (600 / cap.get(3)))  # Scale height accordingly

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # More compatible codec
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Resize frame for consistency
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face(s)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        rect = dlib.rectangle(x, y, x + w, y + h)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[36:42]
        right_eye = shape[42:48]
        mouth = shape[48:68]

        # Draw bounding boxes
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)
        cv2.polylines(frame, [mouth], True, (0, 0, 255), 2)

    # Write the frame to video file
    if frame is not None:
        out.write(frame)
    else:
        print("Warning: Empty frame captured!")

    # Display the output
    cv2.imshow('Driver Face Tracking', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video recording complete. Now attempting playback...")

# ===============================
# 🎥 **Playback Test**
# ===============================
cap = cv2.VideoCapture('output.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Playback complete or error in video.")
        break
    cv2.imshow('Recorded Video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
