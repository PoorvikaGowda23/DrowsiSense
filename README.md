# Driver Drowsiness Detection System 😴🚗

A real-time drowsiness detection system that uses **computer vision** and **deep learning** to monitor driver alertness and prevent accidents caused by fatigue.

## 🌟 Features

- **👁️ Eye Aspect Ratio (EAR) Analysis**
  - Detects prolonged eye closure
  - Configurable threshold for sensitivity
- **👄 Mouth Aspect Ratio (MAR) Analysis**
  - Detects yawning as fatigue indicator
  - Adjustable yawn duration threshold
- **🤖 CNN-Based Verification**
  - Deep learning model for additional verification
  - 4-class classification (alert, drowsy, yawning, etc.)
- **🚨 Real-Time Alerts**
  - Visual indicators on video feed
  - Audio alarm for immediate attention
- **📊 Performance Metrics**
  - Real-time graphs of EAR/MAR values
  - Detection event timeline

## 🛠 Tech Stack

| Component               | Technology                         |
|-------------------------|------------------------------------|
| Computer Vision         | OpenCV, dlib                       |
| Deep Learning           | TensorFlow/Keras CNN               |
| Facial Landmarks        | 68-point shape predictor           |
| Web Interface           | Streamlit                          |
| Audio Alerts            | Pygame                             |
| Data Processing         | NumPy, imutils                     |

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip package manager
- Webcam or video file for testing

### Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-username/driver-drowsiness-detection.git
    cd driver-drowsiness-detection
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download pre-trained models**
    - Place `shape_predictor_68_face_landmarks.dat` in project root
    - Ensure CNN model (`drowsiness_cnn_model.h5`) is in specified path

4. **Run the application**
    ```bash
    streamlit run app.py
    ```

5. **Access the application**
    Open in your browser.

## 📂 Project Structure
