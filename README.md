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
    git clone https://github.com/PoorvikaGowda23/DrowsiSense
    cd DrowsiSense
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


## 🎮 Usage

### ⚙️ Configuration
1. Adjust detection thresholds in the sidebar:
   - Eye closure duration
   - Yawn detection sensitivity
   - CNN probability threshold

### 🎥 Video Sources
- **Webcam**: Real-time detection from your camera
- **Upload Video**: Process pre-recorded footage

### 📊 Monitoring
- Real-time video feed with detection overlays
- EAR/MAR value graphs
- Detection event timeline

### 🚨 Alerts
- Visual indicators on detected drowsiness/yawns
- Audio alarm sounds when thresholds are exceeded

## 🏗️ Model Training

To retrain the CNN model:
1. Organize your dataset in `dataset_new/train` and `dataset_new/test` directories
2. Run:
   ```bash
   python cnn.py
   ```
3. The script will:
    - Preprocess images
    - Train the CNN model
    - Save updated weights to drowsiness_cnn_model.h5

## 📝 Notes
- For best results, ensure proper lighting on the driver's face
- System works best with frontal face views
- Performance may vary based on camera quality
