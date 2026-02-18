# Emotion-Detector
# Real-Time Emotion Detector with Emoji Overlay 🎭

This is a Python-based computer vision project that uses **DeepFace** and **OpenCV** to detect human emotions in real-time through a webcam. The application not only identifies the emotion but also overlays a corresponding emoji on the video feed.

## 🚀 Features
* Real-time Detection: High-speed emotion analysis using the DeepFace framework.
* Emoji Feedback: Dynamically displays emojis based on the detected emotion.
* User-Friendly: Simple "press 'q' to quit interface.
* Robustness: Uses `enforce_detection=False` to avoid crashes.

## 🛠️ Tech Stack
* Language: Python
* Libraries: OpenCV, DeepFace, TensorFlow

## 📂 Project Structure
```text
Emotion-Detector/
├── main.py              # The main script with emoji overlay
├── emojis/              # Folder containing emoji PNG files
│   ├── happy.png
│   ├── sad.png
│   └── ... 
└── README.md            # Project documentation
