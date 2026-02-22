# 🎭 Advanced Emotion Detector with Emoji Overlay

An interactive real-time facial emotion recognition system developed using **Python**, **OpenCV**, and **DeepFace**. This application detects multiple faces, predicts emotions, and overlays corresponding emojis with real-time accuracy percentages.

## 🚀 Features
* **Multi-Face Detection:** Capable of detecting and analyzing multiple faces simultaneously.
* **Emoji Overlay:** Dynamically places transparent emojis over detected faces based on the dominant emotion (Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust).
* **Real-time Accuracy:** Displays the emotion label and its confidence percentage above each face.
* **Screenshot Function:** Press the **'s'** key to capture a screenshot of the current frame with all overlays.
* **Modern UI:** Features green bounding boxes and clear text labels for a professional look.

## 🛠️ Tech Stack
* **Language:** Python 3.11+
* **Computer Vision:** OpenCV (`cv2`)
* **Deep Learning Framework:** DeepFace (Backbone: TensorFlow/Keras)
* **Assets:** Custom transparent PNG emojis

## 📁 Project Structure
```text
Emotion-Detector/
├── emojis/               # Folder containing emoji PNGs (happy.png, sad.png, etc.)
├── emotion_app.py        # Main application script with emoji overlay
├── emotion-detector.py   # Simple text-based emotion detector
├── webcam_test.py        # Script to test webcam functionality
└── README.md             # Project documentation
