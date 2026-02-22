import cv2
from deepface import DeepFace
import os
from datetime import datetime

# 1. Path Fix
base_path = os.path.dirname(os.path.abspath(__file__))
emojis_folder = os.path.join(base_path, "emojis")

emotion_emoji_map = {
    'happy': 'happy.png',
    'sad': 'sad.png',
    'angry': 'angry.png',
    'neutral': 'neutral.png',
    'surprise': 'surprise.png',
    'fear': 'fear.png',
    'disgust': 'disgust.png'
}

# Load emojis
emojis = {}
for emotion, filename in emotion_emoji_map.items():
    path = os.path.join(emojis_folder, filename)
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            emojis[emotion] = img
            print(f"✅ Loaded {filename}")
        else:
            print(f"❌ Failed to load {filename}")
    else:
        print(f"⚠️ Missing file: {path}")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        for face in results:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            dominant_emotion = face['dominant_emotion']
            accuracy = face['emotion'][dominant_emotion]

            # UI Improvements
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{dominant_emotion} ({accuracy:.1f}%)", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Emoji Overlay Logic (More Robust)
            if dominant_emotion in emojis:
                emoji_img = emojis[dominant_emotion]
                
                # Emoji size (40%)
                e_sz = int(w * 0.4)
                emoji_resized = cv2.resize(emoji_img, (e_sz, e_sz))

                # Overlay location: Muna box eka athulema uda corner eke
                # Frame eken pita yanne nathuwa ROI eka hadagamu
                y_offset = y + 5
                x_offset = x + 5

                if emoji_resized.shape[2] == 4: # PNG eka transparent nam
                    b, g, r, a = cv2.split(emoji_resized)
                    overlay = cv2.merge((b, g, r))
                    mask = cv2.merge((a, a, a))

                    # Frame eka athule ROI eka checking
                    if y_offset + e_sz < frame.shape[0] and x_offset + e_sz < frame.shape[1]:
                        roi = frame[y_offset:y_offset+e_sz, x_offset:x_offset+e_sz]
                        bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
                        fg = cv2.bitwise_and(overlay, mask)
                        frame[y_offset:y_offset+e_sz, x_offset:x_offset+e_sz] = cv2.add(bg, fg)

    except Exception as e:
        pass

    cv2.imshow("🎭 Advanced Emotion Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", frame)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
