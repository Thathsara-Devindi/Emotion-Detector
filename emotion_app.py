import cv2
from deepface import DeepFace
import os

# Emotion --> emoji mapping
emotion_emoji_map = {
    'happy': 'happy.png',
    'sad': 'sad.png',
    'angry': 'angry.png',
    'neutral': 'neutral.png',
    'surprise': 'surprise.png',
    'fear': 'fear.png',
    'disgust': 'disgust.png'
}

#load emojis
emojis = {}
for emotion, filename in emotion_emoji_map.items():
    path = os.path.join("emojis", filename)
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            emojis[emotion] = img
        else:
            print(f"⚠️ Cannot load {path}")
    else:
        print(f"⚠️ Missing {path}")

cap = cv2.VideoCapture(0)
print("🎥 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )
        dominant_emotion = result[0]['dominant_emotion']

        if dominant_emotion in emojis:
            emoji = cv2.resize(emojis[dominant_emotion], (120, 120))

            if emoji.shape[2] == 4:
                b, g, r, a = cv2.split(emoji)
                overlay = cv2.merge((b, g, r))
                mask = cv2.merge((a, a, a))

                roi = frame[10:130, 10:130]
                bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
                fg = cv2.bitwise_and(overlay, mask)
                frame[10:130, 10:130] = cv2.add(bg, fg)

    except Exception as e:
        print("⚠️ Detection error:", e)

    cv2.imshow("🎭 Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
