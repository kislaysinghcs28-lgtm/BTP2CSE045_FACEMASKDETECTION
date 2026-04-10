import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import threading
from collections import deque

# Load trained model
model = load_model("mask_model.keras")

# Labels
labels = ["Mask", "No Mask"]

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

# Sound function
def play_sound(sound_path):
    if os.path.exists(sound_path):
        threading.Thread(
            target=lambda: os.system(f"afplay '{sound_path}'"),
            daemon=True
        ).start()

# Buffer for smoothing
prediction_buffer = deque(maxlen=3)

# Last detected label
last_label = -1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face, verbose=0)
        label = np.argmax(pred)
        confidence = np.max(pred)

        # Ignore low confidence predictions
        if confidence < 0.80:
            continue

        # Add to buffer
        prediction_buffer.append(label)

        # Majority vote
        final_label = max(set(prediction_buffer), key=prediction_buffer.count)

        text = f"{labels[final_label]}: {confidence:.2f}"
        color = (0, 255, 0) if final_label == 0 else (0, 0, 255)

        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # INSTANT SOUND SWITCH (NO DELAY)
        if final_label != last_label:
            if final_label == 0:
                play_sound("")
            else:
                play_sound("alert.wav")

            last_label = final_label

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()