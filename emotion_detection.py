import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained Mini-XCEPTION model
model = load_model("mini_xception.h5", compile=False)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define colors for each emotion (BGR format)
emotion_colors = {
    'Angry': (0, 0, 255),        # Red
    'Disgust': (0, 255, 0),      # Green
    'Fear': (255, 0, 0),         # Blue
    'Happy': (0, 255, 255),      # Yellow
    'Sad': (255, 0, 255),        # Magenta
    'Surprise': (255, 255, 0),   # Cyan
    'Neutral': (255, 255, 255)   # White
}

# Load face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # Create a sidebar for emotion bars
    bar_img = np.zeros((480, 200, 3), dtype=np.uint8)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict emotion
            preds = model.predict(roi, verbose=0)[0]
            emotion_probability = np.max(preds)
            label = emotion_labels[preds.argmax()]
            color = emotion_colors[label]

            # Draw rectangle and emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Draw emotion bars on sidebar
            for i, (emotion, prob) in enumerate(zip(emotion_labels, preds)):
                bar_width = int(prob * 150)
                bar_color = emotion_colors[emotion]
                cv2.rectangle(bar_img, (10, 30 * i + 10), (10 + bar_width, 30 * i + 30), bar_color, -1)
                cv2.putText(bar_img, f"{emotion} {int(prob * 100)}%", (10, 30 * i + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Combine webcam feed + sidebar
    combined = np.hstack((cv2.resize(frame, (640, 480)), bar_img))

    # Show the final frame
    cv2.imshow('Emotion Detector with UI', combined)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
