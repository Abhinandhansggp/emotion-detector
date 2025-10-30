import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load pre-trained model
model = load_model("mini_xception.h5", compile=False)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define colors for each emotion
emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 255, 0),
    'Fear': (255, 0, 0),
    'Happy': (0, 255, 255),
    'Sad': (255, 0, 255),
    'Surprise': (255, 255, 0),
    'Neutral': (255, 255, 255)
}

st.set_page_config(page_title="Emotion Detector", layout="wide")
st.title("😊 Emotion Detection Web App")
st.write("Upload an image or use webcam to detect emotions in faces.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Webcam capture
use_webcam = st.checkbox("Use Webcam")

def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = model.predict(roi, verbose=0)[0]
        label = emotion_labels[preds.argmax()]
        color = emotion_colors[label]

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image

if uploaded_file is not None:
    # Process uploaded image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    result_img = detect_emotion(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    st.image(result_img, channels="BGR")

elif use_webcam:
    st.write("Click 'Start' to use webcam.")
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Camera not working")
            break
        frame = detect_emotion(frame)
        FRAME_WINDOW.image(frame, channels="BGR")
    camera.release()
else:
    st.info("👆 Upload an image or use your webcam to get started!")
