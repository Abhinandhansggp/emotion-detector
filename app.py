import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your pre-trained model
model = load_model("mini_xception.h5", compile=False)

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Define custom video processor
class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)
                preds = model.predict(roi)[0]
                label = emotion_labels[preds.argmax()]
                label_position = (x, y)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

st.title("😊 Real-Time Emotion Detection Web App")

st.markdown("This app uses your webcam for live emotion detection.")

webrtc_streamer(
    key="emotion",
    video_transformer_factory=EmotionDetector
)
