# emotion_music_gui.py

import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
import random
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import webbrowser
from joblib import load

# Load the trained model + PCA
model = load("emotion_model.pkl")
pca = load("pca_transform.pkl")

# Initialize MediaPipe FaceMesh for real-time inference
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Spotify credentials (Replace with your own keys)
SPOTIFY_CLIENT_ID = "f50119de959f47c493435016de459412"
SPOTIFY_CLIENT_SECRET = "d82cccab45d04b37bbb7e7c422731461"

# Authenticate with Spotify
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
)

class EmotionMusicGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initCamera()

    def initUI(self):
        self.setWindowTitle("AI-Driven Emotional Music Generator")
        self.setGeometry(100, 100, 800, 600)

        self.video_label = QLabel(self)
        self.emotion_label = QLabel("Emotion: Detecting...", self)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.emotion_label)
        self.setLayout(layout)

    def initCamera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip horizontally for a mirror-like preview
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Predict emotion
            emotion = self.detect_emotion(frame_rgb)
            self.emotion_label.setText(f"Emotion: {emotion}")

            # Convert the frame back to display in PyQt
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def detect_emotion(self, frame_rgb):
        """Extract 468 landmarks with MediaPipe, run them through the PCA & model."""
        results = mp_face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Flatten (x,y) for all 468 landmarks -> 936 features
            features = []
            for lm in landmarks:
                features.append(lm.x)
                features.append(lm.y)

            features = np.array(features).reshape(1, -1)
            # Apply PCA
            features_reduced = pca.transform(features)
            # Predict emotion
            emotion = model.predict(features_reduced)[0]
            self.play_music(emotion)
            return emotion
        else:
            return "neutral"

    def play_music(self, emotion):
        """Search Spotify for a track matching the emotion and open it in the browser."""
        query = f"{emotion} mood music"
        results = sp.search(q=query, type='track', limit=5)
        if results['tracks']['items']:
            track = random.choice(results['tracks']['items'])
            webbrowser.open(track['external_urls']['spotify'])
        else:
            print("No song found for this emotion.")

    def closeEvent(self, event):
        """Release the camera on window close."""
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionMusicGUI()
    window.show()
    sys.exit(app.exec())
