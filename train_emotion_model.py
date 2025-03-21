# train_emotion_model.py
import os
import zipfile
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump
import subprocess

# 1. Download FER2013 dataset from Kaggle
def download_fer2013():
    print("Downloading FER2013 dataset from Kaggle...")
    cmd = ["kaggle", "datasets", "download", "-d", "msambare/fer2013", "-p", "."]
    subprocess.run(cmd, check=True)
    
    # Unzip fer2013.zip
    if not os.path.exists("fer2013.csv"):
        with zipfile.ZipFile("fer2013.zip", "r") as zip_ref:
            zip_ref.extractall(".")
    print("FER2013 dataset downloaded and extracted.")

# 2. Extract face landmarks using MediaPipe
def extract_landmarks(csv_file="fer2013.csv", limit_per_label=500):
    """
    Reads the CSV, uses MediaPipe to get face landmarks for each image.
    limit_per_label = max images per emotion label to speed up training.
    """
    import pandas as pd
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    
    data = pd.read_csv(csv_file)
    X, y = [], []
    
    # The CSV has columns: emotion, pixels, usage
    # Emotions: 0=angry,1=disgust,2=fear,3=happy,4=sad,5=surprise,6=neutral
    # We'll map them to text labels
    label_map = {
        0: "angry",
        1: "disgust",  # often merged with 'angry' or 'sad'
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral"
    }
    
    count_per_label = {k: 0 for k in label_map.values()}
    
    for i, row in data.iterrows():
        emotion_id = row["emotion"]
        if emotion_id not in label_map:
            continue
        emotion_label = label_map[emotion_id]
        
        # Respect limit_per_label
        if count_per_label[emotion_label] >= limit_per_label:
            continue
        
        # Convert pixels to a 48x48 grayscale image
        pixels = list(map(int, row["pixels"].split()))
        img = np.array(pixels, dtype=np.uint8).reshape(48, 48)
        
        # Convert to BGR for MediaPipe
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        results = mp_face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            # take first face only
            landmarks = results.multi_face_landmarks[0].landmark
            features = []
            for lm in landmarks:
                features.append(lm.x)
                features.append(lm.y)
            X.append(features)
            y.append(emotion_label)
            count_per_label[emotion_label] += 1
    
    return np.array(X), np.array(y)

def train_model(X, y):
    """
    Perform PCA + Logistic Regression, return model + pca
    """
    # PCA
    pca = PCA(n_components=50)
    X_reduced = pca.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.2f}")
    return model, pca

if __name__ == "__main__":
    # 1. Download dataset
    if not os.path.exists("fer2013.csv"):
        download_fer2013()
    
    # 2. Extract landmarks
    print("Extracting landmarks from FER2013...")
    X, y = extract_landmarks("fer2013.csv", limit_per_label=300)  # you can adjust limit
    print(f"Extracted {len(X)} samples.")
    
    # 3. Train model
    print("Training model...")
    model, pca = train_model(X, y)
    
    # 4. Save model
    dump(model, "emotion_model.pkl")
    dump(pca, "pca_transform.pkl")
    print("Model + PCA saved successfully!")
