import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump

def main():
    # Path to dataset
    data_path = "C:/Users/priya/OneDrive/Documents/my projects/mood based music generator/fer2013/train"
    emotions = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print("Emotions found:", emotions)

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    
    X, y = [], []

    # Collect Data
    for emotion in emotions:
        emotion_folder = os.path.join(data_path, emotion)
        print(f"Processing folder: {emotion_folder}")
        
        for img_file in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip invalid images
            
            # Convert to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(rgb_img)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                features = [coord for lm in landmarks for coord in (lm.x, lm.y)]
                
                X.append(features)
                y.append(emotion)
    
    X = np.array(X)
    y = np.array(y)
    print(f"Total samples collected: {len(X)}")

    # Handle PCA safely
    n_components = min(X.shape[0], 50)  # Prevents PCA errors if fewer samples exist
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {acc:.2f}")
    
    # Save Model + PCA
    dump(model, "emotion_model.pkl")
    dump(pca, "pca_transform.pkl")
    print("Model saved as 'emotion_model.pkl' and PCA as 'pca_transform.pkl'")

if __name__ == "__main__":
    main()
