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
    # Path to your dataset folder (containing subfolders like happy, sad, etc.)
    data_path = "fer2013/train"  # Adjust if needed
    emotions = os.listdir(data_path)
    print("Emotions found:", emotions)

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    
    X, y = [], []

    # 1. Collect Data
    for emotion in emotions:
        emotion_folder = os.path.join(data_path, emotion)
        # Skip files that aren't folders
        if not os.path.isdir(emotion_folder):
            continue
        
        print(f"Processing folder: {emotion_folder}")
        for img_file in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue  # skip broken images
            
            # Convert to RGB for MediaPipe
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(rgb_img)
            
            if results.multi_face_landmarks:
                # Take the first face only
                landmarks = results.multi_face_landmarks[0].landmark
                features = []
                # Flatten (x, y) for 468 landmarks => 936 features
                for lm in landmarks:
                    features.append(lm.x)
                    features.append(lm.y)
                
                X.append(features)
                y.append(emotion)
    
    X = np.array(X)
    y = np.array(y)
    print(f"Total samples collected: {len(X)}")

    # 2. Optional: Dimensionality Reduction (PCA)
    pca = PCA(n_components=50)  # You can tweak this number
    X_reduced = pca.fit_transform(X)
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )
    
    # 4. Train Model (Logistic Regression)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {acc:.2f}")

    # 6. Save Model + PCA
    dump(model, "emotion_model.pkl")
    dump(pca, "pca_transform.pkl")
    print("Saved model as 'emotion_model.pkl' and PCA as 'pca_transform.pkl'")

if __name__ == "__main__":
    main()
