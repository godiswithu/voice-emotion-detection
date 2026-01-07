import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATA_PATH = "data/audio_speech_actors_01-24"
MODEL_PATH = "model"

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

X = []
y = []

print("🔄 Extracting features...")

for actor in os.listdir(DATA_PATH):
    actor_path = os.path.join(DATA_PATH, actor)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]

            emotion_map = {
                "01": "neutral",
                "02": "calm",
                "03": "happy",
                "04": "sad",
                "05": "angry",
                "06": "fearful",
                "07": "disgust",
                "08": "surprised"
            }

            emotion = emotion_map.get(emotion_code)
            if emotion is None:
                continue

            file_path = os.path.join(actor_path, file)
            features = extract_features(file_path)

            X.append(features)
            y.append(emotion)

X = np.array(X)
y = np.array(y)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

os.makedirs(MODEL_PATH, exist_ok=True)
joblib.dump(model, "model/emotion_model.pkl")
joblib.dump(encoder, "model/label_encoder.pkl")

print("💾 Model saved successfully")
