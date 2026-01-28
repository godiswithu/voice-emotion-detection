import streamlit as st
import librosa
import numpy as np
import joblib

st.set_page_config(page_title="Voice Emotion Detection", layout="centered")

st.title(" Voice Emotion Detection")


model = joblib.load("model/emotion_model.pkl")
encoder = joblib.load("model/label_encoder.pkl")

def extract_features(audio):
    y, sr = librosa.load(audio, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc.reshape(1, -1)

uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("Predict Emotion"):
        features = extract_features(uploaded_file)
        prediction = model.predict(features)
        emotion = encoder.inverse_transform(prediction)

        st.success(f" Detected Emotion: **{emotion[0]}**")

