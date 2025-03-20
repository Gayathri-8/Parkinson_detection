import numpy as np
import pandas as pd
import os
import librosa
import tensorflow as tf
import streamlit as st
import sounddevice as sd
import wave
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

model = load_model("parkinsons_rnn_lstm_model.h5")

def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

def record_audio(filename="recorded.wav", duration=5, samplerate=44100):
    st.info("🎙️ Recording... Speak now!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    st.success("✅ Recording saved!")

prediction_labels = {
    0: "No Parkinson’s Detected",
    1: "Mild Parkinson’s",
    2: "Moderate Parkinson’s",
    3: "Severe Parkinson’s"
}

def plot_confidence_scores(predictions_prob):
    num_classes = min(predictions_prob.shape[1], 4)  # Ensure a maximum of 4 classes (0 to 3)
    labels = [prediction_labels.get(i, f"Class {i}") for i in range(num_classes)]  # Match valid labels

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=predictions_prob[0][:num_classes], palette="Blues")
    plt.xticks(rotation=30)
    plt.ylabel("Confidence Score")
    plt.xlabel("Prediction Category")
    plt.title("Confidence Scores")
    st.pyplot(plt)

st.title("🧠 Parkinson's Disease Prediction")
st.sidebar.header("🎤 Input Options")

input_option = st.sidebar.radio("Select Input Type:", ["Upload CSV", "Upload Audio (WAV)", "Record Audio"])

if input_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        X = df.select_dtypes(include=[np.number]).drop(columns=['hoehn_yahr'], errors='ignore')
        X_scaled = StandardScaler().fit_transform(X)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

        predictions_prob = model.predict(X_reshaped)
        predictions = np.argmax(predictions_prob, axis=1)

        predictions_text = [prediction_labels.get(pred, "Unknown") for pred in predictions]
        st.write("### 🏥 Predictions:")
        st.write(predictions_text)

        st.write("### 📊 Prediction Probability Distribution")
        plot_confidence_scores(predictions_prob)

elif input_option == "Upload Audio (WAV)":
    uploaded_audio = st.file_uploader("Upload WAV file", type=["wav"])
    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav.write(uploaded_audio.read())
            temp_wav_path = temp_wav.name

        mfcc_features = extract_mfcc(temp_wav_path).reshape(1, -1, 1)
        predictions_prob = model.predict(mfcc_features)
        prediction = np.argmax(predictions_prob, axis=1)[0]

        st.write(f"### 🏥 Prediction: {prediction_labels.get(prediction, 'Unknown')}")

        st.write("### 📊 Confidence Scores")
        plot_confidence_scores(predictions_prob)

elif input_option == "Record Audio":
    if st.button("🎙️ Start Recording"):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        record_audio(temp_file, duration=5)

        mfcc_features = extract_mfcc(temp_file).reshape(1, -1, 1)
        predictions_prob = model.predict(mfcc_features)
        prediction = np.argmax(predictions_prob, axis=1)[0]

        st.write(f"### 🏥 Prediction: {prediction_labels.get(prediction, 'Unknown')}")

        st.write("### 📊 Confidence Scores")
        plot_confidence_scores(predictions_prob)