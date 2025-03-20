import numpy as np
import pandas as pd
import os
import librosa
import tensorflow as tf
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

extract_zip("pva_wav_train.zip", "pva_wav_train")
extract_zip("pva_wav_test.zip", "pva_wav_test")

def load_and_preprocess_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[np.number])
    df.fillna(0, inplace=True)

    y = df['hoehn_yahr'].values if 'hoehn_yahr' in df.columns else np.zeros(len(df))
    unique_classes = np.unique(y)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_classes)}
    y = np.array([label_mapping[label] for label in y])

    X = df.drop(columns=['hoehn_yahr'], errors='ignore').values
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y

csv_train_path = "plmpva_train.csv"
csv_test_path = "plmpva_test-WithPDRS.csv"
X_train_csv, y_train = load_and_preprocess_csv(csv_train_path)
X_test_csv, y_test = load_and_preprocess_csv(csv_test_path)

num_classes = len(np.unique(y_train))

def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

def load_audio_features(audio_folder, num_samples):
    audio_files = os.listdir(audio_folder)[:num_samples]
    return np.array([extract_mfcc(os.path.join(audio_folder, file)) for file in audio_files])

audio_train_path = "pva_wav_train/wav/"
audio_test_path = "pva_wav_test/wav/"

X_train_audio = load_audio_features(audio_train_path, len(X_train_csv))
X_test_audio = load_audio_features(audio_test_path, len(X_test_csv))

X_train = np.hstack((X_train_csv, X_train_audio))
X_test = np.hstack((X_test_csv, X_test_audio))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

def build_rnn_lstm_model(input_shape, num_classes):
    model = Sequential([
        SimpleRNN(64, return_sequences=True, input_shape=input_shape),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

rnn_lstm_model = build_rnn_lstm_model((X_train.shape[1], 1), num_classes)
rnn_lstm_model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_rnn, y_test))
rnn_lstm_model.save("parkinsons_rnn_lstm_model.h5")

y_pred_rnn_lstm = np.argmax(rnn_lstm_model.predict(X_test_rnn), axis=1)

final_predictions = np.round((y_pred_rf + y_pred_rnn_lstm) / 2).astype(int)

accuracy = accuracy_score(y_test, final_predictions)
precision = precision_score(y_test, final_predictions, average='macro')
recall = recall_score(y_test, final_predictions, average='macro')
f1 = f1_score(y_test, final_predictions, average='macro')

print(f"Hybrid Model Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")