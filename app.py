import os
from flask import Flask, request, jsonify
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

model_path = '/Users/noshitha/Desktop/Github/Audio_analytics/rf_emotion_classifier.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

with open(model_path, 'rb') as file:
    model = pickle.load(file)

observed_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_features(file_path, mfcc=True, chroma=True, mel=True, pitch=True, energy=True):
    audio, sr = librosa.load(file_path, sr=None)
    features = []
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        features.extend(mfccs)
    if chroma:
        stft = np.abs(librosa.stft(audio))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        features.extend(chroma)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
        features.extend(mel)
    if pitch:
        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
        pitch = np.mean(pitches[pitches > 0])
        features.append(pitch)
    if energy:
        energy = np.mean(librosa.feature.rms(y=audio).T, axis=0)
        features.append(energy)
    return features

def pad_features(features, max_len=180):
    padded_features = np.zeros((1, max_len))
    if len(features) > max_len:
        padded_features[0, :max_len] = features[:max_len]
    else:
        padded_features[0, :len(features)] = features
    return padded_features

def predict_emotion(file_path):
    features = extract_features(file_path)
    padded_features = pad_features(features)
    prediction = model.predict(padded_features)
    label_encoder = LabelEncoder()
    label_encoder.fit(observed_emotions)
    emotion = label_encoder.inverse_transform(prediction)
    return emotion[0]

@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['file']
    file_path = 'temp.wav'
    audio_file.save(file_path)
    emotion = predict_emotion(file_path)
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
