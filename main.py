import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import librosa
import joblib
import io
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model

# Force TensorFlow to use CPU (Fixes GPU errors)


app = FastAPI()

# Load the trained model and label encoder
MODEL_PATH = "speech_emotion_recognition_model_optimized (1).h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

try:
    model = load_model(MODEL_PATH)
    encoder = joblib.load(LABEL_ENCODER_PATH)
    model_status = "Model loaded successfully"
except Exception as e:
    model_status = f"Model loading failed: {str(e)}"

# Feature extraction function
def extract_features(audio_data, sr):
    features = np.array([])

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    features = np.hstack((features, mfccs))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    chroma = np.mean(chroma.T, axis=0)
    features = np.hstack((features, chroma))

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    mel = np.mean(mel.T, axis=0)
    features = np.hstack((features, mel))

    return features.reshape(1, -1, 1)

# ✅ Health Check Route
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_status": model_status}

# 🎤 Speech Emotion Prediction Route
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file as bytes
    audio_bytes = await file.read()

    # Convert bytes to an audio file for librosa
    audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # Extract features
    features = extract_features(audio_data, sr)

    # Make prediction
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_emotion = encoder.inverse_transform([predicted_index])[0]

    return {"predicted_emotion": predicted_emotion}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)