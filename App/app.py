import streamlit as st
import pickle
import os
import sys
import soundfile as sf
from utils import extract_features

# Load model và scaler
scaler, model = pickle.load(open("SVM_model_with_scaling.pkl", "rb"))

label_map = ['neutral', 'calm', 'angry', 'happy', 'disgust', 'sad', 'fear', 'surprised']

st.title("🎙️ Emotion recognition from voice")

# --- Upload file .wav ---
st.subheader("📁 Uploads your .wav file:")

uploaded_file = st.file_uploader("Select .wav file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format="audio/wav")

    features = extract_features("temp.wav")
    if features is not None:
        features_scaled = scaler.transform([features])
        prediction_index = model.predict(features_scaled)[0]

        if 0 <= prediction_index < len(label_map):
            emotion = label_map[prediction_index]
            st.success(f"💬 Recognized emotion: **{emotion}**")
        else:
            st.error("⚠️ Unexpected prediction.")
    else:
        st.error("Cannot extract MFCCs features.")

    os.remove("temp.wav")

st.write("Python executable:", sys.executable)
