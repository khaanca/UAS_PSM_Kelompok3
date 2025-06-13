import streamlit as st
import os
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
from respiratory_pipeline import predict_respiratory_condition

# ---------- Custom Style ----------
st.set_page_config(page_title="Respiratory Sound Classifier", layout="centered")

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }
    .main-title {
        font-size: 2.5em;
        font-weight: 600;
        color: #1c1c1e;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .subtitle {
        font-size: 1.1em;
        color: #6e6e73;
        text-align: center;
        margin-bottom: 2em;
    }
    .upload-box {
        border: 2px dashed #d1d1d6;
        padding: 2em;
        border-radius: 20px;
        background: #f9f9f9;
        text-align: center;
        transition: 0.3s;
    }
    .upload-box:hover {
        background: #f1f1f1;
        border-color: #91caff;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown('<div class="main-title">Respiratory Sound Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a <strong>WAV</strong> file of respiratory audio to analyze and classify</div>', unsafe_allow_html=True)

# ---------- Upload Section ----------
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag & drop or browse your WAV file", type=["wav"])
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Model Check ----------
model_path = "respiratory_classifier.pkl"

if not os.path.exists(model_path):
    st.error(f"Model file `{model_path}` not found. Please ensure it exists.")
else:
    model = joblib.load(model_path)

    if uploaded_file:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("Processing the audio...")

        # ---------- Waveform Display ----------
        try:
            y, sr = librosa.load("temp_audio.wav", sr=None)
            fig, ax = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax, color="#007aff")
            ax.set_title("Waveform of Uploaded Audio", fontsize=14)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not display waveform: {e}")

        # ---------- Prediction ----------
        try:
            result = predict_respiratory_condition("temp_audio.wav", model_path)

            st.success("Prediction Complete!")
            st.markdown(f"""
                <div style='
                    padding: 1.5em;
                    background: #e6f0ff;
                    border-radius: 15px;
                    margin-top: 1em;
                '>
                    <h4 style='margin-bottom: 0.5em;'>Predicted Class: <span style="color:#007aff;">{result['prediction']}</span></h4>
                    <p style='font-size: 1.1em;'>Confidence Score: <strong>{result['probability']:.2f}</strong></p>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
