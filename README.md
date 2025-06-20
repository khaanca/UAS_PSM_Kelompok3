# UAS PSM Kelompok 6: Sound-based Respiratory Disease Detection using Random Forest

This is a Streamlit-based web application for classifying respiratory conditions from uploaded **WAV** audio recordings. It uses a pre-trained Scikit-learn model and a custom audio preprocessing pipeline.

## How to Run

1. Clone the repository

    ```bash
    git clone https://github.com/khaanca/UAS_PSM_Kelompok6.git
    cd UAS_PSM_Kelompok6
    ```

2. (Optional) Create virtual environment

    ```bash
    python -m venv venv
    source venv/bin/activate        # On Windows: venv\Scripts\activate
    ```

3. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

    Or manually:

    ```bash
    pip install streamlit librosa scikit-learn matplotlib pandas joblib
    ```

4. Run the app

    ```bash
    streamlit run app.py
    ```

## File Descriptions

- `app.py` – Streamlit interface for uploading and classifying respiratory sounds.
- `respiratory_pipeline.py` – Pipeline for audio preprocessing and feature extraction.
- `respiratory_classifier.pkl` – Trained classifier model used for prediction.

## Notes

- Only `.wav` files are supported.
- Audio is automatically trimmed and features extracted using `librosa`.
- Prediction results include the most likely class and confidence score.

## Example

After launching the app, you’ll see a web page where you can upload a respiratory audio `.wav` file. The system will display a waveform and predict the condition with a confidence score.
