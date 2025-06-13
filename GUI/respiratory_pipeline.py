import os
import numpy as np
import pandas as pd
import joblib
import librosa
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Custom transformer to load audio files
class AudioLoader(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = {}
        for file_path in X:
            filename = file_path.split('\\')[-1] if '\\' in file_path else file_path.split('/')[-1]
            y, sr = librosa.load(file_path, mono=True)
            result[filename] = {'data': y, 'sample_rate': sr}
        return result

# Custom transformer to trim audio to consistent duration
class AudioTrimmer(BaseEstimator, TransformerMixin):
    def __init__(self, target_duration=7.8560090702947845):
        self.target_duration = target_duration

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        trimmed = {}
        for filename, audio_info in X.items():
            target_samples = int(self.target_duration * audio_info['sample_rate'])
            if len(audio_info['data']) < target_samples:
                trimmed_data = np.pad(audio_info['data'], (0, target_samples - len(audio_info['data'])), 'constant')
            else:
                trimmed_data = audio_info['data'][:target_samples]

            trimmed[filename] = {
                'data': trimmed_data,
                'sample_rate': audio_info['sample_rate'],
                'duration': self.target_duration
            }
        return trimmed

# Custom transformer to extract features
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = {}
        for filename, audio_info in X.items():
            y_audio = audio_info['data']
            sr = audio_info['sample_rate']

            features[filename] = {
                'chroma_stft': librosa.feature.chroma_stft(y=y_audio, sr=sr),
                'mfcc': librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13),
                'mel_spectrogram': librosa.feature.melspectrogram(y=y_audio, sr=sr),
                'spectral_contrast': librosa.feature.spectral_contrast(y=y_audio, sr=sr),
                'spectral_centroid': librosa.feature.spectral_centroid(y=y_audio, sr=sr),
                'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=y_audio, sr=sr),
                'spectral_rolloff': librosa.feature.spectral_rolloff(y=y_audio, sr=sr),
                'zero_crossing_rate': librosa.feature.zero_crossing_rate(y=y_audio),
            }

        return features

# Custom transformer to compute statistics
class FeatureStatisticsCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, excluded_features=None):
        self.excluded_features = excluded_features or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feature_stats = []
        for filename, features in X.items():
            file_stats = {'filename': filename}
            for feature_name, feature_data in features.items():
                file_stats[f'{feature_name}_mean'] = np.mean(feature_data)
                file_stats[f'{feature_name}_std'] = np.std(feature_data)
                file_stats[f'{feature_name}_max'] = np.max(feature_data)
                file_stats[f'{feature_name}_min'] = np.min(feature_data)
            feature_stats.append(file_stats)

        df = pd.DataFrame(feature_stats)

        for feature in self.excluded_features:
            if feature in df.columns:
                df = df.drop(feature, axis=1)

        return df.select_dtypes(exclude=['object'])

# Create preprocessing pipeline
def create_respiratory_pipeline():
    excluded_features = ['mel_spectrogram_min', 'chroma_stft_max']
    preprocessing_pipeline = Pipeline([
        ('load_audio', AudioLoader()),
        ('trim_audio', AudioTrimmer()),
        ('extract_features', FeatureExtractor()),
        ('calculate_statistics', FeatureStatisticsCalculator(excluded_features=excluded_features))
    ])
    return preprocessing_pipeline

# Predict function
def predict_respiratory_condition(wav_file_path, model_path='respiratory_classifier.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    model = joblib.load(model_path)
    pipeline = create_respiratory_pipeline()
    features_df = pipeline.transform([wav_file_path])
    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    max_prob = np.max(probabilities)
    return {
        'prediction': prediction,
        'probability': max_prob,
        'all_probabilities': dict(zip(model.classes_, probabilities))
    }
