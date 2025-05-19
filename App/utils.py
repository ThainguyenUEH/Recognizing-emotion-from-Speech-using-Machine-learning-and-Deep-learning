import librosa
import numpy as np

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        mfcc_mean = np.mean(mfcc, axis=1)

        return mfcc_mean  
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None
