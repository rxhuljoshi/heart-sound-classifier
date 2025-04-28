import numpy as np
import librosa
import soundfile as sf

def load_audio(file_path, sr=22050):
    """
    Load and preprocess an audio file.
    
    Args:
        file_path (str): Path to the audio file
        sr (int): Target sampling rate
        
    Returns:
        numpy.ndarray: Preprocessed audio signal
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=sr)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        return audio
    except Exception as e:
        raise Exception(f"Error loading audio file: {str(e)}")

def extract_features(audio, sr=22050, n_mels=128, hop_length=512):
    """
    Extract mel spectrogram features from audio.
    
    Args:
        audio (numpy.ndarray): Audio signal
        sr (int): Sampling rate
        n_mels (int): Number of mel bands
        hop_length (int): Hop length for STFT
        
    Returns:
        numpy.ndarray: Flattened mel spectrogram features
    """
    try:
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            hop_length=hop_length
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Flatten the spectrogram
        features = mel_spec_db.flatten()
        
        return features
    except Exception as e:
        raise Exception(f"Error extracting features: {str(e)}")

def process_audio_file(file_path, target_sr=22050, duration=5):
    """
    Process an audio file and extract features.
    
    Args:
        file_path (str): Path to the audio file
        target_sr (int): Target sampling rate
        duration (int): Target duration in seconds
        
    Returns:
        numpy.ndarray: Extracted features
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=target_sr, duration=duration)
        
        # Ensure consistent length
        target_length = duration * target_sr
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Extract features
        # 1. MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # 2. Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spec_cent_mean = np.mean(spectral_centroids)
        spec_cent_std = np.std(spectral_centroids)
        
        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # 4. Root Mean Square Energy
        rms = librosa.feature.rms(y=audio)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Combine all features
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            [spec_cent_mean, spec_cent_std],
            [zcr_mean, zcr_std],
            [rms_mean, rms_std]
        ])
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def save_audio(audio_data, file_path, sr=22050):
    """
    Save audio data to file
    """
    sf.write(file_path, audio_data, sr)

def get_prediction_label(prediction):
    """
    Convert prediction to human-readable label
    """
    labels = [
        'Normal',
        'Noisy Normal',
        'Murmur',
        'Noisy Murmur',
        'Extrasystole'
    ]
    return labels[prediction]

def get_condition_description(label):
    """
    Get detailed description of the heart condition
    """
    descriptions = {
        'Normal': 'Regular heartbeat with clear lub-dub sounds.',
        'Noisy Normal': 'Normal heartbeat with some background noise, but no abnormalities.',
        'Murmur': 'Extra or unusual sounds caused by turbulent blood flow through the heart.',
        'Noisy Murmur': 'Heart murmur with additional background noise in the recording.',
        'Extrasystole': 'Extra or premature heartbeat that occurs outside the regular heart rhythm.'
    }
    return descriptions.get(label, 'Description not available.') 