import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import random

def add_noise(audio, noise_factor=0.005):
    """
    Add random noise to the audio signal.
    
    Args:
        audio (numpy.ndarray): Audio signal
        noise_factor (float): Factor to control noise level
        
    Returns:
        numpy.ndarray: Audio signal with added noise
    """
    noise = np.random.normal(0, noise_factor, len(audio))
    return audio + noise

def time_shift(audio, shift_max=0.1):
    """
    Shift the audio signal in time.
    
    Args:
        audio (numpy.ndarray): Audio signal
        shift_max (float): Maximum shift in seconds
        
    Returns:
        numpy.ndarray: Shifted audio signal
    """
    shift = int(len(audio) * shift_max)
    direction = np.random.randint(2)
    if direction == 1:
        shift = -shift
    augmented = np.roll(audio, shift)
    return augmented

def change_pitch(audio, sr, n_steps=2):
    """
    Change the pitch of the audio signal.
    
    Args:
        audio (numpy.ndarray): Audio signal
        sr (int): Sampling rate
        n_steps (int): Number of steps to shift pitch
        
    Returns:
        numpy.ndarray: Pitch-shifted audio signal
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def change_speed(audio, speed_factor=1.2):
    """
    Change the speed of the audio signal.
    
    Args:
        audio (numpy.ndarray): Audio signal
        speed_factor (float): Factor to change speed
        
    Returns:
        numpy.ndarray: Speed-changed audio signal
    """
    return librosa.effects.time_stretch(audio, rate=speed_factor)

def augment_audio(audio, sr):
    """
    Apply random augmentation to the audio signal.
    
    Args:
        audio (numpy.ndarray): Audio signal
        sr (int): Sampling rate
        
    Returns:
        list: List of augmented audio signals
    """
    augmented_signals = []
    
    # Original signal
    augmented_signals.append(audio)
    
    # Add noise
    noise_audio = add_noise(audio)
    augmented_signals.append(noise_audio)
    
    # Time shift
    shifted_audio = time_shift(audio)
    augmented_signals.append(shifted_audio)
    
    # Pitch shift
    pitch_shifted = change_pitch(audio, sr)
    augmented_signals.append(pitch_shifted)
    
    # Speed change
    speed_changed = change_speed(audio)
    augmented_signals.append(speed_changed)
    
    return augmented_signals

def process_directory(input_dir, output_dir, sr=22050):
    """
    Process all audio files in a directory and create augmented versions.
    
    Args:
        input_dir (str): Input directory containing audio files
        output_dir (str): Output directory for augmented files
        sr (int): Sampling rate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of audio files
    audio_files = [f for f in os.listdir(input_dir) 
                  if f.endswith(('.wav', '.mp3'))]
    
    print(f"\nProcessing files in {input_dir}")
    for file_name in tqdm(audio_files):
        # Load audio file
        input_path = os.path.join(input_dir, file_name)
        audio, sr = librosa.load(input_path, sr=sr)
        
        # Generate augmented versions
        augmented_signals = augment_audio(audio, sr)
        
        # Save augmented files
        base_name = os.path.splitext(file_name)[0]
        for i, aug_audio in enumerate(augmented_signals):
            output_path = os.path.join(output_dir, f"{base_name}_aug_{i}.wav")
            sf.write(output_path, aug_audio, sr)

def main():
    """
    Main function to augment the processed audio files.
    """
    print("\nStarting audio augmentation...")
    
    # Define source and target directories
    source_dirs = {
        'Normal': 'data/processed/Normal',
        'Noisy_Normal': 'data/processed/Noisy_Normal',
        'Murmur': 'data/processed/Murmur',
        'Noisy_Murmur': 'data/processed/Noisy_Murmur',
        'Extrasystole': 'data/processed/Extrasystole'
    }
    
    target_dir = 'data/augmented'
    
    # Create target directories if they don't exist
    for class_name in source_dirs.keys():
        os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)
    
    # Process each class
    for class_name, source_dir in source_dirs.items():
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory {source_dir} not found")
            continue
            
        print(f"\nProcessing {class_name}...")
        target_class_dir = os.path.join(target_dir, class_name)
        
        # Get list of audio files
        audio_files = [f for f in os.listdir(source_dir) 
                      if f.endswith('.wav')]
        
        # Process each file
        for file_name in tqdm(audio_files, desc=f"Processing {class_name}"):
            input_path = os.path.join(source_dir, file_name)
            base_name = os.path.splitext(file_name)[0]
            
            try:
                # Load audio file
                audio, sr = librosa.load(input_path, sr=22050)
                
                # Create augmented versions
                augmented_audios = augment_audio(audio, sr)
                
                # Save augmented versions
                for i, aug_audio in enumerate(augmented_audios):
                    output_path = os.path.join(target_class_dir, 
                                             f"{base_name}_aug{i}.wav")
                    sf.write(output_path, aug_audio, sr)
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
    
    print("\nAudio augmentation completed!")

if __name__ == "__main__":
    main() 