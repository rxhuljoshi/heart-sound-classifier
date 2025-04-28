import os
import shutil
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np

def create_directory_structure():
    """
    Create necessary directories for the project.
    """
    directories = [
        'data/raw/Normal',
        'data/raw/Noisy_Normal',
        'data/raw/Murmur',
        'data/raw/Noisy_Murmur',
        'data/raw/Extrasystole',
        'data/processed/Normal',
        'data/processed/Noisy_Normal',
        'data/processed/Murmur',
        'data/processed/Noisy_Murmur',
        'data/processed/Extrasystole',
        'data/test/Normal',
        'data/test/Noisy_Normal',
        'data/test/Murmur',
        'data/test/Noisy_Murmur',
        'data/test/Extrasystole'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            # Add .gitkeep to maintain directory structure in git
            gitkeep_file = os.path.join(directory, '.gitkeep')
            if not os.path.exists(gitkeep_file):
                with open(gitkeep_file, 'w') as f:
                    pass
        except Exception as e:
            print(f"Warning: Could not create directory {directory}: {str(e)}")
            continue

def process_audio_file(input_path, output_path, target_sr=22050, duration=5):
    """
    Process an audio file:
    - Resample to target sampling rate
    - Normalize audio
    - Trim to specified duration
    - Save as WAV format
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to save processed audio file
        target_sr (int): Target sampling rate
        duration (int): Target duration in seconds
    """
    try:
        # Load audio file
        audio, sr = librosa.load(input_path, sr=target_sr, duration=duration)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Save processed audio
        sf.write(output_path, audio, target_sr)
        
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def organize_dataset(source_dir, target_dir):
    """
    Organize the dataset by processing and moving files to appropriate directories.
    
    Args:
        source_dir (str): Source directory containing audio files
        target_dir (str): Target directory for processed files
    """
    # Create directory structure
    create_directory_structure()
    
    # Process each class directory
    classes = ['Normal', 'Noisy_Normal', 'Murmur', 'Noisy_Murmur', 'Extrasystole']
    
    for class_name in classes:
        source_class_dir = os.path.join(source_dir, class_name.lower())
        target_class_dir = os.path.join(target_dir, class_name)
        
        if not os.path.exists(source_class_dir):
            print(f"Warning: Source directory {source_class_dir} not found")
            continue
            
        print(f"\nProcessing {class_name} files...")
        
        # Get list of audio files
        audio_files = [f for f in os.listdir(source_class_dir) 
                      if f.endswith(('.wav', '.mp3'))]
        
        # Process each file
        for file_name in tqdm(audio_files, desc=f"Processing {class_name}"):
            input_path = os.path.join(source_class_dir, file_name)
            output_path = os.path.join(target_class_dir, 
                                     os.path.splitext(file_name)[0] + '.wav')
            
            # Process and save the file
            if process_audio_file(input_path, output_path):
                print(f"Processed: {file_name}")
            else:
                print(f"Failed to process: {file_name}")

def main():
    """
    Main function to organize and preprocess the dataset.
    """
    print("Starting dataset organization...")
    
    # Define source and target directories
    source_dirs = {
        'Normal': 'data/normal',
        'Noisy_Normal': 'data/noisynormal',
        'Murmur': 'data/murmur',
        'Noisy_Murmur': 'data/noisy_murmur',
        'Extrasystole': 'data/extrasystole'
    }
    
    target_dir = 'data/processed'
    
    # Create target directories if they don't exist
    for class_name in source_dirs.keys():
        os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)
    
    # Process each class
    for class_name, source_dir in source_dirs.items():
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory {source_dir} not found")
            continue
            
        print(f"\nProcessing {class_name} files...")
        target_class_dir = os.path.join(target_dir, class_name)
        
        # Get list of audio files
        audio_files = [f for f in os.listdir(source_dir) 
                      if f.endswith(('.wav', '.mp3'))]
        
        # Process each file
        for file_name in tqdm(audio_files, desc=f"Processing {class_name}"):
            input_path = os.path.join(source_dir, file_name)
            output_path = os.path.join(target_class_dir, 
                                     os.path.splitext(file_name)[0] + '.wav')
            
            try:
                # Process and save the file
                if process_audio_file(input_path, output_path):
                    print(f"Processed: {file_name}")
                else:
                    print(f"Failed to process: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
    
    print("\nDataset organization completed!")

if __name__ == "__main__":
    main() 