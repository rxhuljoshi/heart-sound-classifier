import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import create_model, train_model, save_model
from utils import process_audio_file
from tqdm import tqdm
import joblib
from sklearn.ensemble import RandomForestClassifier

def prepare_dataset(base_dir):
    """
    Prepare dataset by loading and processing audio files.
    
    Args:
        base_dir (str): Base directory containing class subdirectories
        
    Returns:
        tuple: (features, labels)
    """
    features = []
    labels = []
    
    classes = ['Normal', 'Noisy_Normal', 'Murmur', 'Noisy_Murmur', 'Extrasystole']
    
    print("Preparing dataset...")
    for class_idx, class_name in enumerate(classes):
        # Process both processed and augmented files
        for data_dir in ['data/processed', 'data/augmented']:
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
                
            print(f"Processing {class_name} files from {data_dir}...")
            
            # Get list of audio files
            audio_files = [f for f in os.listdir(class_dir) 
                          if f.endswith('.wav')]
            
            # Process each file
            for file_name in tqdm(audio_files, desc=f"Processing {class_name}"):
                file_path = os.path.join(class_dir, file_name)
                try:
                    # Extract features
                    audio_features = process_audio_file(file_path)
                    features.append(audio_features)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    return np.array(features), np.array(labels)

def evaluate_model(model, X_test, y_test, classes):
    """
    Evaluate the model and generate performance metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        classes: List of class names
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save confusion matrix plot
    os.makedirs('data/plots', exist_ok=True)
    plt.savefig('data/plots/confusion_matrix.png')
    plt.close()

def plot_training_history(history):
    """
    Plot training history metrics.
    
    Args:
        history: Model training history
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data/plots/training_history.png')
    plt.close()

def main():
    """
    Main function to train the model.
    """
    print("Starting model training...")
    
    # Prepare dataset
    X, y = prepare_dataset('data')
    
    if len(X) == 0:
        print("Error: No data found")
        return
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    # Print classification report
    classes = ['Normal', 'Noisy_Normal', 'Murmur', 'Noisy_Murmur', 'Extrasystole']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Save model
    os.makedirs('data/models', exist_ok=True)
    model_path = 'data/models/heart_sound_model.joblib'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Create and save confusion matrix plot
    os.makedirs('data/plots', exist_ok=True)
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('data/plots/confusion_matrix.png')
    plt.close()
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 