import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from model import load_model, predict
from utils import process_audio_file

def load_test_data(data_dir):
    """
    Load test data from the specified directory.
    
    Args:
        data_dir (str): Directory containing test audio files
        
    Returns:
        tuple: (features, labels, file_names)
    """
    features = []
    labels = []
    file_names = []
    
    classes = ['Normal', 'Noisy_Normal', 'Murmur', 'Noisy_Murmur', 'Extrasystole']
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found")
            continue
            
        print(f"Loading {class_name} files...")
        for file_name in os.listdir(class_dir):
            if file_name.endswith(('.wav', '.mp3')):
                file_path = os.path.join(class_dir, file_name)
                try:
                    # Process audio file
                    audio_features = process_audio_file(file_path)
                    features.append(audio_features)
                    labels.append(class_idx)
                    file_names.append(file_name)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    return np.array(features), np.array(labels), file_names

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(y_true, y_pred_proba, classes, save_path):
    """
    Plot and save ROC curves for each class.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        classes: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curves(y_true, y_pred_proba, classes, save_path):
    """
    Plot and save Precision-Recall curves for each class.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        classes: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_pred_proba[:, i])
        avg_precision = average_precision_score(y_true == i, y_pred_proba[:, i])
        plt.plot(recall, precision, 
                label=f'{class_name} (AP = {avg_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_evaluation_report(y_true, y_pred, y_pred_proba, classes, file_names):
    """
    Generate a detailed evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        classes: List of class names
        file_names: List of file names
    """
    # Create results directory
    os.makedirs('data/evaluation', exist_ok=True)
    
    # Save classification report
    report = classification_report(y_true, y_pred, target_names=classes)
    with open('data/evaluation/classification_report.txt', 'w') as f:
        f.write(report)
    
    # Create detailed results DataFrame
    results = []
    for i, (true_label, pred_label, probs, file_name) in enumerate(
        zip(y_true, y_pred, y_pred_proba, file_names)):
        results.append({
            'File': file_name,
            'True_Label': classes[true_label],
            'Predicted_Label': classes[pred_label],
            'Confidence': probs[pred_label],
            'Correct': true_label == pred_label
        })
    
    # Save detailed results
    df_results = pd.DataFrame(results)
    df_results.to_csv('data/evaluation/detailed_results.csv', index=False)
    
    # Generate plots
    plot_confusion_matrix(y_true, y_pred, classes, 
                         'data/evaluation/confusion_matrix.png')
    plot_roc_curves(y_true, y_pred_proba, classes, 
                    'data/evaluation/roc_curves.png')
    plot_precision_recall_curves(y_true, y_pred_proba, classes, 
                               'data/evaluation/precision_recall_curves.png')

def main():
    # Load model
    print("Loading model...")
    model = load_model('data/models/heart_sound_model.joblib')
    
    # Load test data
    print("\nLoading test data...")
    test_dir = 'data/test'  # You should have a separate test directory
    X_test, y_test, file_names = load_test_data(test_dir)
    
    if len(X_test) == 0:
        print("Error: No test data found")
        return
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Generate evaluation report
    print("\nGenerating evaluation report...")
    classes = ['Normal', 'Noisy_Normal', 'Murmur', 'Noisy_Murmur', 'Extrasystole']
    generate_evaluation_report(y_test, y_pred, y_pred_proba, classes, file_names)
    
    print("\nEvaluation completed! Results saved in data/evaluation/")

if __name__ == "__main__":
    main() 