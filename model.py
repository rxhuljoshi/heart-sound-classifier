import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

def create_model():
    """
    Create a Random Forest model for audio classification
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    return model

def train_model(X, y):
    """
    Train the Random Forest model
    """
    model = create_model()
    model.fit(X, y)
    return model

def save_model(model, path):
    """
    Save the trained model
    """
    joblib.dump(model, path)

def load_model(model_path):
    """
    Load a trained scikit-learn model from disk.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        object: Loaded scikit-learn model
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def predict(model, features):
    """
    Make predictions using the loaded model.
    
    Args:
        model: Loaded scikit-learn model
        features (numpy.ndarray): Flattened mel spectrogram features
        
    Returns:
        tuple: (predicted_class, class_probabilities)
    """
    try:
        # Reshape features if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        # Get prediction and probabilities
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        return prediction, probabilities
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}") 