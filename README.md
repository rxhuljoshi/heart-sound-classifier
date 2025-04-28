# Heart Sound Classification System

A machine learning-based system for classifying heart sounds into different categories using audio analysis and deep learning techniques.

## Features

- Audio file processing and feature extraction
- Machine learning model for heart sound classification
- Web interface for easy interaction
- Patient details management
- Comprehensive medical reports
- Visual analysis with waveforms and spectrograms

## Categories

The system classifies heart sounds into five categories:
1. Normal
2. Noisy Normal
3. Murmur
4. Noisy Murmur
5. Extrasystole

## Project Structure

```
heart-sound-classifier/
├── data/
│   ├── raw/              # Raw audio files
│   ├── models/           # Trained models
│   └── plots/            # Training plots and visualizations
├── app.py               # Streamlit web application
├── model.py             # Model operations
├── train.py             # Model training script
├── utils.py             # Utility functions
└── requirements.txt     # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-sound-classifier.git
cd heart-sound-classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Place your audio files in the appropriate class directories under `data/raw/`:
```
data/raw/
├── Normal/
├── Noisy_Normal/
├── Murmur/
├── Noisy_Murmur/
└── Extrasystole/
```

2. Run the training script:
```bash
python train.py
```

The script will:
- Process all audio files
- Train the model
- Generate evaluation metrics
- Save the trained model

### Running the Web Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Use the web interface to:
   - Enter patient details
   - Upload heart sound recordings
   - View analysis results
   - Generate medical reports

## Data Requirements

- Audio files should be in WAV or MP3 format
- Recommended sampling rate: 22050 Hz
- Files should be organized in class-specific directories

## Model Performance

The model's performance metrics are saved in:
- `data/plots/confusion_matrix.png`
- `data/plots/training_history.png`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset providers
- Contributors
- Medical professionals who provided guidance

## Contact

For questions and support, please open an issue in the GitHub repository. 