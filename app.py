import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
import librosa.display
from model import load_model, predict
from utils import process_audio_file, get_prediction_label, get_condition_description
from datetime import datetime
import pandas as pd

# Configure matplotlib style
plt.style.use('seaborn')

# Define custom colors for the application
CUSTOM_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Set matplotlib parameters
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#666666',
    'axes.grid': True,
    'grid.color': '#dddddd',
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'text.color': '#666666',
    'axes.labelcolor': '#666666',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
})

# Function to create a figure with light theme
def create_figure(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    return fig, ax

st.set_page_config(
    page_title="Heart Sound Classifier",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set Streamlit theme
st.markdown("""
    <style>
        .stApp {
            background-color: white;
            color: #262730;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #FF3333;
        }
        .stTextInput>div>div>input {
            background-color: white;
            color: #262730;
        }
        .stSelectbox>div>div>select {
            background-color: white;
            color: #262730;
        }
        .stTextArea>div>div>textarea {
            background-color: #262730;
            color: #FAFAFA;
            border: 1px solid #666666;
        }
        .stNumberInput>div>div>input {
            background-color: #262730;
            color: #FAFAFA;
            border: 1px solid #666666;
        }
        .stDateInput>div>div>input {
            background-color: #262730;
            color: #FAFAFA;
            border: 1px solid #666666;
        }
        .stExpander {
            background-color: #262730;
            border: 1px solid #666666;
        }
        .stMarkdown {
            color: #FAFAFA;
        }
        .stAlert {
            background-color: #262730;
            border: 1px solid #666666;
        }
        .stProgress > div > div > div {
            background-color: #FF4B4B;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #262730;
        }
        .stTabs [data-baseweb="tab"] {
            color: #FAFAFA;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FF4B4B;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

def save_results(patient_name, patient_id, age, gender, date, symptoms, prediction, confidence):
    """Save patient results to a CSV file."""
    # Create records directory if it doesn't exist
    os.makedirs('data/records', exist_ok=True)
    
    # Prepare the record
    record = {
        'patient_name': patient_name,
        'patient_id': patient_id,
        'age': age,
        'gender': gender,
        'date': date,
        'symptoms': symptoms,
        'prediction': prediction,
        'confidence': f"{confidence:.2%}",
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Convert to DataFrame
    df_record = pd.DataFrame([record])
    
    # Path to records file
    records_file = 'data/records/patient_records.csv'
    
    # Append to existing file or create new one
    if os.path.exists(records_file):
        df_record.to_csv(records_file, mode='a', header=False, index=False)
    else:
        df_record.to_csv(records_file, index=False)

def main():
    # Tabs for main sections
    tab0, tab1, tab2, tab3 = st.tabs(["Home", "Patient Details", "Results", "Saved Records"])
    
    # State to store patient details and results
    if 'patient_details_submitted' not in st.session_state:
        st.session_state['patient_details_submitted'] = False
    if 'patient_info' not in st.session_state:
        st.session_state['patient_info'] = {}
    if 'last_result' not in st.session_state:
        st.session_state['last_result'] = None
    
    with tab0:
        # Centered heart and content, no image
        st.markdown("""
        <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 60px;'>
            <div style='font-size: 80px; margin-bottom: 10px;'>❤️</div>
            <h1 style='margin-bottom: 0; text-align: center;'>Heart Sound Classification System</h1>
            <p style='font-size: 20px; color: #FAFAFA; text-align: center; max-width: 600px;'>
                An intelligent assistant for classifying heart sounds and supporting cardiac diagnosis.<br>
                Upload a heart sound, enter patient details, and get instant analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
    
    with tab1:
        st.header("Enter Patient Details and Upload Audio")
        with st.form("patient_details_form"):
            col1, col2 = st.columns(2)
            with col1:
                patient_name = st.text_input("Patient Name")
                age = st.number_input("Age", min_value=0, max_value=120, value=30)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            with col2:
                patient_id = st.text_input("Patient ID")
                date = st.date_input("Date")
                symptoms = st.text_area("Symptoms")
            with st.expander("Medical History (optional)"):
                medical_history = st.text_area("Previous Medical Conditions", placeholder="Enter any relevant medical history")
                medications = st.text_area("Current Medications", placeholder="Enter current medications")
            uploaded_file = st.file_uploader("Upload Heart Sound Recording", type=['wav', 'mp3'])
            submitted = st.form_submit_button("Submit Patient Details and Audio")
        if submitted:
            st.session_state['patient_details_submitted'] = True
            st.session_state['patient_info'] = {
                'patient_name': patient_name,
                'age': age,
                'gender': gender,
                'patient_id': patient_id,
                'date': date,
                'symptoms': symptoms,
                'medical_history': medical_history,
                'medications': medications
            }
            st.session_state['uploaded_file'] = uploaded_file
            st.success("Patient details and audio submitted! Please go to the Results tab to view analysis.")
    
    with tab2:
        st.header("Results")
        if not st.session_state.get('patient_details_submitted') or not st.session_state.get('uploaded_file'):
            st.info("Please fill and submit patient details and upload audio in the first tab.")
        else:
            uploaded_file = st.session_state['uploaded_file']
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                features = process_audio_file(temp_path)
                if features is not None:
                    st.subheader("Audio Waveform")
                    audio, sr = librosa.load(temp_path)
                    fig_wave, ax_wave = create_figure(figsize=(10, 3))
                    librosa.display.waveshow(audio, sr=sr, color=CUSTOM_COLORS[0])
                    plt.title("Waveform", color='#FAFAFA')
                    plt.tight_layout()
                    st.pyplot(fig_wave)
                    model = load_model('data/models/heart_sound_model.joblib')
                    prediction, probabilities = predict(model, features)
                    classes = ['Normal', 'Noisy Normal', 'Murmur', 'Noisy Murmur', 'Extrasystole']
                    st.session_state['last_result'] = {
                        'prediction': prediction,
                        'probabilities': probabilities.tolist(),
                        'classes': classes
                    }
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = create_figure(figsize=(10, 5))
                        bars = ax.bar(classes, probabilities, color=CUSTOM_COLORS)
                        for i, bar in enumerate(bars):
                            if i == prediction:
                                bar.set_color('#FF4B4B')
                            else:
                                bar.set_color('#666666')
                        plt.title("Prediction Probabilities", color='#FAFAFA')
                        plt.xticks(rotation=45, color='#FAFAFA')
                        plt.yticks(color='#FAFAFA')
                        plt.tight_layout()
                        st.pyplot(fig)
                    with col2:
                        risk_score = 0
                        if classes[prediction] in ['Murmur', 'Noisy Murmur', 'Extrasystole']:
                            risk_score = probabilities[prediction] * 100
                        elif classes[prediction] == 'Noisy Normal':
                            risk_score = probabilities[prediction] * 50
                        fig2, ax2 = create_figure(figsize=(4, 2.5))
                        ax2.set_aspect('auto')
                        theta_bg = np.linspace(np.pi, 0, 200)
                        x_bg = 0.5 + 0.45 * np.cos(theta_bg)
                        y_bg = 0.1 + 0.8 * np.sin(theta_bg)
                        ax2.plot(x_bg, y_bg, color='#666666', linewidth=8, solid_capstyle='round', zorder=1)
                        theta_fg = np.linspace(np.pi, np.pi - (risk_score/100)*np.pi, 200)
                        x_fg = 0.5 + 0.45 * np.cos(theta_fg)
                        y_fg = 0.1 + 0.8 * np.sin(theta_fg)
                        arc_color = '#FF4B4B' if risk_score > 50 else '#81C784'
                        ax2.plot(x_fg, y_fg, color=arc_color, linewidth=10, solid_capstyle='round', zorder=2)
                        angle = np.pi - (risk_score/100)*np.pi
                        x_needle = 0.5 + 0.42 * np.cos(angle)
                        y_needle = 0.1 + 0.75 * np.sin(angle)
                        ax2.plot([0.5, x_needle], [0.1, y_needle], color='#FAFAFA', linewidth=2, zorder=3)
                        ax2.scatter([0.5], [0.1], color='#FAFAFA', s=30, zorder=4)
                        ax2.text(0.5, -0.05, f'{risk_score:.1f}%', ha='center', va='center', fontsize=16, fontweight='bold', color=arc_color)
                        for pct in [0, 25, 50, 75, 100]:
                            tick_angle = np.pi - (pct/100)*np.pi
                            x_tick = 0.5 + 0.52 * np.cos(tick_angle)
                            y_tick = 0.1 + 0.88 * np.sin(tick_angle)
                            ax2.text(x_tick, y_tick, f'{pct}%', ha='center', va='center', fontsize=8, color='#FAFAFA')
                        ax2.axis('off')
                        plt.title("Heart Disease Risk Level", pad=20, color='#FAFAFA')
                        plt.tight_layout()
                        st.pyplot(fig2)
                    with st.expander("Condition Description"):
                        st.write(get_condition_description(classes[prediction]))
                    with st.expander("Save/Export Results"):
                        if st.button("Save Results"):
                            info = st.session_state['patient_info']
                            save_results(
                                info['patient_name'], info['patient_id'], info['age'], info['gender'], info['date'],
                                info['symptoms'], classes[prediction], probabilities[prediction])
                            st.success("Results saved successfully!")
            except Exception as e:
                st.error(f"Error processing audio file: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # Show last result if available and not just processed
        if st.session_state['last_result'] and not (st.session_state.get('patient_details_submitted') and st.session_state.get('uploaded_file')):
            res = st.session_state['last_result']
            prediction = res['prediction']
            probabilities = np.array(res['probabilities'])
            classes = res['classes']
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = create_figure(figsize=(10, 5))
                bars = ax.bar(classes, probabilities, color=CUSTOM_COLORS)
                for i, bar in enumerate(bars):
                    if i == prediction:
                        bar.set_color('#FF4B4B')
                    else:
                        bar.set_color('#666666')
                plt.title("Prediction Probabilities", color='#FAFAFA')
                plt.xticks(rotation=45, color='#FAFAFA')
                plt.yticks(color='#FAFAFA')
                plt.tight_layout()
                st.pyplot(fig)
            with col2:
                risk_score = 0
                if classes[prediction] in ['Murmur', 'Noisy Murmur', 'Extrasystole']:
                    risk_score = probabilities[prediction] * 100
                elif classes[prediction] == 'Noisy Normal':
                    risk_score = probabilities[prediction] * 50
                fig2, ax2 = create_figure(figsize=(4, 2.5))
                ax2.set_aspect('auto')
                theta_bg = np.linspace(np.pi, 0, 200)
                x_bg = 0.5 + 0.45 * np.cos(theta_bg)
                y_bg = 0.1 + 0.8 * np.sin(theta_bg)
                ax2.plot(x_bg, y_bg, color='#666666', linewidth=8, solid_capstyle='round', zorder=1)
                theta_fg = np.linspace(np.pi, np.pi - (risk_score/100)*np.pi, 200)
                x_fg = 0.5 + 0.45 * np.cos(theta_fg)
                y_fg = 0.1 + 0.8 * np.sin(theta_fg)
                arc_color = '#FF4B4B' if risk_score > 50 else '#81C784'
                ax2.plot(x_fg, y_fg, color=arc_color, linewidth=10, solid_capstyle='round', zorder=2)
                angle = np.pi - (risk_score/100)*np.pi
                x_needle = 0.5 + 0.42 * np.cos(angle)
                y_needle = 0.1 + 0.75 * np.sin(angle)
                ax2.plot([0.5, x_needle], [0.1, y_needle], color='#FAFAFA', linewidth=2, zorder=3)
                ax2.scatter([0.5], [0.1], color='#FAFAFA', s=30, zorder=4)
                ax2.text(0.5, -0.05, f'{risk_score:.1f}%', ha='center', va='center', fontsize=16, fontweight='bold', color=arc_color)
                for pct in [0, 25, 50, 75, 100]:
                    tick_angle = np.pi - (pct/100)*np.pi
                    x_tick = 0.5 + 0.52 * np.cos(tick_angle)
                    y_tick = 0.1 + 0.88 * np.sin(tick_angle)
                    ax2.text(x_tick, y_tick, f'{pct}%', ha='center', va='center', fontsize=8, color='#FAFAFA')
                ax2.axis('off')
                plt.title("Heart Disease Risk Level", pad=20, color='#FAFAFA')
                plt.tight_layout()
                st.pyplot(fig2)
            with st.expander("Condition Description"):
                st.write(get_condition_description(classes[prediction]))
            with st.expander("Save/Export Results"):
                if st.button("Save Results"):
                    info = st.session_state['patient_info']
                    save_results(
                        info['patient_name'], info['patient_id'], info['age'], info['gender'], info['date'],
                        info['symptoms'], classes[prediction], probabilities[prediction])
                    st.success("Results saved successfully!")
    
    with tab3:
        st.header("Saved Patient Records")
        os.makedirs('data/records', exist_ok=True)
        records_file = 'data/records/patient_records.csv'
        # Add confirmation before clearing all records
        if os.path.exists(records_file):
            confirm_clear = st.checkbox("I want to clear all saved patient records.", key="confirm_clear_records")
            if confirm_clear and st.button("Clear All Records", key="clear_records"):
                os.remove(records_file)
                st.success("All saved patient records have been cleared.")
        if os.path.exists(records_file):
            df = pd.read_csv(records_file)
            search_term = st.text_input("Search records by patient name or ID")
            if search_term:
                df = df[df['patient_name'].str.contains(search_term, case=False) | 
                       df['patient_id'].str.contains(search_term, case=False)]
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Records",
                    data=csv,
                    file_name="patient_records.csv",
                    mime="text/csv"
                )
            else:
                st.info("No records found matching your search.")
        else:
            st.info("No saved records found.")

    # Single footer at the bottom
    st.markdown("""
    <div style='text-align: center; color: #FAFAFA; margin-top: 40px;'>
        <em>Created with ❤️ for heart sound classification.<br>Using machine learning to help diagnose heart conditions.</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 