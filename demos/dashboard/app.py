import streamlit as st
import torch
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.dyt import DyTTransformer
from src.data.loader import get_loader

# Page Config
st.set_page_config(page_title="Sepsis Prediction Dashboard", layout="wide")

# Title
st.title("🏥 ICU Sepsis Prediction Dashboard")
st.markdown("Real-time risk monitoring using **DyT (Dynamic Transformer)**.")

# Sidebar
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model Checkpoint", "models/test_run/dyt_best.pth")
data_path = st.sidebar.text_input("Test Data Path", "data/processed_splits/test.parquet")

@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # We need to know input_dim. Let's cheat and load a sample first or hardcode if known.
    # But better to load data first.
    return None

@st.cache_data
def load_data(path):
    return pd.read_parquet(path)

# Load Data
if os.path.exists(data_path):
    df = load_data(data_path)
    patient_ids = df['PatientID'].unique()
    selected_patient = st.sidebar.selectbox("Select Patient ID", patient_ids)
else:
    st.error(f"Data not found at {data_path}")
    st.stop()

# Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Get input dim from data
feature_cols = [c for c in df.columns if c not in ['PatientID', 'SepsisLabel', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']]
input_dim = len(feature_cols)

@st.cache_resource
def get_model(checkpoint, input_dim):
    model = DyTTransformer(input_dim=input_dim, d_model=64, n_heads=4, num_layers=2)
    try:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = get_model(model_path, input_dim)

if model and selected_patient:
    # Prepare Patient Data
    patient_data = df[df['PatientID'] == selected_patient].copy()
    
    # Sort by time if possible, assuming ICULOS or just index
    if 'ICULOS' in patient_data.columns:
        patient_data = patient_data.sort_values('ICULOS')
        time_axis = patient_data['ICULOS'].values
        time_label = "ICU Length of Stay (Hours)"
    else:
        time_axis = np.arange(len(patient_data))
        time_label = "Time Steps"
        
    features = patient_data[feature_cols].values
    labels = patient_data['SepsisLabel'].values
    
    # Inference
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device) # [1, Seq, Dim]
    
    # Time gaps
    if 'ICULOS' in patient_data.columns:
        gaps = np.zeros(len(features))
        gaps[1:] = patient_data['ICULOS'].values[1:] - patient_data['ICULOS'].values[:-1]
    else:
        gaps = np.ones(len(features))
    
    t_gaps = torch.tensor(gaps, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    
    with torch.no_grad():
        logits, _ = model(x, t_gaps)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
    # Visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Sepsis Risk Trajectory")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot Risk
        ax.plot(time_axis, probs, label="Predicted Risk", color="red", linewidth=2)
        ax.fill_between(time_axis, probs, color="red", alpha=0.1)
        
        # Plot Label (if exists)
        if labels.sum() > 0:
            onset_idx = np.where(labels == 1)[0]
            if len(onset_idx) > 0:
                onset_time = time_axis[onset_idx[0]]
                ax.axvline(x=onset_time, color="black", linestyle="--", label="Sepsis Onset")
                
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Sepsis Probability")
        ax.set_xlabel(time_label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
    with col2:
        st.subheader("Current Status")
        current_risk = probs[-1]
        delta = probs[-1] - probs[-2] if len(probs) > 1 else 0
        
        st.metric("Current Risk Score", f"{current_risk:.2%}", f"{delta:.2%}")
        
        if current_risk > 0.5:
            st.error("⚠️ HIGH RISK ALERT")
        elif current_risk > 0.2:
            st.warning("⚠️ Elevated Risk")
        else:
            st.success("✅ Stable")
            
        st.markdown("### Key Vitals (Last Step)")
        # Try to find common vitals in columns
        vitals_map = {'HR': 'Heart Rate', 'O2Sat': 'O2 Saturation', 'SBP': 'Systolic BP', 'Temp': 'Temperature'}
        
        for col, name in vitals_map.items():
            # Check if column exists (case insensitive)
            match = [c for c in feature_cols if c.lower() == col.lower()]
            if match:
                val = patient_data[match[0]].iloc[-1]
                st.write(f"**{name}:** {val:.1f}")

    # Detailed Data View
    with st.expander("View Raw Data"):
        st.dataframe(patient_data)
