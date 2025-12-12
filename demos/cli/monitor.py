import time
import sys
import os
import pandas as pd
import numpy as np
import torch
import random
import argparse

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.dyt import DyTTransformer

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_data_and_model(data_path, model_path):
    print("Loading data and model...")
    df = pd.read_parquet(data_path)
    
    # Get input dim
    feature_cols = [c for c in df.columns if c not in ['PatientID', 'SepsisLabel', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']]
    input_dim = len(feature_cols)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DyTTransformer(input_dim=input_dim, d_model=64, n_heads=4, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return df, model, feature_cols, device

def simulate_patient(df, model, feature_cols, device, patient_id=None, delay=0.5):
    # Select a positive patient if none specified
    if patient_id is None:
        sepsis_patients = df[df['SepsisLabel'] == 1]['PatientID'].unique()
        if len(sepsis_patients) > 0:
            patient_id = random.choice(sepsis_patients)
        else:
            patient_id = df['PatientID'].unique()[0]
            
    print(f"Simulating Patient: {patient_id}")
    time.sleep(1)
    
    patient_data = df[df['PatientID'] == patient_id].copy()
    if 'ICULOS' in patient_data.columns:
        patient_data = patient_data.sort_values('ICULOS')
    
    features = patient_data[feature_cols].values
    labels = patient_data['SepsisLabel'].values
    
    # Pre-calculate predictions for the whole sequence for speed
    # In a real scenario, we'd do this step-by-step, but for demo smooth animation, batch is fine.
    # Actually, let's do step-by-step to be authentic to the "Online" nature.
    
    history_features = []
    history_gaps = []
    
    # Find vital columns for display
    vital_map = {}
    for v in ['HR', 'O2Sat', 'SBP', 'Resp']:
        match = [c for c in feature_cols if c.lower() == v.lower()]
        if match:
            vital_map[v] = match[0]
            
    print("Starting Monitoring...")
    time.sleep(1)
    
    for i in range(len(patient_data)):
        clear_screen()
        
        # Current Step Data
        current_feats = features[i]
        current_label = labels[i]
        
        # Prepare Input (History + Current)
        history_features.append(current_feats)
        
        # Gaps
        if i == 0:
            gap = 0.0
        else:
            if 'ICULOS' in patient_data.columns:
                gap = patient_data['ICULOS'].iloc[i] - patient_data['ICULOS'].iloc[i-1]
            else:
                gap = 1.0
        history_gaps.append(gap)
        
        # Tensorize
        # We need to limit history to max_len (e.g., 200)
        start_idx = max(0, len(history_features) - 200)
        seq_feats = np.array(history_features[start_idx:])
        seq_gaps = np.array(history_gaps[start_idx:])
        
        x = torch.tensor(seq_feats, dtype=torch.float32).unsqueeze(0).to(device)
        t = torch.tensor(seq_gaps, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        
        # Predict
        with torch.no_grad():
            logits, _ = model(x, t)
            prob = torch.sigmoid(logits[0, -1]).item()
            
        # --- DISPLAY UI ---
        print("==========================================")
        print(f"🏥 ICU MONITOR | Patient: {patient_id}")
        print("==========================================")
        print(f"Time Step: {i} | ICULOS: {patient_data['ICULOS'].iloc[i] if 'ICULOS' in patient_data.columns else i} hrs")
        print("------------------------------------------")
        
        # Vitals
        print("VITALS:")
        for v_name, v_col in vital_map.items():
            val = patient_data[v_col].iloc[i]
            print(f"  {v_name:<5}: {val:.1f}")
            
        print("------------------------------------------")
        
        # Risk Gauge
        bar_len = 20
        filled = int(prob * bar_len)
        bar = "█" * filled + "-" * (bar_len - filled)
        
        status = "NORMAL"
        color_code = "\033[92m" # Green
        if prob > 0.5:
            status = "CRITICAL ALERT!"
            color_code = "\033[91m" # Red
        elif prob > 0.2:
            status = "WARNING"
            color_code = "\033[93m" # Yellow
            
        print(f"SEPSIS RISK: [{bar}] {prob:.1%}")
        print(f"STATUS: {color_code}{status}\033[0m")
        
        if current_label == 1:
            print("\n🔴 CLINICAL ONSET (Ground Truth)")
            
        print("==========================================")
        print("Press Ctrl+C to stop")
        
        time.sleep(delay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed_splits/test.parquet')
    parser.add_argument('--model', default='models/test_run/dyt_best.pth')
    parser.add_argument('--patient', type=int, help='Specific patient ID')
    parser.add_argument('--delay', type=float, default=0.5, help='Speed of simulation')
    args = parser.parse_args()
    
    df, model, feats, device = load_data_and_model(args.data, args.model)
    try:
        simulate_patient(df, model, feats, device, args.patient, args.delay)
    except KeyboardInterrupt:
        print("\nMonitoring Stopped.")
