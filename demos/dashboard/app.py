import streamlit as st
import torch
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import time
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.dyt import DyTTransformer

# Page Config
st.set_page_config(page_title="Sepsis Prediction Suite", layout="wide")

st.title("🏥 ICU Sepsis Prediction Suite")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Patient Dashboard", "⚡ Real-time Monitor", "📈 Model Comparison"])

# Shared Functions
@st.cache_resource
def load_model(checkpoint_path, input_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DyTTransformer(input_dim=input_dim, d_model=64, n_heads=4, num_layers=2)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        return None, None

@st.cache_data
def load_data(path):
    return pd.read_parquet(path)

# ==========================================
# TAB 1: PATIENT DASHBOARD (Original app.py)
# ==========================================
with tab1:
    st.markdown("### Interactive Patient History Review")
    st.markdown("Retrospective analysis of patient stays with DyT risk trajectories.")
    
    col_conf, col_main = st.columns([1, 3])
    
    with col_conf:
        st.subheader("Configuration")
        # Hardcoded paths content for deployment simplicity
        model_path = "models/test_run/dyt_best.pth"
        data_path = "data/processed_splits/test.parquet"
        
        if os.path.exists(data_path):
            df = load_data(data_path)
            patient_ids = df['PatientID'].unique()
            selected_patient = st.selectbox("Select Patient ID", patient_ids)
        else:
            st.error("Data not found.")
            df = None

    if df is not None and selected_patient:
        patient_data = df[df['PatientID'] == selected_patient].copy()
        
        # Sort
        if 'ICULOS' in patient_data.columns:
            patient_data = patient_data.sort_values('ICULOS')
            time_axis = patient_data['ICULOS'].values
        else:
            time_axis = np.arange(len(patient_data))

        # Model Inference for Dashboard
        feature_cols = [c for c in df.columns if c not in ['PatientID', 'SepsisLabel', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']]
        input_dim = len(feature_cols)
        model, device = load_model(model_path, input_dim)

        if model:
            features = patient_data[feature_cols].values
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            # Simple gaps
            gaps = np.ones(len(features))
            t_gaps = torch.tensor(gaps, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            
            with torch.no_grad():
                logits, _ = model(x, t_gaps)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_axis, probs, label="Risk", color="red")
            ax.fill_between(time_axis, probs, color="red", alpha=0.1)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Sepsis Probability")
            ax.set_xlabel("Time (Hours)")
            ax.grid(True, alpha=0.3)
            
            if 'SepsisLabel' in patient_data.columns and patient_data['SepsisLabel'].sum() > 0:
                 onset = patient_data[patient_data['SepsisLabel'] == 1].iloc[0]
                 if 'ICULOS' in onset:
                     ax.axvline(onset['ICULOS'], color='black', linestyle='--', label='Onset')
            
            ax.legend()
            
            st.pyplot(fig)
            
            # Metrics
            curr = probs[-1]
            if curr > 0.5:
                st.error(f"Final Risk: {curr:.1%} (HIGH)")
            else:
                st.success(f"Final Risk: {curr:.1%} (STABLE)")

# ==========================================
# TAB 2: REAL-TIME MONITOR (Ported monitor.py)
# ==========================================
with tab2:
    st.markdown("### Real-time Bedside Monitor Simulation")
    st.markdown("Simulates a live data stream and visualizes alarms using Streamlit elements.")
    
    col_run, col_disp = st.columns([1, 4])
    
    with col_run:
        start_sim = st.button("▶️ Start Simulation")
        sim_speed = st.slider("Speed (sec/step)", 0.1, 2.0, 0.5)
        
    placeholder = st.empty()
    
    if start_sim:
        history_risk = []
        
        # Mock Data Generator (from monitor.py)
        def generate_mock_data(step):
            base_hr = 80 + (step * 0.5) + random.uniform(-2, 2)
            base_map = 90 - (step * 0.3) + random.uniform(-2, 2)
            base_o2 = 98 - (step * 0.1) + random.uniform(-1, 1)
            return {'HR': base_hr, 'MAP': base_map, 'O2Sat': base_o2, 'Resp': 16 + (step * 0.1)}

        for step in range(50):
            data = generate_mock_data(step)
            
            # Mock Risk
            risk = 1 / (1 + np.exp(-(step - 25) / 5)) # S curve centered at 25
            history_risk.append(risk)
            
            with placeholder.container():
                # Top: KPI Metrics
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Heart Rate", f"{data['HR']:.1f}", delta=f"{data['HR']-80:.1f}", delta_color="inverse")
                kpi2.metric("MAP", f"{data['MAP']:.1f}", delta=f"{data['MAP']-90:.1f}")
                kpi3.metric("O2 Saturation", f"{data['O2Sat']:.1f}")
                kpi4.metric("Risk Score", f"{risk:.1%}")
                
                # Alert Banner
                if risk > 0.5:
                    st.error("🚨 CRITICAL ALERT: SEPSIS RISK > 50%")
                elif risk > 0.2:
                    st.warning("⚠️ WARNING: ELEVATED RISK")
                else:
                    st.success("✅ PATIENT STABLE")
                
                # Chart
                fig_live, ax_live = plt.subplots(figsize=(10, 3))
                ax_live.plot(history_risk, color='red')
                ax_live.set_ylim(0, 1.05)
                ax_live.set_title("Real-time Risk Trend")
                ax_live.set_ylabel("Probability")
                st.pyplot(fig_live)
            
            time.sleep(sim_speed)

# ==========================================
# TAB 3: MODEL COMPARISON (Results)
# ==========================================
with tab3:
    st.markdown("### Model Performance Analysis")
    st.markdown("Comparative metrics between DyT (Proposed) and TFT (Baseline).")
    
    res_dir = "results"
    
    col_roc, col_prc = st.columns(2)
    
    with col_roc:
        st.subheader("ROC Curve")
        roc_path = os.path.join(res_dir, "roc_comparison.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="DyT vs TFT ROC Curves")
        else:
            st.warning("ROC Plot not found. Run comparison script first.")
            
    with col_prc:
        st.subheader("Precision-Recall Curve")
        prc_path = os.path.join(res_dir, "prc_comparison.png")
        if os.path.exists(prc_path):
            st.image(prc_path, caption="DyT vs TFT PRC Curves")
        else:
            st.warning("PRC Plot not found.")
            
    st.markdown("#### Quantitative Results")
    csv_path = os.path.join(res_dir, "dyt_results.csv")
    if os.path.exists(csv_path):
        st.dataframe(pd.read_csv(csv_path, nrows=10))
    else:
        st.info("No raw results CSV found.")
