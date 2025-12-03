import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Columns requested by user
REQUIRED_COLUMNS = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel'
]

def load_psv(file_path):
    return pd.read_csv(file_path, sep='|')

def process_set(data_dir, set_name):
    print(f"Processing {set_name} from {data_dir}...")
    files = [f for f in os.listdir(data_dir) if f.endswith('.psv')]
    
    dfs = []
    for f in tqdm(files, desc=f"Reading {set_name}"):
        file_path = os.path.join(data_dir, f)
        df = load_psv(file_path)
        
        # Filter columns
        # Note: Some files might miss columns? PhysioNet data is usually consistent in schema, but full of NaNs.
        # We select only the required ones.
        # Check if all required columns exist
        available_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
        df = df[available_cols]
        
        # Add Patient ID
        patient_id = f.split('.')[0]
        df.insert(0, 'PatientID', patient_id)
        
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def create_unified_table(raw_base_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sets = ['training_setA', 'training_setB']
    all_data = []
    
    for s in sets:
        set_dir = os.path.join(raw_base_dir, s)
        if os.path.exists(set_dir):
            df_set = process_set(set_dir, s)
            all_data.append(df_set)
        else:
            print(f"Warning: {set_dir} does not exist.")
            
    if not all_data:
        print("No data found.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    
    output_path = os.path.join(output_dir, 'unified_data.parquet')
    print(f"Saving unified table to {output_path}...")
    full_df.to_parquet(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    raw_base = "data/raw/training"
    unified_output = "data/unified"
    create_unified_table(raw_base, unified_output)
