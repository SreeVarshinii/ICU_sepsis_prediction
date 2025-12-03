import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def process_and_split():
    data_path = "data/unified/unified_data.parquet"
    output_dir = "data/processed_splits"
    
    if not os.path.exists(data_path):
        print("Data file not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading data...")
    df = pd.read_parquet(data_path)
    
    # Splitting by PatientID
    print("Splitting data...")
    patient_ids = df['PatientID'].unique()
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    
    train_mask = df['PatientID'].isin(train_ids)
    val_mask = df['PatientID'].isin(val_ids)
    test_mask = df['PatientID'].isin(test_ids)
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Train: {len(train_df)} rows, {len(train_ids)} patients")
    print(f"Val: {len(val_df)} rows, {len(val_ids)} patients")
    print(f"Test: {len(test_df)} rows, {len(test_ids)} patients")
    
    # Standardization
    print("Computing statistics on Train...")
    feature_cols = [c for c in df.columns if c not in ['PatientID', 'SepsisLabel', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']]
    
    train_mean = train_df[feature_cols].mean()
    train_std = train_df[feature_cols].std()
    train_std = train_std.replace(0, 1.0)
    
    # Save stats
    stats = pd.DataFrame({'mean': train_mean, 'std': train_std})
    stats.to_csv(os.path.join(output_dir, 'stats.csv'))
    
    def preprocess_split(split_df, split_name):
        print(f"Processing {split_name}...")
        
        # 1. Create Mask
        mask = (~split_df[feature_cols].isnull()).astype(int)
        mask.columns = [f"{c}_mask" for c in feature_cols]
        
        # 2. Standardization
        scaled_features = (split_df[feature_cols] - train_mean) / train_std
        
        # 3. Fill NaNs with 0 (mean)
        scaled_features = scaled_features.fillna(0)
        
        # 4. Concatenate
        result = pd.concat([
            split_df[['PatientID', 'ICULOS', 'Unit1', 'Unit2', 'HospAdmTime']],
            scaled_features,
            mask,
            split_df[['SepsisLabel']]
        ], axis=1)
        
        # Save
        save_path = os.path.join(output_dir, f"{split_name}.parquet")
        result.to_parquet(save_path, index=False)
        print(f"Saved {save_path}")

    preprocess_split(train_df, "train")
    preprocess_split(val_df, "val")
    preprocess_split(test_df, "test")

if __name__ == "__main__":
    process_and_split()
