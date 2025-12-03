import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def analyze_data():
    data_path = "data/unified/unified_data.parquet"
    if not os.path.exists(data_path):
        print("Data file not found.")
        return

    print("Loading data...")
    df = pd.read_parquet(data_path)
    print(f"Dataset Shape: {df.shape}")

    # Unique Patients
    n_patients = df['PatientID'].nunique()
    print(f"Number of unique patients: {n_patients}")

    # Class Balance (SepsisLabel)
    sepsis_counts = df['SepsisLabel'].value_counts()
    print("\nClass Balance (Rows):")
    print(sepsis_counts)
    print(f"Sepsis Prevalence (Rows): {sepsis_counts.get(1, 0) / len(df):.2%}")

    # Patient-level Class Balance
    patient_labels = df.groupby('PatientID')['SepsisLabel'].max()
    print("\nClass Balance (Patients):")
    print(patient_labels.value_counts())
    print(f"Sepsis Prevalence (Patients): {patient_labels.sum() / n_patients:.2%}")

    # Missing Values
    print("\nMissing Values (Top 20):")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Missing %': missing_pct})
    missing_df = missing_df.sort_values(by='Missing %', ascending=False)
    print(missing_df.head(20))

    # Splitting
    print("\nCreating Splits...")
    patient_ids = df['PatientID'].unique()
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    print(f"Train Patients: {len(train_ids)}")
    print(f"Val Patients: {len(val_ids)}")
    print(f"Test Patients: {len(test_ids)}")
    
    # Standardization Stats (Train only)
    train_mask = df['PatientID'].isin(train_ids)
    train_df = df[train_mask]
    feature_cols = [c for c in df.columns if c not in ['PatientID', 'SepsisLabel', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']]
    
    print("\nFeature Statistics (Train):")
    print(train_df[feature_cols].describe().loc[['mean', 'std']])

if __name__ == "__main__":
    analyze_data()
