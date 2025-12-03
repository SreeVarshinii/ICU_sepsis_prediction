import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

class SepsisDataset(Dataset):
    def __init__(self, parquet_file, max_len=200):
        self.data = pd.read_parquet(parquet_file)
        self.patient_ids = self.data['PatientID'].unique()
        self.max_len = max_len
        
        # Pre-group by patient for faster access
        self.grouped = self.data.groupby('PatientID')

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        group = self.grouped.get_group(pid)
        
        # Feature columns: Exclude IDs and Label
        # We assume columns ending in '_mask' are masks, others are features
        # Actually, let's just grab everything except metadata
        exclude_cols = ['PatientID', 'SepsisLabel', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
        feature_cols = [c for c in group.columns if c not in exclude_cols]
        
        features = group[feature_cols].values
        labels = group['SepsisLabel'].values
        
        # Time gaps: We can approximate by index difference if we assume hourly
        # But real data has 'ICULOS'. Let's use that if available.
        if 'ICULOS' in group.columns:
            iculos = group['ICULOS'].values
            # Time gap = current - previous
            # First step gap = 0 (or 1)
            time_gaps = np.zeros_like(iculos)
            time_gaps[1:] = iculos[1:] - iculos[:-1]
            time_gaps[0] = 0 # Start
        else:
            time_gaps = np.ones(len(features)) # Default 1 hour
            
        # Truncate or Pad
        seq_len = len(features)
        if seq_len > self.max_len:
            # Take last max_len
            features = features[-self.max_len:]
            labels = labels[-self.max_len:]
            time_gaps = time_gaps[-self.max_len:]
            seq_len = self.max_len
        
        # Convert to tensors
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        time_gaps = torch.tensor(time_gaps, dtype=torch.float32).unsqueeze(-1) # [Seq, 1]
        
        return features, labels, time_gaps, seq_len

def collate_fn(batch):
    features, labels, time_gaps, lengths = zip(*batch)
    
    max_len = max(lengths)
    feature_dim = features[0].shape[1]
    
    padded_features = torch.zeros(len(batch), max_len, feature_dim)
    padded_labels = torch.zeros(len(batch), max_len)
    padded_gaps = torch.zeros(len(batch), max_len, 1)
    mask = torch.zeros(len(batch), max_len) # 1 for valid
    
    for i, (f, l, g, length) in enumerate(zip(features, labels, time_gaps, lengths)):
        padded_features[i, :length, :] = f
        padded_labels[i, :length] = l
        padded_gaps[i, :length, :] = g
        mask[i, :length] = 1
        
    return padded_features, padded_labels, padded_gaps, mask

def get_loader(parquet_file, batch_size=32, shuffle=True, max_len=200, num_workers=0):
    dataset = SepsisDataset(parquet_file, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)
    return loader
