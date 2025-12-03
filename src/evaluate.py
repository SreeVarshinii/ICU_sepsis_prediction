import argparse
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import os
import pandas as pd

from data.loader import get_loader
from models.dyt import DyTTransformer
from models.tft import TFTBaseline

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    test_loader = get_loader(args.test_path, batch_size=args.batch_size, shuffle=False, max_len=args.max_len)
    sample_batch = next(iter(test_loader))
    input_dim = sample_batch[0].shape[-1]
    
    # Load Model
    if args.model == 'dyt':
        model = DyTTransformer(input_dim=input_dim, d_model=args.d_model, n_heads=args.n_heads, num_layers=args.num_layers).to(device)
    elif args.model == 'tft':
        model = TFTBaseline(input_dim=input_dim, d_model=args.d_model, n_heads=args.n_heads, num_layers=args.num_layers).to(device)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    all_targets = []
    all_probs = []
    all_masks = []
    
    print("Running Inference...")
    with torch.no_grad():
        for features, labels, time_gaps, mask in tqdm(test_loader):
            features = features.to(device)
            time_gaps = time_gaps.to(device)
            mask = mask.to(device)
            
            logits, _ = model(features, time_gaps, mask)
            probs = torch.sigmoid(logits)
            
            all_targets.append(labels.cpu().numpy().flatten())
            all_probs.append(probs.cpu().numpy().flatten())
            all_masks.append(mask.cpu().numpy().flatten())
            
    # Flatten
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    all_masks = np.concatenate(all_masks)
    
    # Filter masked values
    valid_indices = all_masks == 1
    y_true = all_targets[valid_indices]
    y_score = all_probs[valid_indices]
    
    # Metrics
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    
    print(f"Results for {args.model}:")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    
    # Save results
    results = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
    results.to_csv(os.path.join(args.output_dir, f"{args.model}_results.csv"), index=False)
    
    # Lead Time Analysis (Simplified)
    # We need patient IDs for this, but loader currently doesn't return them in batch.
    # For now, we report classification metrics.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='data/processed_splits/test.parquet')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['dyt', 'tft'], required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=200)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    evaluate(args)
