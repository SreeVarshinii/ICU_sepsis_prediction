import argparse
import torch
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np

from data.loader import get_loader
from models.dyt import DyTTransformer
from models.tft import TFTBaseline
from models.loss import FocalBCELoss

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading data...")
    train_loader = get_loader(args.train_path, batch_size=args.batch_size, shuffle=True, max_len=args.max_len)
    val_loader = get_loader(args.val_path, batch_size=args.batch_size, shuffle=False, max_len=args.max_len)
    
    # Determine input dim from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    print(f"Input Dimension: {input_dim}")
    
    # Forecasting Config
    TARGET_INDICES = [0, 1, 4, 6] # HR, O2Sat, MAP, Resp
    # Mask indices are offset by 10 (10 features)
    MASK_INDICES = [x + 10 for x in TARGET_INDICES]
    HORIZON = 3
    
    # Initialize Model
    if args.model == 'dyt':
        model = DyTTransformer(input_dim=input_dim, d_model=args.d_model, n_heads=args.n_heads, num_layers=args.num_layers).to(device)
    elif args.model == 'tft':
        model = TFTBaseline(input_dim=input_dim, d_model=args.d_model, n_heads=args.n_heads, num_layers=args.num_layers).to(device)
    else:
        raise ValueError("Invalid model type")
        
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion_cls = FocalBCELoss()
    criterion_reg = torch.nn.MSELoss(reduction='none') # Element-wise for masking
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for features, labels, time_gaps, mask in loop:
            features, labels = features.to(device), labels.to(device)
            time_gaps, mask = time_gaps.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            logits, forecast = model(features, time_gaps, mask)
            
            # Classification Loss (Masked)
            loss_cls = criterion_cls(logits, labels)
            loss_cls = (loss_cls * mask).sum() / (mask.sum() + 1e-8)
            
            # Forecasting Loss
            # Construct Targets
            target_list = []
            mask_list = []
            batch_size, seq_len, _ = features.shape
            
            for k in range(1, HORIZON + 1):
                # Shift features: Target at t is value at t+k
                if k < seq_len:
                    val_k = features[:, k:, TARGET_INDICES] # [B, L-k, 4]
                    pad_val = torch.zeros(batch_size, k, 4).to(device)
                    val_k = torch.cat([val_k, pad_val], dim=1)
                    
                    m_k = features[:, k:, MASK_INDICES] # [B, L-k, 4]
                    pad_m = torch.zeros(batch_size, k, 4).to(device)
                    m_k = torch.cat([m_k, pad_m], dim=1)
                else:
                    # Sequence too short for horizon
                    val_k = torch.zeros(batch_size, seq_len, 4).to(device)
                    m_k = torch.zeros(batch_size, seq_len, 4).to(device)

                target_list.append(val_k)
                
                # Combine with sequence mask (if current t is padding, don't train)
                seq_mask_expanded = mask.unsqueeze(-1) # [B, L, 1]
                m_k = m_k * seq_mask_expanded
                mask_list.append(m_k)
            
            target_forecast = torch.cat(target_list, dim=-1) # [B, L, 12]
            mask_forecast = torch.cat(mask_list, dim=-1) # [B, L, 12]
            
            loss_reg = criterion_reg(forecast, target_forecast)
            loss_reg = (loss_reg * mask_forecast).sum() / (mask_forecast.sum() + 1e-8)
            
            # Total Loss
            loss = loss_cls + 0.5 * loss_reg # Increase weight for forecasting
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels, time_gaps, mask in val_loader:
                features, labels = features.to(device), labels.to(device)
                time_gaps, mask = time_gaps.to(device), mask.to(device)
                
                logits, forecast = model(features, time_gaps, mask)
                
                loss_cls = criterion_cls(logits, labels)
                loss_cls = (loss_cls * mask).sum() / (mask.sum() + 1e-8)
                
                val_loss += loss_cls.item() # Monitor classification loss primarily
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/processed_splits/train.parquet')
    parser.add_argument('--val_path', type=str, default='data/processed_splits/val.parquet')
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--model', type=str, choices=['dyt', 'tft'], required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=200)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    train(args)
