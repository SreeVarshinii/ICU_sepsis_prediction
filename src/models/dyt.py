import torch
import torch.nn as nn
from .layers import DynamicTanh, TemporalAttention

class DyTTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super(DyTTransformer, self).__init__()
        
        # Input Embedding
        # We assume input_dim includes features + masks. 
        # We might want to separate them or just project them all.
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional Encoding (Standard or Learnable)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 200, d_model)) # Max len 200
        
        # DyT Layers (Pre-Norm style or Post-Norm? Paper usually suggests replacing LayerNorm)
        # We'll implement a custom TransformerEncoderLayer that uses DyT
        self.layers = nn.ModuleList([
            DyTTransformerLayer(d_model, n_heads, dropout) for _ in range(num_layers)
        ])
        
        # Heads
        self.classifier = nn.Linear(d_model, 1) # Sepsis Risk
        self.forecaster = nn.Linear(d_model, input_dim) # Predict next step (simplified)

    def forward(self, x, time_gaps, mask=None):
        # x: [Batch, Seq, Dim]
        # time_gaps: [Batch, Seq, 1]
        
        seq_len = x.size(1)
        
        # Embedding
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer Layers
        for layer in self.layers:
            x = layer(x, time_gaps, mask)
            
        # Output (Take last step for classification? Or all steps?)
        # For Sepsis, we classify at each time step.
        logits = self.classifier(x).squeeze(-1) # [Batch, Seq]
        forecast = self.forecaster(x) # [Batch, Seq, Dim]
        
        return logits, forecast

class DyTTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(DyTTransformerLayer, self).__init__()
        self.attn = TemporalAttention(d_model, n_heads, dropout)
        self.dyt1 = DynamicTanh(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dyt2 = DynamicTanh(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, time_gaps, mask=None):
        # Sublayer 1: Attention
        # Norm first (Pre-Norm)
        x_norm = self.dyt1(x)
        attn_out = self.attn(x_norm, time_gaps, mask)
        x = x + self.dropout1(attn_out)
        
        # Sublayer 2: FFN
        x_norm = self.dyt2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout2(ffn_out)
        
        return x
