import torch
import torch.nn as nn
import math

class TFTBaseline(nn.Module):
    """
    Simplified Temporal Fusion Transformer (TFT) Baseline using standard LayerNorm.
    """
    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super(TFTBaseline, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 200, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, 1)
        self.forecaster = nn.Linear(d_model, input_dim)

    def forward(self, x, time_gaps=None, mask=None):
        # x: [Batch, Seq, Dim]
        # time_gaps: Ignored in baseline (or concatenated)
        
        seq_len = x.size(1)
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        
        # Transformer
        # mask argument in PyTorch TransformerEncoder is for padding mask (src_key_padding_mask)
        # or attn_mask for causal.
        # We'll use causal mask here.
        
        x = self.transformer(x, mask=causal_mask)
        
        logits = self.classifier(x).squeeze(-1)
        forecast = self.forecaster(x)
        
        return logits, forecast
