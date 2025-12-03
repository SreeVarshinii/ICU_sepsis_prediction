import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicTanh(nn.Module):
    """
    Dynamic Tanh (DyT) Normalization Layer.
    Adapts scaling based on input statistics, improving convergence for irregular time-series.
    """
    def __init__(self, num_features, eps=1e-5):
        super(DynamicTanh, self).__init__()
        self.num_features = num_features
        self.eps = eps
        
        # Learnable parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Dynamic scaling network
        # Predicts alpha (scale factor) from input statistics (mean, var)
        self.scale_net = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
            nn.Sigmoid() # Alpha in [0, 1]
        )

    def forward(self, x):
        # x: [Batch, Seq, Features]
        
        # Compute statistics over the sequence dimension (or local window)
        # Here we compute over the sequence for simplicity in this implementation
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        
        # Standardize
        x_norm = (x - mean) / std
        
        # Dynamic Tanh
        # Calculate dynamic scale factor alpha
        stats = torch.cat([mean, std], dim=-1) # [Batch, 1, Features*2]
        alpha = self.scale_net(stats) # [Batch, 1, Features]
        
        # Apply DyT: x_dyt = tanh(alpha * x_norm + beta) * gamma
        # Note: Original paper might vary slightly, but core idea is dynamic scaling inside tanh
        # We'll use a variation: x_out = gamma * tanh(alpha * x_norm) + beta
        
        out = self.gamma * torch.tanh(alpha * x_norm) + self.beta
        
        return out

class TemporalAttention(nn.Module):
    """
    Temporal Attention Mechanism handling variable time gaps.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Time encoding linear layer
        self.time_linear = nn.Linear(1, d_model)

    def forward(self, x, time_gaps, mask=None):
        # x: [Batch, Seq, d_model]
        # time_gaps: [Batch, Seq, 1] - Time since last measurement or absolute time diffs
        # mask: [Batch, Seq] - 1 for valid, 0 for padding
        
        batch_size, seq_len, _ = x.shape
        
        # Incorporate time info into Query and Key
        time_emb = self.time_linear(time_gaps) # [Batch, Seq, d_model]
        x_time = x + time_emb
        
        q = self.q_linear(x_time).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x_time).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # mask is [Batch, Seq] -> [Batch, 1, 1, Seq]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Causal Masking (Optional, but usually needed for forecasting)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        scores = scores.masked_fill(causal_mask, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_linear(out)
