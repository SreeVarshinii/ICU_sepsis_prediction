import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from models.dyt import DyTTransformer
from models.tft import TFTBaseline
from models.loss import FocalBCELoss

def test_models():
    batch_size = 4
    seq_len = 50
    input_dim = 30 # Features + Masks
    
    x = torch.randn(batch_size, seq_len, input_dim)
    time_gaps = torch.abs(torch.randn(batch_size, seq_len, 1)) # Positive time gaps
    mask = torch.ones(batch_size, seq_len) # All valid
    
    print("Testing DyT Transformer...")
    model_dyt = DyTTransformer(input_dim=input_dim)
    logits, forecast = model_dyt(x, time_gaps, mask)
    print(f"DyT Output Shapes: Logits {logits.shape}, Forecast {forecast.shape}")
    assert logits.shape == (batch_size, seq_len)
    assert forecast.shape == (batch_size, seq_len, input_dim)
    
    print("Testing TFT Baseline...")
    model_tft = TFTBaseline(input_dim=input_dim)
    logits, forecast = model_tft(x) # TFT ignores time_gaps in this baseline
    print(f"TFT Output Shapes: Logits {logits.shape}, Forecast {forecast.shape}")
    assert logits.shape == (batch_size, seq_len)
    assert forecast.shape == (batch_size, seq_len, input_dim)
    
    print("Testing Focal Loss...")
    criterion = FocalBCELoss()
    targets = torch.randint(0, 2, (batch_size, seq_len)).float()
    loss = criterion(logits, targets)
    print(f"Focal Loss: {loss.item()}")
    assert not torch.isnan(loss)

if __name__ == "__main__":
    test_models()
