import time
import random
import argparse
import pandas as pd
import torch
import numpy as np
import sys
import os
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich import box

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.dyt import DyTTransformer

# Mock Data Generator (if no file provided)
def generate_mock_data(step):
    # Simulate vitals for a septic patient
    # HR increases, MAP decreases
    base_hr = 80 + (step * 0.5) + random.uniform(-2, 2)
    base_map = 90 - (step * 0.3) + random.uniform(-2, 2)
    base_o2 = 98 - (step * 0.1) + random.uniform(-1, 1)
    
    return {
        'HR': base_hr,
        'MAP': base_map,
        'O2Sat': base_o2,
        'Resp': 16 + (step * 0.1),
        'SBP': base_map + 20,
        'DBP': base_map - 10,
        'Temp': 37.0 + (step * 0.05)
    }

def load_model(checkpoint_path, input_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DyTTransformer(input_dim=input_dim, d_model=64, n_heads=4, num_layers=2)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        return None, None

def make_layout():
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(
        Layout(name="vitals"),
        Layout(name="risk")
    )
    return layout

def run_monitor(args):
    console = Console()
    layout = make_layout()
    
    # Header
    layout["header"].update(
        Panel(Text("ICU REAL-TIME MONITORING SYSTEM - BED 04", justify="center", style="bold white on blue"), box=box.HEAVY)
    )
    
    # Load Model
    # We need input_dim. Let's assume 40 for now or load from data
    input_dim = 40 # Placeholder
    model, device = load_model(args.checkpoint, input_dim)
    
    if not model:
        console.print(f"[red]Error loading model from {args.checkpoint}. Running in Simulation Mode.[/red]")
    
    # Simulation Loop
    history = []
    risk_history = []
    
    with Live(layout, refresh_per_second=4) as live:
        for step in range(100):
            # 1. Get Data
            data = generate_mock_data(step)
            history.append(data)
            
            # 2. Inference (Mocked if model fails)
            if model:
                # Prepare tensor... (omitted for brevity in demo script, using mock risk)
                pass
            
            # Mock Risk Calculation
            # Sigmoid-like curve
            risk = 1 / (1 + np.exp(-(step - 50) / 10))
            risk_history.append(risk)
            
            # 3. Update Vitals Panel
            vitals_table = Table(title="Live Vitals", expand=True, box=box.SIMPLE)
            vitals_table.add_column("Parameter", style="cyan")
            vitals_table.add_column("Value", justify="right")
            vitals_table.add_column("Status", justify="center")
            
            for k, v in data.items():
                status = "[green]NORMAL[/green]"
                if k == 'HR' and v > 100: status = "[yellow]HIGH[/yellow]"
                if k == 'MAP' and v < 65: status = "[red]LOW[/red]"
                vitals_table.add_row(k, f"{v:.1f}", status)
                
            layout["vitals"].update(Panel(vitals_table, title="Patient Vitals", border_style="blue"))
            
            # 4. Update Risk Panel
            risk_color = "green"
            alert_msg = "STABLE"
            if risk > 0.2: 
                risk_color = "yellow"
                alert_msg = "WARNING: ELEVATED RISK"
            if risk > 0.5: 
                risk_color = "red"
                alert_msg = "CRITICAL: SEPSIS ALERT"
                
            risk_text = Text(f"\n\nSEPSIS RISK SCORE\n{risk:.1%}\n\n{alert_msg}", justify="center", style=f"bold {risk_color}")
            layout["risk"].update(Panel(risk_text, title="AI Prediction", border_style=risk_color))
            
            # Footer
            layout["footer"].update(Panel(Text(f"System Active | Time Step: {step} | Model: DyT-Transformer", style="dim")))
            
            time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='models/test_run/dyt_best.pth')
    args = parser.parse_args()
    run_monitor(args)
