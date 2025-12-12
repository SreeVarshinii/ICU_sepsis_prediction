# 🚀 Sepsis Prediction Demos

This folder contains three demos to showcase the **DyT (Dynamic Transformer)** model for early sepsis prediction.

## 1. Interactive Dashboard (Streamlit)
A web-based dashboard for exploring patient cases and visualizing risk trajectories.

**Run:**
```bash
streamlit run demos/dashboard/app.py
```
*Note: Requires `streamlit` installed (`pip install streamlit`).*

## 2. Real-time ICU Monitor (CLI)
A terminal-based simulator that mimics a bedside monitor, streaming patient data in real-time.

**Run:**
```bash
python demos/cli/monitor.py
```
*Options:*
- `--patient [ID]`: Simulate a specific patient.
- `--delay [seconds]`: Adjust speed (default 0.5s).

## 3. Comparative Notebook
A Jupyter notebook to compare DyT predictions against the TFT baseline.

**Run:**
Open `demos/notebook/comparison.ipynb` in Jupyter Lab or VS Code.

---
**Prerequisites:**
- Ensure you are in the project root environment.
- Data must be at `data/processed_splits/test.parquet`.
- Models must be at `models/test_run/`.
