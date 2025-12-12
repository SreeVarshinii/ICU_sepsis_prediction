# Early Sepsis Prediction with a Dynamic Tanh (DyT) Transformer

🔗 **Live Demo (Streamlit Dashboard):**  
👉 https://icu-sepsisprediction-monitoring.streamlit.app/

---

## Abstract
Early detection of sepsis in Intensive Care Units (ICUs) is challenging due to irregular sampling, missing data, and non-stationary physiological signals. We propose a **Dynamic Tanh (DyT) Transformer**, a novel Transformer-based architecture that replaces Layer Normalization with an adaptive, input-dependent normalization scheme and incorporates time-aware self-attention. Evaluated on the **PhysioNet 2019 Sepsis Challenge** dataset, the DyT Transformer substantially outperforms a standard Transformer baseline, achieving higher discrimination and precision under extreme class imbalance.

---

## Model Overview
We compare two architectures:

- **DyT Transformer (Proposed)**  
  Incorporates Dynamic Tanh normalization, temporal attention, and missingness-aware embeddings.
- **TFT Baseline**  
  Standard Transformer encoder using LayerNorm and conventional self-attention.

---

## Methodology

### Dynamic Tanh (DyT) Normalization
DyT replaces LayerNorm with an adaptive nonlinear normalization:
```math
y = \gamma \cdot \tanh\left(\alpha \cdot \frac{x - \mu}{\sigma}\right) + \beta
```
where $\alpha$ and $\beta$ are dynamically learned from input statistics.  
This allows the model to preserve clinically important spikes while remaining stable under distribution shifts.

### Temporal Attention
To model irregular time intervals, time-gap embeddings are injected into the attention mechanism:
```math
Q = W_q(x + \text{TimeEmb}(\Delta t)), \quad
K = W_k(x + \text{TimeEmb}(\Delta t))
```
This enables attention weights to depend explicitly on elapsed time between observations.

### Missingness-Aware Inputs
Observed values are concatenated with binary masks, allowing the model to distinguish between measured and imputed data.

### Loss Function
A **Focal Binary Cross-Entropy Loss** is used to address extreme class imbalance (~1.8% positive sepsis cases), emphasizing hard positive examples.

---

## Input & Output
- **Input:** ICU time-series of shape `[Batch, Sequence Length, Features]`, including vitals, labs, missingness masks, and time deltas.
- **Output:**  
  - Sepsis risk probability at each time step (classification)  
  - Auxiliary next-step vital sign forecasting (multi-task learning)

---

## Results

| Model | AUROC | AUPRC |
|------|------|------|
| **DyT Transformer (Proposed)** | **0.9024** | **0.1837** |
| TFT Baseline (LayerNorm) | 0.7001 | 0.0571 |

**Interpretation:** Dynamic normalization and temporal modeling substantially improve both discrimination and precision for early sepsis detection.

### Performance Plots
| ROC Curve | PR Curve |
| :---: | :---: |
| ![ROC](results/roc_comparison.png) | ![PRC](results/prc_comparison.png) |

---

## Implementation
- **Framework**: PyTorch  
- **Dataset**: PhysioNet 2019 Sepsis Challenge  
- **Evaluation**: Hospital-based cross-validation  
- **Visualization**: Streamlit-based real-time monitoring dashboard

---

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/SreeVarshinii/ICU_sepsis_prediction.git
    cd ICU_sepsis_prediction
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

### 1. Data Preparation
Download the PhysioNet 2019 data and place it in `data/raw`. Then run:
```bash
# Create unified parquet file
python src/data/preprocessing.py

# Generate train/val/test splits
python src/data/make_splits.py
```

### 2. Training
Train the proposed DyT model and the TFT baseline:
```bash
# Train DyT
python src/train.py --model dyt --epochs 20 --save_dir models/dyt_run

# Train TFT
python src/train.py --model tft --epochs 20 --save_dir models/tft_run
```

### 3. Evaluation
Evaluate models and generate plots:
```bash
python src/evaluate.py --model dyt --checkpoint models/dyt_run/dyt_best.pth --output_dir results
python src/evaluate.py --model tft --checkpoint models/tft_run/tft_best.pth --output_dir results

# Generate Comparison Plots
python src/compare.py --results_dir results
```

### 4. Live Demo & Dashboard
Launch the unified web interface containing the Patient Dashboard and Real-time Monitor locally:
```bash
streamlit run demos/dashboard/app.py
```

---

## 📂 Project Structure

```
Sepsis pred/
├── data/                   # Data storage (raw, processed, splits)
├── demos/                  # Visualization & Demo scripts
│   ├── dashboard/          # Streamlit App
│   └── monitor.py          # CLI Monitor Sim
├── models/                 # Saved model checkpoints
├── notebooks/              # Exploratory Data Analysis
├── reports/                # Documentation & Reports
├── results/                # Evaluation plots & CSVs
├── src/                    # Source Code
│   ├── data/               # Data loading & preprocessing
│   ├── models/             # PyTorch Model Architectures (DyT, TFT)
│   ├── train.py            # Training Loop
│   ├── evaluate.py         # Evaluation Script
│   └── compare.py          # Comparison Script
└── tests/                  # Unit Tests
```

---

## License
Open-source research code.  
Please cite the **PhysioNet 2019 Sepsis Challenge** dataset when using this repository.
