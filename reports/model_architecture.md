# Model Architecture Report

## 1. Overview
This project implements two Transformer-based architectures for Sepsis prediction:
1.  **DyT-based Transformer (Proposed)**: Incorporates Dynamic Tanh normalization and Temporal Attention.
2.  **TFT Baseline**: A standard Transformer using LayerNorm.

## 2. DyT-based Transformer
The DyT Transformer is designed to handle the irregularities and non-stationarity of ICU time-series data.

### Key Components
-   **Dynamic Tanh (DyT) Layer**:
    -   Replaces standard LayerNorm.
    -   Computes dynamic scaling factors ($\alpha, \beta$) based on input statistics (mean, variance).
    -   Formula: $y = \gamma \cdot \tanh(\alpha \cdot \frac{x - \mu}{\sigma}) + \beta$
    -   **Benefit**: Adapts to rapid shifts in vital signs distribution, common in sepsis onset.

-   **Temporal Attention**:
    -   Modifies the standard Self-Attention mechanism.
    -   Injects time-gap information into the Query and Key projections.
    -   $Q = W_q(x + \text{TimeEmb}(\Delta t))$, $K = W_k(x + \text{TimeEmb}(\Delta t))$
    -   **Benefit**: Explicitly models the varying time intervals between observations (e.g., frequent vitals vs. sparse labs).

-   **Missingness-Aware Embeddings**:
    -   Input features are concatenated with binary masks ($x_{obs}, m_{obs}$).
    -   The model learns to treat observed and imputed values differently.

-   **Focal BCE Loss**:
    -   Used for the classification head to handle extreme class imbalance (1.8% positive).
    -   Down-weights easy negatives, focusing training on hard positive examples.

## 3. TFT Baseline
The baseline is a standard Transformer Encoder architecture.
-   **Normalization**: Standard `LayerNorm`.
-   **Attention**: Standard Scaled Dot-Product Attention (no time-gap injection).
-   **Loss**: Standard BCE (can be swapped for Focal for fair comparison).

## 4. Input & Output
-   **Input**: Sequence of shape `[Batch, Seq_Len, Features]`.
    -   Features include: Standardized Vitals/Labs + Missingness Masks + Time Deltas.
-   **Outputs**:
    1.  **Classification**: Sepsis Risk Score (0-1) at each time step.
    2.  **Forecasting**: Predicted values for the next time step (Auxiliary task).
