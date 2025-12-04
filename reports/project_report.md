# Sepsis Prediction with Dynamic Tanh (DyT) Transformer

## 1. Abstract
This project develops a deep learning framework for the early prediction of Sepsis in ICU patients using the PhysioNet 2019 Challenge dataset. We propose a novel **Dynamic Tanh (DyT) Transformer** architecture designed to handle the specific challenges of Electronic Health Record (EHR) data: irregular sampling, high missingness, and non-stationary distribution of vital signs. We compare this proposed model against a strong baseline, the **Temporal Fusion Transformer (TFT)** with standard Layer Normalization. The system is evaluated on its ability to predict sepsis onset 6 hours in advance (Classification) and forecast future vital signs for the next 1-3 hours (Multi-step Regression).

## 2. Problem Statement
Sepsis is a life-threatening condition caused by the body's extreme response to an infection. Early detection is critical for patient survival, as mortality increases significantly with every hour of delayed treatment.

**Challenges in ICU Data:**
*   **Irregular Sampling**: Vital signs (HR, BP) are measured frequently (hourly), while lab tests (Lactate, WBC) are sparse (daily or ad-hoc).
*   **Missingness**: The majority of entries in the data matrix are missing (e.g., Bilirubin is missing >90% of the time).
*   **Non-Stationarity**: A patient's physiological state can change rapidly, shifting the statistical distribution of input features.
*   **Class Imbalance**: Sepsis is a rare event in the timeline (prevalence ~1.8% of time steps), making standard training objectives ineffective.

**Objective:** Develop a model that effectively leverages sparse, irregular multivariate time-series to predict sepsis risk with high sensitivity and sufficient lead time.

## 3. Related Work
*   **PhysioNet 2019 Challenge**: Top solutions often relied on ensemble methods (XGBoost, LGBM) with extensive feature engineering. Deep learning approaches (LSTMs) showed promise but struggled with long-term dependencies and irregular sampling.
*   **Transformers in Healthcare**: Standard Transformers (Vaswani et al.) have been adapted for EHR data (e.g., BEHRT), but often rely on standard Layer Normalization, which assumes a fixed distribution of features across the sequence.
*   **Temporal Fusion Transformers (TFT)**: Lim et al. introduced TFT for interpretable multi-horizon forecasting, using gating mechanisms and variable selection. It serves as our primary baseline.
*   **Dynamic Normalization**: Recent works suggest that adapting normalization statistics (e.g., Adaptive Instance Norm in style transfer) can help in non-stationary domains. Our **DyT** layer builds on this concept for time-series.

## 4. Proposed Architecture: DyT-based Transformer
Our model introduces three key innovations to the standard Transformer Encoder architecture:

### 4.1. Dynamic Tanh (DyT) Normalization
Standard LayerNorm standardizes inputs using fixed learnable parameters ($\gamma, \beta$). In contrast, DyT adapts the scaling non-linearly based on the input's current statistics.

$$
\mu = \text{mean}(x), \quad \sigma = \text{std}(x)
$$
$$
\alpha = \text{MLP}([\mu, \sigma])
$$
$$
y = \gamma \cdot \tanh(\alpha \cdot \frac{x - \mu}{\sigma}) + \beta
$$

**Significance**: This allows the model to dynamically "squash" or expand the feature space when it detects sudden physiological shifts (e.g., onset of shock), preventing gradient saturation and improving convergence.

### 4.2. Temporal Attention
We modify the Self-Attention mechanism to explicitly account for irregular time gaps ($\Delta t$) between observations.

$$
\text{TimeEmb} = \text{Linear}(\Delta t)
$$
$$
Q = W_q(x + \text{TimeEmb}), \quad K = W_k(x + \text{TimeEmb})
$$

**Significance**: This allows the attention mechanism to weigh recent observations differently from distant ones, not just based on sequence position but on actual elapsed time.

### 4.3. Missingness-Aware Embeddings
Instead of simple imputation, we concatenate the input features $x$ with a binary mask $m$ indicating observed values.
$$
x_{input} = [x_{imputed} || m_{observed}]
$$
**Significance**: The model learns that a "missing" Lactate value is fundamentally different from a "normal" Lactate value, capturing the informative nature of medical ordering patterns.

### 4.4. Focal Loss
To handle the 1:50 class imbalance, we use Focal Binary Cross Entropy Loss:
$$
FL(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t)
$$
This down-weights easy negatives (healthy patients) and focuses learning on hard positives (sepsis cases).

## 5. Evaluation Methodology & Results

### 5.1. Experimental Setup
*   **Dataset**: PhysioNet 2019 (40,336 patients).
*   **Splitting**: Hospital-based Cross-Validation (Train: 70%, Val: 15%, Test: 15%) to ensure generalization to new facilities.
*   **Baselines**:
    *   **TFT (LayerNorm)**: Standard Transformer with LayerNorm.
    *   **DyT (Proposed)**: Transformer with DyT and Temporal Attention.

### 5.2. Metrics
*   **AUROC (Area Under ROC)**: Measures discrimination capability.
*   **AUPRC (Area Under Precision-Recall)**: Critical for imbalanced datasets; measures positive predictive value.
*   **Early Warning Lead Time**: The average time difference between the first correct model prediction (Risk > threshold) and the actual sepsis onset label.

### 5.3. Quantitative Results
The following plots demonstrate the performance comparison between the proposed DyT Transformer and the TFT Baseline.

#### ROC Curve Comparison
![ROC Curve](/c:/Users/varsh/analytics/github/Sepsis%20pred/results/roc_comparison.png)
*Figure 1: Receiver Operating Characteristic (ROC) curves for DyT vs. TFT.*

#### Precision-Recall Curve Comparison
![PRC Curve](/c:/Users/varsh/analytics/github/Sepsis%20pred/results/prc_comparison.png)
*Figure 2: Precision-Recall Curves (PRC) for DyT vs. TFT.*

**Analysis:**
The DyT model demonstrates improved discrimination (higher AUROC) and better positive predictive value (higher AUPRC) compared to the baseline, validating the hypothesis that dynamic normalization helps in capturing sepsis onset patterns.

## 6. Limitations & Future Scope

### 6.1. Limitations
*   **Computational Cost**: The attention mechanism is $O(L^2)$, which limits the sequence length (currently truncated to 200 steps).
*   **Imputation Bias**: Simple forward-filling is used for the values; while masks help, more advanced imputation (e.g., Gaussian Processes) could be explored.
*   **Black Box Nature**: While Attention weights offer some interpretability, the DyT scaling factors are complex to visualize clinically.

### 6.2. Future Scope
*   **Clinical Deployment**: Integrate the model into a mock EHR stream to test real-time inference latency.
*   **Multimodal Integration**: Incorporate clinical notes (text) using LLMs alongside the time-series data.
*   **Federated Learning**: Train across multiple hospitals without sharing raw patient data to preserve privacy.
