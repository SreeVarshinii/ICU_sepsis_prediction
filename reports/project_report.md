# Project Report: Early Sepsis Prediction in ICU

## 1. Introduction

### Project Overview
This project focuses on the **early prediction of Sepsis** in Intensive Care Unit (ICU) patients using advanced Deep Learning techniques. Sepsis is a life-threatening condition that arises when the body's response to an infection causes injury to its own tissues and organs.

### Problem Statement
Early detection of sepsis is critical for patient survival. However, ICU data is characterized by **irregular sampling** (vitals are checked at different intervals) and **missing values**. Traditional models often struggle to capture the complex temporal dynamics of patient health trajectories under these conditions. The goal is to build a model that can predict sepsis onset hours in advance, allowing for timely intervention.

## 2. Proposed Solution

We propose the **Dynamic Transformer (DyT)**, a novel neural network architecture designed specifically for irregular time-series data.

**Key Innovations:**
*   **Temporal Attention**: A mechanism that weighs the importance of past observations based on the actual time gaps, not just sequence order.
*   **Dynamic Tanh**: A specialized activation function that adapts to the irregularity of the data, stabilizing gradients and improving learning on sparse inputs.

## 3. Tools and Technologies Used

*   **Core Framework**: Python, PyTorch
*   **Data Processing**: Pandas, NumPy
*   **Evaluation**: Scikit-learn (AUROC, AUPRC)
*   **Visualization & Demos**:
    *   **Streamlit**: For the interactive web dashboard.
    *   **Rich/ANSI**: For the real-time CLI monitor.
    *   **Matplotlib/Seaborn**: For static plotting and analysis.

## 4. Architecture Diagram

The following diagram illustrates the data flow through the DyT model:

```mermaid
graph LR
    Input[Input Vitals & Time Gaps] --> Embed[Feature Embedding]
    Embed --> PosEnc[Positional Encoding]
    PosEnc --> Layer1[DyT Layer 1]
    
    subgraph DyT_Layer [DyT Layer Structure]
        direction TB
        Norm1[Dynamic Tanh Norm] --> Attn[Temporal Attention]
        Attn --> Add1[Add & Norm]
        Add1 --> FFN[Feed Forward Network]
        FFN --> Add2[Add & Norm]
    end
    
    Layer1 --> DyT_Layer
    DyT_Layer --> LayerN[DyT Layer N]
    LayerN --> Head[Classification Head]
    Head --> Output[Sepsis Risk Score (0-1)]
    
    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style Output fill:#9f9,stroke:#333,stroke-width:2px
    style DyT_Layer fill:#e1f5fe,stroke:#01579b,stroke-dasharray: 5 5
```

## 5. Live Demo

To demonstrate the functionality and performance of the system, we have developed three distinct modules:

1.  **Interactive Patient Dashboard (Streamlit)**:
    *   A web-based interface for clinicians to review patient history.
    *   Visualizes the **Risk Trajectory** alongside key vitals (Heart Rate, BP).
    *   Highlights the "Early Warning" period before clinical onset.

2.  **Real-time ICU Monitor (CLI)**:
    *   A simulation of a bedside monitor.
    *   Streams patient data in real-time.
    *   Triggers **ALERTS** when the predicted risk crosses a critical threshold.

3.  **Comparative Notebook**:
    *   A technical deep-dive comparing DyT against the TFT baseline.
    *   Showcases specific patient cases where DyT successfully predicts sepsis while the baseline fails.

## 6. Evaluation

The model was evaluated on a held-out test set using standard binary classification metrics.

**Performance Metrics:**

| Model | AUROC | AUPRC |
| :--- | :--- | :--- |
| **DyT (Proposed)** | **0.9024** | **0.1837** |
| TFT (Baseline) | 0.7001 | 0.0571 |

*   **AUROC (Area Under Receiver Operating Characteristic)**: Indicates the model's ability to distinguish between sepsis and non-sepsis states. A score of **0.90** represents excellent discrimination.
*   **AUPRC (Area Under Precision-Recall Curve)**: Critical for imbalanced datasets like sepsis prediction. DyT significantly outperforms the baseline (0.18 vs 0.05).

## 7. Results

The results demonstrate that the DyT model provides reliable early warnings.

*   **Risk Trajectories**: In positive cases, the model's risk score typically begins to rise **4-6 hours** before the clinical onset of sepsis.
*   **False Alarms**: The Dynamic Tanh mechanism helps reduce noise-induced false alarms compared to standard RNNs or Transformers.

*(Note: Screenshots of the Dashboard and CLI monitor from the Live Demo section should be inserted here during the presentation.)*

## 8. Limitations and Future Scope

### Limitations
*   **Data Imbalance**: Despite improvements, the low prevalence of sepsis makes achieving high Precision difficult (AUPRC is 0.18).
*   **Computational Cost**: The attention mechanism scales quadratically with sequence length, limiting the lookback window (currently 200 steps).

### Future Scope
*   **Explainability**: Integrating SHAP values to tell clinicians *which* vital sign triggered the alarm.
*   **Multimodal Data**: Incorporating lab results and clinical notes (NLP) to improve accuracy.
*   **Deployment**: Containerizing the application (Docker) for hospital deployment.
