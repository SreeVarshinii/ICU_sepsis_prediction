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

### Core Frameworks
*   **Python**: Chosen for its dominance in Data Science and extensive library ecosystem.
*   **PyTorch**: Selected over TensorFlow for its dynamic computation graph, which simplifies the implementation of complex custom layers like **Temporal Attention** and **Dynamic Tanh**.

### Data Processing
*   **Pandas & NumPy**: Essential for efficient manipulation of the structured tabular data.
*   **PyArrow (Parquet)**: Used for data storage. Parquet is a columnar format that provides significantly faster I/O and smaller file sizes compared to CSV, which is crucial when handling millions of ICU time steps.

### Visualization & Deployment
*   **Streamlit**: Chosen for the **Interactive Dashboard** because it allows for the rapid conversion of Python scripts into web apps without needing HTML/CSS/JS expertise. It natively supports PyTorch tensors and Matplotlib figures.
*   **Rich**: Used for the **CLI Monitor** to provide beautiful, readable terminal output with color-coded alerts, simulating a real-time backend log.

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

To demonstrate the system's capability in a realistic setting, we present a **Real-time Sepsis Monitoring Suite**.

### 5.1. Demo Components

1.  **Clinician Dashboard (Streamlit)**:
    *   **Target Audience**: Doctors and Nurses.
    *   **Function**: Provides a visual history of the patient's vitals and the model's predicted risk trajectory.
    *   **Key Feature**: "Risk Forecast" — showing not just the current risk, but the trend over the last 12 hours.

2.  **Backend Monitor (CLI)**:
    *   **Target Audience**: System Administrators / Central Monitoring Unit.
    *   **Function**: Simulates the processing of a live data stream from bedside monitors.
    *   **Key Feature**: Low-latency inference logging and color-coded **ALERTS** when risk exceeds 50%.

### 5.2. Real-time Operation Workflow

1.  **Data Streaming**: The system reads patient data sequentially, simulating a live feed where one hour of data arrives at a time.
2.  **Preprocessing**:
    *   The new data point is standardized using pre-computed hospital statistics.
    *   Missing values are masked, and the time-gap since the last observation is calculated.
3.  **Inference**:
    *   The **DyT Transformer** processes the updated sequence (history + new point).
    *   It outputs a **Sepsis Risk Score** (0-1) and a **Forecast** for the next 3 hours of vitals.
4.  **Decision**:
    *   If Risk > 0.5: Trigger **HIGH RISK ALERT**.
    *   If Risk > 0.2: Trigger **WARNING**.
    *   Otherwise: Status **STABLE**.

### 5.3. Demo Screenshot
*(Placeholder: Dashboard Screenshot will be inserted here)*

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
