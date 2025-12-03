# Data Analysis Report: Sepsis Prediction (PhysioNet 2019)

## 1. Dataset Overview
The dataset consists of ICU records from the PhysioNet 2019 Challenge (Sets A & B), merged into a unified parquet file.

- **Total Rows**: ~1.55 Million (Estimated based on missingness counts)
- **Total Patients**: 40,336
- **Features**: Vital signs, Laboratory values, Demographics.
- **Target**: `SepsisLabel` (Binary: 0 = Normal, 1 = Sepsis)

## 2. Class Balance
The dataset is highly imbalanced, which is typical for medical anomaly detection tasks.

- **Sepsis Prevalence (Rows)**: ~1.8% (Time-step level)
- **Sepsis Prevalence (Patients)**: ~7.2% (Patient level)
    - *Note: This indicates that while only 7% of patients develop sepsis, the positive labels are sparse even within their timelines.*

## 3. Missing Values Analysis
Missingness is a critical feature of this dataset. Vital signs are sampled frequently, while laboratory tests are sparse.

| Feature | Missing Count | Missing % | Type |
| :--- | :--- | :--- | :--- |
| **Fibrinogen** | ~1.5M | **93.5%** | Lab |
| **Bilirubin_total** | ~1.5M | **93.0%** | Lab |
| **TroponinI** | ~1.5M | **90.0%** | Lab |
| **Lactate** | ~1.4M | **85.0%** | Lab |
| **EtCO2** | ~1.5M | **96.3%** | Vital (Sparse) |
| **Temp** | ~1.0M | **66.2%** | Vital |
| **HR** | ~153K | **9.9%** | Vital (Frequent) |
| **MAP** | ~193K | **12.5%** | Vital (Frequent) |
| **O2Sat** | ~202K | **13.1%** | Vital (Frequent) |

> [!NOTE]
> High missingness in labs suggests that **missingness patterns themselves are informative**. For example, a doctor ordering a Lactate test might indicate suspicion of sepsis. We will use missingness masks as features.

## 4. Data Splitting Strategy
To prevent data leakage, we split data at the **Patient level**.

- **Train (70%)**: 28,235 Patients
- **Validation (15%)**: 6,050 Patients
- **Test (15%)**: 6,051 Patients

## 5. Standardization & Preprocessing Plan
Based on the analysis, the following preprocessing steps will be applied:

1.  **Splitting**: Applied first to avoid leakage.
2.  **Standardization**: Z-score normalization ($x' = \frac{x - \mu}{\sigma}$) using **Training Set statistics**.
    - *Mean/Std computed only on observed values.*
3.  **Missingness Handling**:
    - **Imputation**: Forward Fill (carry forward last known value) -> Fill remaining with 0 (Mean after standardization).
    - **Masking**: Create binary mask channels (1 = Observed, 0 = Missing) for the model to learn sampling patterns.
4.  **Sequence Generation**:
    - Sliding windows (e.g., max 200 steps) for Transformer input.
    - Padding for shorter sequences.

## 6. Next Steps
- Implement the `SepsisDataset` class to apply these transformations on the fly or pre-compute them.
- Proceed to model training with the defined splits.
