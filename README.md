# STLD-Net: Spatial–Temporal Label Decoupling Network for Multi-label ECG Classification

**STLD-Net** is a deep learning framework designed for **multi-label ECG classification**.  
It leverages both **spatial** and **temporal** information in ECG signals, while explicitly modeling **label dependencies**.

---

## Key Features

- **Temporal Modeling:**  
  Uses **Bi-directional LSTM (Bi-LSTM)** to capture temporal dependencies in ECG sequences.

- **Spatial Modeling:**  
  Uses **Vision Transformer (ViT)** to extract spatial patterns across multiple ECG leads.

- **Label Decoupling:**  
  Decouples labels to reduce interference among correlated labels, improving multi-label prediction accuracy.

- **Label Co-occurrence:**  
  Captures co-occurrence relationships among ECG labels for better prediction of rare or co-dependent conditions.

---

## Datasets

STLD-Net has been evaluated on the following **two multi-label ECG datasets**:

1. **Dataset A (e.g., PTB-XL):**  
   - Public 12-lead ECG dataset  
   - Multi-label annotations of 17 cardiac conditions  
   - Sampling rate: 500 Hz, signal length varies (~10 seconds)  

2. **Dataset B (e.g., CPSC 2018):**  
   - Multi-lead ECG recordings from hospital patients  
   - Multi-label classification task with 9–16 cardiac conditions  
   - Provides high-quality ECG signals for model training and evaluation  

> Both datasets are used to validate STLD-Net’s ability to model temporal and spatial features while handling label correlations effectively.

---

## Installation

```bash
git clone https://github.com/your-repo/STLD-Net.git
cd STLD-Net
pip install -r requirements.txt
