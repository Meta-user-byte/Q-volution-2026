# QML Model for Option Pricing  
### Team Qubiteers  
Ankit · Soham · Mai · Abdullah K  

---

## Overview

This repository contains our work for **Q-volution Hackathon 2026**, where we develop and compare:

- A classical **Echo State Network (ESN)**
- A photonic **Quantum Reservoir (Fock-space implementation, without Merlin)**

The objective is to forecast the **next-day implied volatility surface** of interest rate swaptions and evaluate whether quantum-enhanced feature extraction improves performance over classical reservoir computing.

---

## Dataset

We use the **Quandela Swaptions Challenge (Level 1)** dataset:

👉 https://huggingface.co/datasets/Quandela/Challenge_Swaptions

Each row corresponds to a full implied volatility surface indexed by:

- **Maturity (Expiry)**
- **Tenor**

The task is a high-dimensional time-series forecasting problem:
Predict tomorrow’s surface from past observations.

---

## How to Download the Dataset

### Using `datasets` 

Install:

```bash
pip install datasets
from datasets import load_dataset

ds = load_dataset("Quandela/Challenge_Swaptions", split="train")
df = ds.to_pandas()
```

## Method Summary

### Preprocessing

- Remove the `Date` column  
- Standardize volatility surfaces  
- Apply PCA and retain `r = 5` components  
- One-step-ahead forecasting setup  
- Chronological 80% train / 20% test split  

PCA explains **~99.98% of variance**, confirming strong low-dimensional structure.

---

## Classical Baseline — Echo State Network (ESN)

### Architecture

- Reservoir size: `N = 400`  
- Spectral radius `< 1` (ensures echo state property)  
- `tanh` activation  
- Linear readout trained with Adam  

### Results

| Metric        | Value     |
|--------------|----------:|
| Surface MSE  | 5.00e-4   |
| Surface RMSE | 2.24e-2   |

The ESN captures dominant nonlinear dynamics in latent space and serves as a strong classical benchmark.

---

### Description of Model 1 - Recurrent Quantum Reservoir Computing (Feedback - Driven)

- Step 1 — Preprocess Data
Standardize volatility surfaces and apply PCA.
Reduce 224 dimensions to 5 latent factors.

- Step 2 — Define Forecast Target
Model the change in PCA factors (residuals).
Next factor = current factor + predicted change.

- Step 3 — Encode Input
Encode PCA factors as phase shifts in a photonic circuit.
Use 10 modes and 5 photons.

- Step 4 — Quantum Feature Extraction
Pass encoded input through a fixed interferometer.
Measure photon statistics to obtain nonlinear features.

- Step 5 — Introduce Recurrence
Include memory in the system.
• Without Merlin: memory in quantum state.
• With Merlin: memory via classical feedback.

- Step 6 — Train Readout
Use ridge regression to map quantum features to predicted residuals.
Only the readout layer is trained.

---
### Final benchmark Results

We executed three novel models, and in which our ---- model beats all baselines.

| Model        | RMSE     |  MAE   |
|--------------|----------:|---------|
| Recurrent QRC model  | 0.0036   |  0.0027 |
| ------- | 0.0010  |    0.00035    |


## Diagnostics Included

We provide:

- PCA true vs predicted trajectories  
- Surface error distribution histogram  
- Single surface point time evolution  
- Multi-step forecast stability  

### Observations

- Strong one-step predictive performance  
- Improved surface reconstruction vs ESN  
- Multi-step rollout may drift (no stability constraint)  
- Raw Fock simulation is computationally heavy  

---

## Computational Complexity

Quantum Fock implementation requires:

- Matrix permanent computation → factorial scaling  
- Dense state evolution → `O(dim²)` with `dim = 2002`  

This makes the raw implementation significantly slower than ESN and motivates optimized quantum toolchains.

---

## Installation

### Core packages

```bash
pip install numpy pandas matplotlib scikit-learn torch
pip install datasets
pip install perceval-quandela
