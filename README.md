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

## Quantum Baseline — Photonic Fock-Space Reservoir (No Merlin)

### Configuration

- 10 optical modes  
- 5 photons  
- Hilbert space dimension = 2002  

### Core Idea

- Encode latent factors as phase shifts  
- Apply fixed interferometric unitary  
- Extract photon-number expectations  
- Train only the linear readout  

### Results

| Metric        | Value     |
|--------------|----------:|
| Surface MSE  | 2.64e-4   |
| Surface RMSE | 1.62e-2   |

### Improvement vs Classical ESN

- ~1.9× lower MSE  
- ~28% reduction in RMSE  

The quantum reservoir provides richer nonlinear mixing via bosonic interference.

---

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
