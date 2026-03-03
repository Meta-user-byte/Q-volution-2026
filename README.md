# QML Model for Option Pricing  
### Team Qubiteers  
[Ankit Sharma](https://www.linkedin.com/in/ankit-sharma-3733a7144/) · 
[Soham Pawar](https://www.linkedin.com/in/soham-pawar-b6881a251/) · 
[Quách Hoa Mai](https://www.linkedin.com/in/quachhoamai14012000/) · 
[Abdullah K](https://www.linkedin.com/in/abdullah-aab-a-kk/)

---


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
***For full information, please read the PDF file.***
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

## Description of Model 1 - Recurrent Quantum Reservoir Computing (Feedback - Driven)

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
## Model-2  Quantum Reservoir Computing for Volatility Surface Prediction

We implement a **Quantum Reservoir Computing (QRC)** model based on a photonic Fock-space simulation using the `Merlin` framework.

The reservoir consists of:
- $8$ photonic modes,
- $4$ indistinguishable photons,
- fixed entangling interferometric layers,
- angle encoding of input features.

The photonic circuit produces measurement probabilities in Fock space:

$$
\Phi(u_t) \in \mathbb{R}^{D},
$$

where $D = 330$ is the Fock-space dimension for $(8\ \text{modes},\ 4\ \text{photons})$.  
These probabilities serve as high-dimensional nonlinear feature embeddings.

---

### 1. Leaky Quantum Echo-State Dynamics

The reservoir state $z_t \in \mathbb{R}^{D}$ evolves according to a **leaky echo-state update**:

$$
z_{t} = (1 - \epsilon)\, z_{t-1} + \epsilon\, \Phi(u_t),
$$

where:
- $\epsilon$ is the leaking rate,
- $u_t = \bigl[x_t,\ \text{feedback}(z_{t-1})\bigr]$ is the augmented input vector.

This leaky integration ensures **fading memory** while preserving the nonlinear quantum transformations induced by the photonic interferometer.

> **Note:** The reservoir itself is **not trained**. Only the linear readout is optimized.

---

### 2. Linear Ridge Readout

To predict $\Delta x_t$, we train a **ridge regression readout**:

$$
W = \left(Z^\top Z + \lambda I \right)^{-1} Z^\top Y,
$$

where:
- $Z$ contains the reservoir states,
- $Y$ contains the target increments,
- $\lambda$ is the Tikhonov regularization parameter.

A bias term is appended to the reservoir state before training.  
The prediction is then:

$$
\widehat{\Delta x_t} = W^\top [z_t,\ 1].
$$

PCA inversion and inverse scaling reconstruct the full volatility surface.

---

## Model 3 - Lyapunov-Based Adaptive Regularization

To ensure dynamical stability and prevent amplification of unstable directions, we introduce an **adaptive regularization mechanism** inspired by Lyapunov analysis.

The finite-difference variation of reservoir states is defined as:

$$
\Delta Z_t = z_{t+1} - z_t.
$$

Stacking these differences forms a matrix whose largest singular value:

$$
\sigma_{\max} = \max\ \text{singular value of}\ \Delta Z,
$$

serves as a proxy for dynamical amplification.  
The ridge parameter is then adapted as:

$$
\lambda_{\text{adaptive}} = \lambda_0 \cdot \max(1,\ \sigma_{\max}).
$$

If the reservoir exhibits expansion ($\sigma_{\max} > 1$), additional regularization is imposed.  
This couples **dynamical stability** with **statistical generalization**.

---

### 4. Multi-Step Forecasting

For multi-step forecasting, the model is operated in **autoregressive mode**:

$$
x_{t+1} = x_t + \widehat{\Delta x_t},
$$

feeding predicted outputs back into the reservoir input.

This allows trajectory generation in PCA space and reconstruction of future volatility surfaces.

## Descrption of Model 4: Sequential (Temporal) Processing QRC

The Sequential Processing QRC model represents the most advanced iteration of our hybrid quantum-classical architecture. It utilizes a fixed photonic quantum system as a high-dimensional nonlinear feature extractor, coupled with a classical readout layer to model the complex, low-dimensional dynamics of implied volatility surfaces.

<h3>Architectural Design:</h3>
The model processes standardized and PCA-reduced market factors through a structured quantum-hybrid pipeline:

<ul>
  <li>Dimensionality Reduction: The volatility surface ($n=224$) is projected onto $r=5$ principal components, capturing approximately 99.98% of the total variance.</li>
  <li>Temporal Windowing: A history window (optimized at a length of 7) determines the depth of the data considered for current predictions.</li>
  <li>Quantum Encoding: Latent vectors are angle-encoded into optical phase shifts via photon-number operators. This encoding implicitly introduces high-order nonlinear interactions between input features.</li>
  <li>Reservoir Dynamics: The reservoir utilizes a Rectangular interferometer circuit (Type 2) operating in a bosonic Fock space. The system configuration for optimal performance includes:
    <ul>
      <li>Modes: 5</li>
      <li>Photon Initial State: All ones (e.g., $[1, 1, 1, 1, 1]$)</li>
      <li>Input Scaling: Type 1</li>
    </ul>
  </li>
  <li>Feature Extraction: High-dimensional nonlinear embeddings are extracted as expectation values of photon number operators, serving as inputs for the classical readout.</li>
</ul>

<h3>Classical Readout Strategies</h3>
We evaluated four distinct readout regressors to map extracted quantum features to predicted latent vectors:

<ul>
  <li>Linear Regression: Tests if the quantum transformation alone provides sufficient linear separation.</li>
  <li>Ridge Regression: Incorporates L2-regularization to improve numerical stability and reduce overfitting.</li>
  <li>Decision Tree: Provides nonlinear piecewise approximations without requiring feature engineering.</li>
  <li>Random Forest: An ensemble-based approach that reduces variance while preserving nonlinear modeling capacity.</li>
</ul>
---
### Final benchmark Results

We executed three novel models, and in which our ---- model beats all baselines.

| Model        | RMSE     |  MAE   |
|--------------|----------:|---------|
| Recurrent QRC model  | 0.0036   |  0.0027 |
| 🥇 Sequential Processing QRC (Ridge Readout) | 0.001962  |     0.001273    |


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
